import torch
import numpy as np
from seva.sampling import DDPMDiscretization, MultiviewCFG, DiscreteDenoiser
from seva.utils import load_model
from seva.model import *
from seva.discriminator_amd import Discriminator_Seva
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from copy import deepcopy
import os
import datetime
from torch.distributed import init_process_group, destroy_process_group, barrier
from seva_dataset import SevaDataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import random
from seva.modules.autoencoder import AutoEncoder
from diffusers import (
    DDPMScheduler
)
from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
import imageio
import gc
import math
import csv


import lightning as L
from lightning.pytorch.strategies import FSDPStrategy
from seva.modules.layers import TimestepEmbedSequential


## Seed
random.seed(0)
torch.random.manual_seed(0)
np.random.seed(0)

### Training params
DATA_PATH = "/code/nnguy185/seva_data"
BATCH_SIZE = 1 ## One scene per batch
STUDENT_LR = 1e-6 
DISCRIMINATOR_LR = 1e-6 
CFG = 2.0
EPOCHS = 20
LOG_EVERY = 200
T = 5
OUTPUT_DIR = "./distillation_fsdp"
STUDENT_SNAPSHOT = None
DISCRIMINATOR_SNAPSHOT = None
START_EPOCH = 1

ACCUMULATION_STEPS = 1
MIXED_PRECISION = 'bf16'
USE_FSDP = False

class customDDPMScheduler(DDPMScheduler):
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
        ) -> torch.FloatTensor:

        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod[timesteps==999] = 0
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod[timesteps==999] = 1
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

def write_to_csv(file_path, fieldnames, row):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the row
        writer.writerow(row)

def _get_constant_lambda(_=None):
    return 1

def get_warmup_schedule(optimizer, warmup_steps=1000, multiplier=10.0, last_epoch=-1):
    """
    Create a schedule with linear warmup for `warmup_steps`, then constant LR.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps (int): Number of warmup steps.
        multiplier (float): Final multiplier after warmup.
        last_epoch (int): Last epoch index (used for resuming training).

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return (multiplier * current_step) / warmup_steps
        return multiplier

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_constant_schedule(optimizer, last_epoch = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    return LambdaLR(optimizer, _get_constant_lambda, last_epoch=last_epoch)


class SevaLoggingCallback(L.Callback):
    def __init__(self, output_dir=OUTPUT_DIR, log_every_n_steps=LOG_EVERY):
        super().__init__()
        self.output_dir = output_dir
        self.log_every_n_steps = log_every_n_steps
        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only on main process
        if trainer.is_global_zero and (batch_idx % self.log_every_n_steps == 0) and ("x_pred" in outputs):
            # 1. Save student/teacher comparison image
            x_gt = batch["x_gt"]
            x_pred = outputs["x_pred"]
            test_index = batch["test_indices"][0]

            with torch.no_grad():
                vae = getattr(pl_module, "vae", None)
                if vae is None:
                    # Lazy init VAE on CPU
                    pl_module.vae = AutoEncoder(chunk_size=1)
                    pl_module.vae.eval()

                pred = x_pred[[[test_index]]].cpu()
                gt = x_gt[[[test_index]]].cpu()

                pred_px = (pl_module.vae.decode(pred, 1).permute(0, 2, 3, 1) + 1) / 2.0
                gt_px = (pl_module.vae.decode(gt, 1).permute(0, 2, 3, 1) + 1) / 2.0

                pred_img = (pred_px[0] * 255).clamp(0, 255).to(torch.uint8).numpy()
                gt_img = (gt_px[0] * 255).clamp(0, 255).to(torch.uint8).numpy()

                combined = np.concatenate([pred_img, gt_img], axis=1)
                imageio.imwrite(os.path.join(
                    self.output_dir,
                    f"sample_epoch_{trainer.current_epoch+1:03d}_step_{batch_idx:06d}.jpg"
                ), combined)


    def on_train_epoch_end(self, trainer, pl_module):
        # Save snapshot at epoch end
        if trainer.is_global_zero:
            student_ckpt = os.path.join(self.output_dir, f"student_ckpt_{trainer.current_epoch+1:03d}.pth")
            disc_ckpt = os.path.join(self.output_dir, f"discriminator_ckpt_{trainer.current_epoch+1:03d}.pth")

            torch.save(pl_module.student_model.state_dict(), student_ckpt)
            torch.save(pl_module.discriminator.state_dict(), disc_ckpt)


class Seva_Lightning(L.LightningModule):

    def __init__(self):
        super().__init__()
        ## Initialize 
        self.discretization = DDPMDiscretization()
        self.denoiser =  DiscreteDenoiser(discretization=self.discretization, num_idx=1000)
        self.sigmas = self.discretization(1)

        ## Noise scheduler
        self.noise_scheduler = customDDPMScheduler.from_pretrained(
             "stabilityai/stable-diffusion-2-1-base", subfolder='scheduler'
        )
        self.ts_D_choices = torch.tensor(np.arange(0, 750)).long()
        self.automatic_optimization = False

    def configure_model(self):
        self.student_model = SGMWrapper(load_model(device="cpu", verbose=True))
        seva_copy = deepcopy(self.student_model.module)
        self.discriminator = Discriminator_Seva(seva_copy)
        if STUDENT_SNAPSHOT is not None:
            state_dict = torch.load(STUDENT_SNAPSHOT, map_location="cpu")
            self.student_model.load_state_dict(state_dict)
        if DISCRIMINATOR_SNAPSHOT is not None:
            state_dict = torch.load(DISCRIMINATOR_SNAPSHOT, map_location="cpu")
            self.discriminator.load_state_dict(state_dict)

        
        self.student_model.train()
        self.discriminator.train()
    
        ## Freeze the Seva backbone in the discriminator
        for param in self.discriminator.input_blocks.parameters():
            param.requires_grad = False
        for param in self.discriminator.middle_block.parameters():
            param.requires_grad = False

      

    def configure_optimizers(self):    
        student_opt = torch.optim.AdamW(self.student_model.parameters(), lr=STUDENT_LR)
        disc_opt = torch.optim.AdamW(
            [p for n, p in self.discriminator.named_parameters() if "disc_head" in n],
            lr=DISCRIMINATOR_LR
        )
        return [student_opt, disc_opt], []
    
    def set_requires_grad_generator(self):
        for param in self.student_model.parameters():
            param.requires_grad_ = True
        for param in self.discriminator.parameters():
            param.requires_grad_ = False
    

    def set_requires_grad_discriminator(self):
        for param in self.student_model.parameters():
            param.requires_grad_ = False
        ## Only unfreeze the heads
        for head in self.discriminator.heads:
            for param in head.parameters():
                param.requires_grad_ = True

    

    def run_batch_generator(self, x_in, cond, x_gt, test_indices):
        assert x_in.shape[0] == T
        self.discriminator.eval()
        self.student_model.train()
        self.set_requires_grad_generator()

        x_in *=  torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        s_in = x_in.new_ones([x_in.shape[0]])
    
        first_sigma = self.sigmas[0] * s_in
        x_pred = self.denoiser(self.student_model, x_in, first_sigma, cond, num_frames = T)

        renoise_t = self.ts_D_choices[torch.randint(0, len(self.ts_D_choices), (x_gt.shape[0], ))].to(x_pred.device)
        x_student_re = self.noise_scheduler.add_noise(x_pred, torch.randn_like(x_gt), renoise_t)
        x_student_re = torch.cat((x_student_re, cond["concat"]),1)
       
        fake_logits  = self.discriminator(x = x_student_re, t = renoise_t,  y=cond["crossattn"], dense_y=cond["dense_vector"], num_frames = T)[test_indices]

        adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        recon_loss = F.smooth_l1_loss(x_pred[test_indices], x_gt[test_indices])
        student_loss = adv_loss + 1.0 * recon_loss
        return student_loss, x_pred.detach()
    
    def run_batch_discriminator(self, x_in, cond, x_gt, test_indices):
        assert x_in.shape[0] == T
        self.discriminator.train()
        self.student_model.eval()
        self.set_requires_grad_discriminator()
        with torch.no_grad():
            x_in *=  torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
            s_in = x_in.new_ones([x_in.shape[0]])
        
            first_sigma = self.sigmas[0] * s_in
            x_pred = self.denoiser(self.student_model, x_in, first_sigma, cond, num_frames = T)

            renoise_t = self.ts_D_choices[torch.randint(0, len(self.ts_D_choices), (x_gt.shape[0], ))].to(x_pred.device)
            x_student_re = self.noise_scheduler.add_noise(x_pred, torch.randn_like(x_gt), renoise_t)
            x_teacher_re = self.noise_scheduler.add_noise(x_gt, torch.randn_like(x_gt), renoise_t)

            x_teacher_re = torch.cat((x_teacher_re, cond["concat"]),1)
            x_student_re = torch.cat((x_student_re, cond["concat"]),1)

        
        real_logits = self.discriminator(x = x_teacher_re.detach(), t = renoise_t,    y=cond["crossattn"], dense_y=cond["dense_vector"], num_frames = T)[test_indices]
        fake_logits = self.discriminator(x = x_student_re.detach(), t = renoise_t,    y=cond["crossattn"], dense_y=cond["dense_vector"], num_frames = T)[test_indices]

        discriminator_loss =  (
            F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + 
            F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        )

        return discriminator_loss, x_pred.detach()


    def training_step(self, batch, batch_idx):
        if batch_idx % 2 == 0:
            student_loss, x_pred = self.run_batch_generator(batch["x_in"], batch["cond"], batch["x_gt"], batch["test_indices"])
            self.log("student_loss", student_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.loss = student_loss
            self.phase = "S"
            return {"loss": student_loss, "x_pred": x_pred}
        else:
            discriminator_loss, x_pred = self.run_batch_discriminator(batch["x_in"], batch["cond"], batch["x_gt"], batch["test_indices"])
            self.log("discriminator_loss", discriminator_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.loss = discriminator_loss
            self.phase = "D"
            return {"loss": discriminator_loss, "x_pred": x_pred}
         
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        if self.phase == "S":
            self.manual_backward(self.loss)   
            self.loss = None         
            opt_g.step()
            opt_g.zero_grad()
        if self.phase == "D":
            self.manual_backward(self.loss)
            self.loss = None            
            opt_d.step()
            opt_d.zero_grad()
        


def main():

    ## Lightning module
    lightning_module = Seva_Lightning()
    
    ## Data
    train_dataset = SevaDataset(DATA_PATH, T)  
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=lambda x: x[0] if BATCH_SIZE == 1 else None
    )
    policy = {TimestepEmbedSequential, }
    strategy = FSDPStrategy(cpu_offload=True, auto_wrap_policy=policy)
    # strategy = "fsdp"
    ## Trainer
    trainer = L.Trainer(accelerator="gpu", devices=4, precision=16, strategy=strategy, max_epochs=1, callbacks=[
        SevaLoggingCallback(output_dir=OUTPUT_DIR, log_every_n_steps=LOG_EVERY)
    ])

    trainer.fit(lightning_module, train_loader)

if __name__ == "__main__":
    main()
