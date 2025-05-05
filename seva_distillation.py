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
OUTPUT_DIR = "./distillation_exp"
STUDENT_SNAPSHOT = None
DISCRIMINATOR_SNAPSHOT = None
START_EPOCH = 1
ACCUMULATION_STEPS = 8
WARMUP_STEPS = 500

def add_noise(
    self,
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
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



class Trainer():
    def __init__(self, local_rank, global_rank):

        self.local_rank = local_rank
        self.global_rank = global_rank
        
        ## Initalize the models
        self.prepare_models()

        ## Initalize training data & loaders
        self.prepare_data()

        ## Initalize optimizer
        self.prepare_optimizers()
        
        ## Initalize DDP training
        self.prepare_DDP_modules()
     
    
    def prepare_DDP_modules(self):
        self.student_model = DDP(self.student_model, device_ids=[self.local_rank], find_unused_parameters = True)
        self.discriminator =  DDP(self.discriminator, device_ids=[self.local_rank], find_unused_parameters=True)
        # self.student_model = torch.compile(self.student_model, dynamic=False)
        # self.disciminator = torch.compile(self.discriminator, dynamic=False)

    def prepare_models(self):
        self.student_model = SGMWrapper(load_model(device="cpu", verbose=True))
        seva_copy = deepcopy(self.student_model.module)
        self.discriminator = Discriminator_Seva(seva_copy)
        if STUDENT_SNAPSHOT is not None:
            state_dict = torch.load(STUDENT_SNAPSHOT, map_location="cpu")
            self.student_model.module.load_state_dict(state_dict)
        if DISCRIMINATOR_SNAPSHOT is not None:
            state_dict = torch.load(DISCRIMINATOR_SNAPSHOT, map_location="cpu")
            self.discriminator.load_state_dict(state_dict)

        ## Send to local_rank
        self.student_model.to(self.local_rank)
        self.discriminator.to(self.local_rank)

        ## Initialize 
        self.discretization = DDPMDiscretization()
        self.denoiser =  DiscreteDenoiser(discretization=self.discretization, num_idx=1000, device=self.local_rank)
        self.sigmas = self.discretization(1, device= self.local_rank)

        ## Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
             "stabilityai/stable-diffusion-2-1-base", subfolder='scheduler'
        )
        self.noise_scheduler.add_noise = add_noise.__get__(self.noise_scheduler, DDPMScheduler)

        self.ts_D_choices = torch.tensor(np.arange(0, 750), device=self.local_rank).long()

        ## If it's rank 0, also init a VAE to decode example frames
        if self.local_rank == 0:
            self.vae = AutoEncoder(chunk_size=1) # VAE on cpu for now
        
        

    def prepare_data(self):
        self.train_dataset = SevaDataset(DATA_PATH, T)
        sampler = DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=1, 
            num_workers=1, 
            shuffle=False, 
            drop_last=True, 
            pin_memory=True, 
            sampler=sampler,
            collate_fn=lambda x: x[0] if BATCH_SIZE == 1 else None
        )
        


    def prepare_optimizers(self):
        self.student_optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=STUDENT_LR, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)
        
        self.student_scheduler = get_scheduler(
            "constant",
            optimizer=self.student_optimizer,
            num_warmup_steps= WARMUP_STEPS,
        )


        self.discriminator_optimizer = torch.optim.AdamW(
            [p[1] for p in self.discriminator.named_parameters() if "disc_head" in p[0]], 
            lr=DISCRIMINATOR_LR, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08
        )

        self.discriminator_scheduler = get_scheduler(
            "constant",
            optimizer=self.discriminator_optimizer,
            num_warmup_steps= WARMUP_STEPS,
        )

        # Grad scaler for mixed precision training
        self.scaler = GradScaler(enabled=True)

    
    def send_batch_to_gpu(self, batch):
        x_in = batch["x_in"].to(self.local_rank, non_blocking = True)
        cond = {k:v.to(self.local_rank, non_blocking = True) for k, v in batch["cond"].items()}
        x_gt = batch["x_gt"].to(self.local_rank, non_blocking = True)
        test_indices = batch["test_indices"].to(self.local_rank, non_blocking=True)
        return {"x_in": x_in, "cond": cond, "x_gt": x_gt, "test_indices": test_indices}

    def run_batch(self, x_in, cond, x_gt, test_indices, step, accumulation_steps = 8):
        x_in *=  torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        s_in = x_in.new_ones([x_in.shape[0]])
        
        first_sigma = self.sigmas[0] * s_in

        ## Denoised version from student model
        with autocast( enabled=self.scaler.is_enabled()):
            x_pred = self.denoiser(self.student_model, x_in, first_sigma, cond, num_frames = T)
        
        ## Renoise both teacher and student output
        renoise_t = self.ts_D_choices[torch.randint(0, len(self.ts_D_choices), (x_gt.shape[0], ), device=self.local_rank)]
        x_student_re = self.noise_scheduler.add_noise(x_pred, torch.randn_like(x_gt), renoise_t)
        x_teacher_re = self.noise_scheduler.add_noise(x_gt, torch.randn_like(x_gt), renoise_t)

        x_teacher_re = torch.cat((x_teacher_re, cond["concat"]),1)
        x_student_re = torch.cat((x_student_re, cond["concat"]),1)
        with autocast( enabled=self.scaler.is_enabled()):
            real_logits = self.discriminator(x = x_teacher_re.detach(), t = renoise_t,    y=cond["crossattn"], dense_y=cond["dense_vector"], num_frames = T)[test_indices]
            fake_logits = self.discriminator(x = x_student_re.detach(), t = renoise_t,    y=cond["crossattn"], dense_y=cond["dense_vector"], num_frames = T)[test_indices]

        ## Discriminator step
        discriminator_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + 
                    F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                )
        self.scaler.scale(discriminator_loss).backward()
        if step == accumulation_steps:
            self.scaler.unscale_(self.discriminator_optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.scaler.step(self.discriminator_optimizer)
            self.discriminator_scheduler.step()
            self.discriminator_optimizer.zero_grad()

        ## Generator step
        with autocast( enabled=self.scaler.is_enabled()):
            fake_logits  = self.discriminator(x = x_student_re, t = renoise_t,  y=cond["crossattn"], dense_y=cond["dense_vector"], num_frames = T)[test_indices]

        adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        recon_loss = F.smooth_l1_loss(x_pred[test_indices], x_gt[test_indices])
        student_loss = adv_loss + recon_loss

        self.scaler.scale(student_loss).backward()
        if step == accumulation_steps:
            self.scaler.unscale_(self.student_optimizer)  # Unscale before gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
            self.scaler.step(self.student_optimizer)
            self.student_scheduler.step()
            self.student_optimizer.zero_grad()
            self.scaler.update()
        
        return student_loss.item(), discriminator_loss.item(), x_pred
    

    def train(self):
        self.student_model.module.module.train()
        self.discriminator.module.train()
        if self.local_rank == 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        step_count = 1
        for epoch in range(START_EPOCH - 1, START_EPOCH -1 + EPOCHS):
            self.train_loader.sampler.set_epoch(epoch)
            if self.local_rank == 0:
                train_iter = tqdm(self.train_loader, unit='batch', dynamic_ncols=True)
                train_iter.set_description(f'[TRAIN] Epoch {epoch+1}/{EPOCHS}')
                student_losses = []
                discriminator_losses = [] 
            else:
                train_iter = self.train_loader
            
            for i,batch in enumerate(train_iter):
                batch_gpu = self.send_batch_to_gpu(batch)
                
                student_loss, discriminator_loss, x_pred = self.run_batch(**batch_gpu, step = step_count, accumulation_steps = ACCUMULATION_STEPS)
                
                if step_count == ACCUMULATION_STEPS:
                    step_count = 1
                else:
                    step_count += 1


                if self.local_rank == 0:
                    train_iter.set_postfix(student_loss=f"{student_loss:.4f}", discriminator_loss=f"{discriminator_loss:.4f}")
                    student_losses.append(student_loss)
                    discriminator_losses.append(discriminator_loss)

                if self.local_rank == 0 and i % LOG_EVERY == 0:
                    ## Log student snapshot
                    student_snapshot = f"{OUTPUT_DIR}/student_snapshot.pth"
                    torch.save(self.student_model.module.module.state_dict(), student_snapshot)


                    ## Log discriminator snapshot
                    discriminator_snapshot = f"{OUTPUT_DIR}/discriminator_snapshot.pth"
                    torch.save(self.discriminator.module.state_dict(), discriminator_snapshot)

                    ## Output example images
                    trg_idx = batch["test_indices"][0]
                    pred = x_pred[[[trg_idx]]].detach().cpu()
                    gt = batch["x_gt"][[[trg_idx]]].detach().cpu()
                
                    ## Decode teacher and student outputs to pixel space (on CPU)
                    with torch.no_grad():
                        pred_px = self.vae.decode(pred, 1)
                        gt_px = self.vae.decode(gt, 1)

                    # Convert student prediction
                    pred_px = (pred_px.permute(0, 2, 3, 1) + 1) / 2.0
                    pred_px = (pred_px[0] * 255).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()

                    # Convert teacher prediction
                    gt_px = (gt_px.permute(0, 2, 3, 1) + 1) / 2.0
                    gt_px = (gt_px[0] * 255).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()

                    # Concatenate horizontally
                    combined = np.concatenate([pred_px, gt_px], axis=1)
                    imageio.imwrite(f"{OUTPUT_DIR}/sample_epoch_{epoch+1:03d}_step_{i:06d}.jpg", combined)

                
            ## Save after epoch
            if self.local_rank == 0:
                print(f"[TRAIN] Epoch {epoch+1}/{EPOCHS} | Avg Student Loss: {sum(student_losses)/len(student_losses):.6f} | Avg Discriminator Loss: {sum(discriminator_losses)/len(discriminator_losses):.6f}")
                student_snapshot = f"{OUTPUT_DIR}/student_ckpt_{epoch+1:03d}.pth"
                torch.save(self.student_model.module.module.state_dict(), student_snapshot)

                discriminator_snapshot = f"{OUTPUT_DIR}/discriminator_ckpt_{epoch+1:03d}.pth"
                torch.save(self.discriminator.module.state_dict(), discriminator_snapshot)
    
 

def ddp_setup(local_rank):
    
    init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=100000))
    torch.cuda.set_device(local_rank)
    

def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    ddp_setup(local_rank)

    ## Set up the trainertrainer
    trainer = Trainer(local_rank, global_rank)
    barrier() # syncs processes here

    ## Start training
    trainer.train()
    destroy_process_group()

if __name__ == "__main__":
    main()
