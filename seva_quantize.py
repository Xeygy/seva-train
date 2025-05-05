import modelopt.torch.quantization as mtq
from seva_dataset import SevaDataset
from seva.sampling import DDPMDiscretization, MultiviewCFG, DiscreteDenoiser_Q
from torch.utils.data import DataLoader, Subset
from seva.utils import load_model
from seva.model import *
import random

## PARAMS
T = 8
DATA_PATH = "/home/brandon/data"

# Setup the model
model = SGMWrapper_Q(load_model(device="cpu", verbose=True))
discretization = DDPMDiscretization()
denoiser =  DiscreteDenoiser_Q(discretization=discretization, num_idx=1000)
sigmas = discretization(1)
s_in = torch.ones([T])
sigma = sigmas[0] * s_in


# Select quantization config
config = mtq.INT8_SMOOTHQUANT_CFG

# Quantization need calibration data. Setup calibration data loader
# An example of creating a calibration data loader looks like the following:
dataset= SevaDataset(DATA_PATH, T)
indices = random.sample(range(len(dataset)), 500) ## Get 500 random scenes 
subset = Subset(dataset, indices)
data_loader = DataLoader(subset, batch_size = 1,   collate_fn=lambda x: x[0])


# Define forward_loop. Please wrap the data loader in the forward_loop
def forward_loop(model):
    for batch in data_loader:
        x_in = batch["x_in"]
        
        ## Inputs from the model
        cond = batch["cond"]
        x_in *=  torch.sqrt(1.0 + sigma ** 2.0)

        ## Descrite denoiser will call the model
        denoised =  denoiser(model, x_in, sigma, cond, num_frames = T)



# Quantize the model and perform calibration (PTQ)
model = mtq.quantize(model, config, forward_loop)

sample_batch =  next(iter(data_loader))


## Save to ONNX
torch.onnx.export(
    model,                  # model to export
    (sample_batch["x_in"], 
    sigma,
    sample_batch["cond"]["concat"],
    sample_batch["cond"]["crossattn"],
    sample_batch["cond"]["dense_vector"]),        # inputs of the model,
    "seva_quantized.onnx",        # filename of the ONNX model
    input_names=["x_in", "sigma", "concat", "crossattn", "dense_vector"],
    output_names=["x_out"],
    dynamic_axes=None
)

