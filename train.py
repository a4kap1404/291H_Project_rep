import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import pickle

from model import *
from py_utils.model_utils import *

from datagen import *
from py_utils.train_utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:", device)

# change this if you generate your own dataset
training_data_path = "./syn_data/Data_N100_v0.pkl" # was being used
# training_data_path = "./syn_data/Data_N2_v0.pkl"
# training_data_path = "./syn_data/Data.pkl"

test_model_after_training = False
# test_model_after_training = True # should be False
model_dir = "models"
modelname = "placer_model" # will save model
model_path = model_dir + "/" + modelname + ".pkl"

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
 
node_dim = 2
edge_dim = 4
pos_dim = 48
# pos_dim = 8
# time_dim = 1
# time_dim = 32
# time_dim = 32
# time_dim = 16
# hidden_dim = 32
hidden_dim = 256
# hidden_dim = 256
time_dim = hidden_dim
# block_count = 2
block_count = 4
# block_count = 1
# x_encode_dim = 16
x_encode_dim = 48

# timesteps = 1000
timesteps = 20 # very low to keep inference time reasonable given no clustering
# timesteps = 50 # very low to keep inference time reasonable given no clustering
beta_start = 1e-4
beta_end = 0.02

epochs = 5
lr = 1e-3

debug = False
# debug = True

model = DiffusionModel(
    node_dim=node_dim,
    edge_dim=edge_dim,
    pos_dim=pos_dim,
    time_dim=time_dim,
    hidden_dim=hidden_dim,
    block_count=block_count,
    x_encode_dim=x_encode_dim
    ).to(device)


# Create dataset and dataloader
dataset = PlacementDataset(training_data_path)
print("N: ", len(dataset))
print(dataset.data[0])
# dataset = create_dummy_dataset(num_graphs=50, num_nodes=20)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # size of 1 is neccesary for now
# print()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# noise schedule
noise_schedule = LinearNoiseSchedule(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end, device=device)

# loss
mse_loss = torch.nn.MSELoss()

print("training model...")

# training
train_ddpm(model, dataloader, optimizer, noise_schedule, num_steps=timesteps, num_epochs=epochs, device=device, debug=debug)

print("done training model, now saving")

with open(model_path, "wb") as f:
    pickle.dump(model, f)


print("model saved")

if test_model_after_training:
    # testing inference
    print("testing inference...")
    # tune these
    w_hpwl = 1e-4
    w_legality = 1e-4
    w_m_legality = 8e-4 # this one cant be violated either
    w_bound_legality = 1e-4 # this SHOULD BE WEIGHTED HIGHLY, as we SHOULD HAVE no violation of this
    # PUT CHECKS IN PLACE in guided_sampling to ensure it does not violatd,
    # maybe even put a manual corrector if by end, we still see violations
    
    # dont change these names unless in py_utils.train_utils.guided_sampling also change
    grad_weights = {
        "w_hpwl": w_hpwl,
        "w_legality": w_legality,
        "w_m_legality": w_m_legality, 
        "w_bound_legality": w_bound_legality
    }

    for weight in grad_weights:
        grad_weights[weight] *= 1000 / timesteps

    guidance_scale = 0.2
    tanh_threshold = 1

    test_iter = iter(dataloader) # currently on training data
    batch = next(test_iter)
    print(f"input_shape: {batch.x.shape}")
    out = guided_sampling(model, noise_schedule, batch, timesteps, grad_weights, guidance_scale, tanh_threshold, device)
    print(f"outptut shape: {out.shape}")
    print("out:", out)
    # check to see if constraints are met
    macro_legality = compute_macro_legality(out, batch)
    boundary_legality = compute_boundary_legality(out, batch)
    if (macro_legality > 0):
        print("macro_legality violation")
    if (boundary_legality > 0):
        print("boundary_legality violation")

    print(f"hpwl (normalized): {compute_hpwl(out, batch)}")