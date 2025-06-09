import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import pickle

from model import *
from model_utils import *

from datagen import *
from train_utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unset = 1

node_dim = 2
edge_dim = 4
# pos_dim = 48
pos_dim = 8
# time_dim = 1
# time_dim = 32
# time_dim = 32
# time_dim = 16
# hidden_dim = 32
hidden_dim = 128
# hidden_dim = 256
time_dim = hidden_dim
block_count = 2
# block_count = 1
# x_encode_dim = 16
x_encode_dim = 48

# training_data_path = "./syn_data/Data_N100_v0.pkl"
# training_data_path = "./syn_data/Data_N2_v0.pkl"
training_data_path = "./syn_data/Data.pkl"

# timesteps = 1000
# timesteps = 20 # seems to produce less horrible results during inference
timesteps = 200
beta_start = 1e-4
# beta_start = 1e-8
beta_end = 0.02
# beta_end = 0.04

epochs = 30
# lr = 1e-3 # standard
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
print("length: ", len(dataset))
print(dataset.data[0])
# dataset = create_dummy_dataset(num_graphs=50, num_nodes=20)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # size of 1 is neccesary for now
# print()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# noise schedule
noise_schedule = LinearNoiseSchedule(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)

# loss
mse_loss = torch.nn.MSELoss()

print("training model...")

# training
train_ddpm(model, dataloader, optimizer, noise_schedule, num_steps=timesteps, num_epochs=epochs, debug=debug)

print("done training model")

# testing inference
print("testing inference...")
# tune these
w_hpwl = 1e-4
w_legality = 1e-4
w_m_legality = 1e-3 # this one cant be violated either
w_bound_legality = 1e-3 # this SHOULD BE WEIGHTED HIGHLY, as we SHOULD HAVE no violation of this
# PUT CHECKS IN PLACE in guided_sampling to ensure it does not violatd,
# maybe even put a manual corrector if by end, we still see violations
grad_w_list = [w_hpwl, w_legality, w_m_legality, w_bound_legality]

guidance_scale = 1

test_iter = iter(dataloader) # currently on training data
batch = next(test_iter)
print(f"input_shape: {batch.x.shape}")
out = guided_sampling(model, noise_schedule, batch, timesteps, grad_w_list, guidance_scale)
print(f"outptut shape: {out.shape}")
# check to see if constraints are met
macro_legality = compute_macro_legality(out, batch)
boundary_legality = compute_boundary_legality(out, batch)
assert(macro_legality == 0)
assert(boundary_legality == 0)

# testing model shaping forward method
# NOTE: MODEL CANNOT NORMAL DO BATCHES, change this and re-run
# note: when a batch of 4 is sent, it flattens to 2d,s
# NOTE: instead, just make it 2d, and just send in 1 graph for now**
# B = 4 # CANT DO THIS
# num_nodes = 100
# num_edges = 234
# output = model(
#     x=torch.ones([num_nodes, 2], dtype=torch.float),
#     edge_index=torch.ones([2, num_edges], dtype=torch.int),
#     edge_attr=torch.ones([num_edges, edge_dim], dtype=torch.float),
#     # t=torch.ones([1,1]),
#     t=torch.tensor([1], dtype=torch.float),
#     node_attr=torch.ones([num_nodes, node_dim], dtype=torch.float)
# )
# print(output.shape)
# exit()



# remove later
# for batch in dataloader:
#     print(batch)
#     print("batch.x shape:", batch.x.shape)
#     print("batch.edge_index shape:", batch.edge_index.shape)
#     print("batch.edge_attr shape:", batch.edge_attr.shape)
#     print("batch.batch shape:", batch.batch.shape)  # tells you which graph each node belongs to
#     exit()