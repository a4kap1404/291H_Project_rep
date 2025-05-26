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

# loss function
mse_loss = nn.MSELoss()

# note how the training loop does not appear to take constraints, and only during guidenace does it
# confirm that is how its supposed to work

# def train_ddpm(model, dataloader, optimizer, noise_schedule, num_steps=1000, num_epochs=50):
#     model.train()
#     for epoch in range(num_epochs):
#         for batch in dataloader:
#             x0 = batch.x
#             t = torch.randint(0, num_steps, (x0.size(0),), device=x0.device)
#             noise = torch.randn_like(x0)
#             xt = noise_schedule.q_sample(x0, t, noise)
#             pred_noise = model(xt, batch.edge_index, batch.edge_attr)
def train_ddpm(model, dataloader, optimizer, noise_schedule, num_steps=1000, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            x0 = batch.x
            t = torch.randint(0, num_steps, (x0.size(0),), device=x0.device)
            noise = torch.randn_like(x0)
            xt = noise_schedule.q_sample(x0, t, noise)
            pred_noise = model(xt, batch.edge_index, batch.edge_attr)
            loss = mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def guided_sampling(model, noise_schedule, graph, steps=1000, w_hpwl=1e-4, w_legality=1.0, w_congest=1.0):
    x = torch.randn(graph.num_nodes, 2).to(next(model.parameters()).device)
    for t in reversed(range(steps)):
        with torch.no_grad():
            pred_noise = model(x, graph.edge_index, graph.edge_attr)
            x0_hat = noise_schedule.predict_x0(x, t, pred_noise)
            # compute guidance gradients
            grad_hpwl = compute_hpwl_gradient(x0_hat, graph)
            grad_legality = compute_legality_gradient(x0_hat, graph)
            grad = w_hpwl * grad_hpwl + w_legality * grad_legality # need to add congesiton
            x = noise_schedule.p_sample(x, t, pred_noise, guidance_grad=grad)
    return x

def compute_hpwl(x, graph):
    """
    Compute the Half-Perimeter Wirelength (HPWL) for 2-pin nets.
    Args:
        x: Tensor of shape [num_nodes, 2] with (x, y) positions.
        graph: A graph object with .edge_index of shape [2, num_edges].
    Returns:
        hpwl: Scalar tensor representing the total HPWL.
    """
    src, dst = graph.edge_index  # shape: [num_edges]
    x_src = x[src]  # shape: [num_edges, 2]
    x_dst = x[dst]  # shape: [num_edges, 2]
    # Compute Manhattan (L1) bounding box size for each 2-pin net
    hpwl_per_edge = (x_src - x_dst).abs().sum(dim=1)  # shape: [num_edges]
    total_hpwl = hpwl_per_edge.sum()
    return total_hpwl


def compute_hpwl_gradient(x, graph):
    """
    Compute the gradient of HPWL w.r.t. x, i.e., ∇ₓ HPWL(x).
    Args:
        x: Tensor of shape [num_nodes, 2] with requires_grad=True
        graph: A graph object with .edge_index
    Returns:
        grad: Tensor of shape [num_nodes, 2] — the gradient of HPWL
    """
    x = x.clone().detach().requires_grad_(True)
    hpwl = compute_hpwl(x, graph)
    grad = torch.autograd.grad(hpwl, x, retain_graph=True)[0]
    return grad

def compute_legality(x, graph, cell_width=1.0, cell_height=1.0):
    """
    Compute the legality penalty as the sum of squared overlaps.
    Only penalizes when cells i and j overlap.
    Args:
        x: Tensor of shape [num_cells, 2], each row is (x, y)
        graph: Not used here but included for compatibility
        cell_width: width of a cell (assumed fixed)
        cell_height: height of a cell (assumed fixed)
    Returns:
        legality: scalar tensor representing total overlap penalty
    """
    num_cells = x.shape[0]
    penalty = 0.0
    for i in range(num_cells):
        for j in range(i + 1, num_cells):  # avoid duplicate pairs and self-pairs
            xi, yi = x[i]
            xj, yj = x[j]
            dx = (cell_width - torch.abs(xi - xj))
            dy = (cell_height - torch.abs(yi - yj))
            if dx > 0 and dy > 0:
                # overlapping in both x and y => overlapping rectangle
                overlap = dx * dy
                penalty += overlap  # or: overlap ** 2 if you want squared area
    return penalty


def compute_legality_gradient(x, graph, cell_width=1.0, cell_height=1.0):
    """
    Compute the gradient of the legality loss w.r.t. x.
    Args:
        x: Tensor of shape [num_cells, 2]
        graph: not used
        cell_width: width of a cell
        cell_height: height of a cell
    Returns:
        grad: Tensor of shape [num_cells, 2] — gradient of legality loss
    """
    x = x.clone().detach().requires_grad_(True)
    legality = compute_legality(x, graph, cell_width, cell_height)
    grad = torch.autograd.grad(legality, x, retain_graph=True)[0]
    return grad

# class PlacementDataset(Dataset):
#     def __init__(self, pkl_file):
#         with open(pkl_file, 'rb') as f:
#             self.data = pickle.load(f)  # list of (chip, G)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         chip, G = self.data[idx]

#         # Extract standard cell positions as a tensor [num_std_cells, 2]
#         positions = []
#         for cell in chip.std_cells:
#             positions.append([cell.x, cell.y])
#         positions = torch.tensor(positions, dtype=torch.float32)

#         # Extract edges as a tensor [num_edges, 2]
#         # Networkx edges are pairs of node indices (u,v)
#         # We'll convert them to a tensor of shape (E, 2)
#         edges = torch.tensor(list(G.edges), dtype=torch.long)

#         return {
#             'positions': positions,
#             'edges': edges
#         }


# class PlacementDataset(Dataset):
#     def __init__(self, pkl_file):
#         with open(pkl_file, 'rb') as f:
#             self.data = pickle.load(f)

#     def __len__(self):
#         return len(self.data)

#     # def __getitem__(self, idx):
#     #     chip, G = self.data[idx]

#     #     # Node features (positions of standard cells)
#     #     positions = torch.tensor([[cell.x, cell.y] for cell in chip.std_cells], dtype=torch.float)

#     #     # Edges as tensor of shape (2, num_edges) for PyG
#     #     # Convert edges list [(u,v), ...] to a tensor of shape [2, E]
#     #     edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

#     #     # If you have edge attributes, create edge_attr here
#     #     # For now, let's just create dummy edge_attr as ones with shape (num_edges, edge_feature_dim)
#     #     edge_attr = torch.ones(edge_index.size(1), 4)  # example with edge_dim=4

#     #     data = Data(x=positions, edge_index=edge_index, edge_attr=edge_attr)

#     #     return data

#     def __getitem__(self, idx):
#         chip, G = self.data[idx]

#         positions = torch.tensor([[cell.x, cell.y] for cell in chip.std_cells], dtype=torch.float)

#         edges = list(G.edges)
#         if len(edges) > 0:
#             edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#             edge_attr = torch.ones(edge_index.size(1), 4)
#         else:
#             edge_index = torch.empty((2, 0), dtype=torch.long)
#             edge_attr = torch.empty((0, 4), dtype=torch.float)

#         return Data(x=positions, edge_index=edge_index, edge_attr=edge_attr)



class PlacementDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chip, G = self.data[idx]

        positions = []
        for cell in chip.std_cells:
            positions.append([cell.x, cell.y])
        x = torch.tensor(positions, dtype=torch.float)

        # edge_index = torch.tensor(list(G.edges)).t().contiguous()  # shape [2, E]
        edge_index = torch.tensor(list(G.edges)).t().contiguous()  # shape [2, E]

        # example edge attributes (if none, create ones)
        # edge_attr = torch.ones(edge_index.size(1), 4)  # edge_dim=4
        edge_attr = torch.ones(edge_index.size(1), 4)  # edge_dim=4


        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionModel(node_dim=2, edge_dim=4, hidden_dim=64, block_count=2).to(device)

# Create dataset and dataloader
dataset = PlacementDataset('Data.pkl')
# dataset = create_dummy_dataset(num_graphs=50, num_nodes=20)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Instantiate optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Instantiate noise schedule
noise_schedule = LinearNoiseSchedule(timesteps=100, beta_start=1e-4, beta_end=0.02)

# MSE loss
mse_loss = torch.nn.MSELoss()

print("training model...")

# Training loop
train_ddpm(model, dataloader, optimizer, noise_schedule, num_steps=100, num_epochs=2)

print("done training model")
