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

from py_utils.datagen import *

# loss function
mse_loss = nn.MSELoss()

def train_ddpm(model, dataloader, optimizer, noise_schedule, num_steps=1000, num_epochs=50, debug=False):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            x0 = batch.x
            # print("x0:", x0.shape)
            # old version # t = torch.randint(0, num_steps, (x0.size(0),), device=x0.device)
            t = torch.randint(0, num_steps, (len(batch),)).to(x0.device)
            # print("t (actual)", t)
            # exit()
            noise = torch.randn_like(x0)
            # print("noise:", noise.shape)
            xt = noise_schedule.q_sample(x0, t, noise)
            # print("xt:", xt.shape)
            pred_noise = model(xt, batch.edge_index, batch.edge_attr, t.to(torch.float), batch.node_attr)

            # print(f"pred_noise x[0]:{pred_noise[0]}")
            # print(f"noise x[0]:{noise[0]}")
            # added
            # torch.clamp(pred_noise, min=-1.0, max=1.0)

            # print("pred_noise:", pred_noise.shape)
            loss = mse_loss(pred_noise, noise)
            # print("loss:", loss)
            # exit()
            optimizer.zero_grad()
            loss.backward()

            if (debug):
                # check gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} grad norm: {param.grad.norm().item()}")
                    else:
                        print(f"{name} has no gradient!")

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"avg loss on epoch #{epoch}: {avg_loss:.5f}")

def guided_sampling(model, noise_schedule, graph, steps=1000, grad_weights=None, guidance_scale=1, tanh_threshold=1.0):
    x = torch.randn_like(graph.x).to(next(model.parameters()).device)
    for t in reversed(range(steps)):

        x_t = x.detach().clone().requires_grad_()

        # print(f"--steps:{t} / {steps}--")
        if t % (steps//10) == 0:
            print(f"--steps:{t} / {steps}--")
            print("x[0]:", x_t[0])
        with torch.no_grad():
            pred_noise = model(x_t, graph.edge_index, graph.edge_attr, torch.tensor([t], dtype=torch.float), 
                               graph.node_attr)
            # x0_hat = noise_schedule.predict_x0(x, t, pred_noise)
            
        # i think this should be here
        x0_hat = noise_schedule.predict_x0(x_t, t, pred_noise)

        gradients = {
            "w_hpwl": compute_hpwl_gradient(x0_hat, graph, x_t),
            # "w_legality": compute_legality_gradient(x0_hat, graph, x_t), # comment this back in if you implement cluster
            "w_m_legality": compute_macro_legality_gradient(x0_hat, graph, x_t),
            "w_bound_legalty": compute_boundary_legality_gradient(x0_hat, graph, x_t)
        }
        
        grad = torch.zeros_like(x_t)
        for metric in gradients:
            if gradients[metric] is not None:
                grad += gradients[metric] * grad_weights[metric]

        x = noise_schedule.p_sample_1(x_t, t, pred_noise, guidance_grad=grad, guidance_scale=guidance_scale)
        # not optimal, but have to for now
        if (t / steps) < tanh_threshold:
            # print("f!!")
            x = torch.tanh(x)
            # if grad_list[2] > 0:
                # x = x * 0.8
    
    # added b/c bad performance
    # epsilon = 1e-4
    # x = torch.tanh(x) * (1 - epsilon)
    print("x[0]:", x[0])
    return x


def compute_hpwl(x, graph):
    src, dst = graph.edge_index  # shape: [num_edges]
    x_src = x[src]  # shape: [num_edges, 2]
    x_dst = x[dst]  # shape: [num_edges, 2]
    # Compute Manhattan (L1) bounding box size for each 2-pin net
    hpwl_per_edge = (x_src - x_dst).abs().sum(dim=1)  # shape: [num_edges]
    total_hpwl = hpwl_per_edge.sum()
    return total_hpwl


def compute_hpwl_gradient(x0, graph, xt):
    xt = xt.clone().detach().requires_grad_(True)
    hpwl = compute_hpwl(x0, graph)
    # print("hpwl:", hpwl)
    grad = torch.autograd.grad(hpwl, xt, retain_graph=True, allow_unused=True)[0]
    # print("hpwl_grad:", grad)
    # print("hpwl_grad_shape:", grad.shape)
    # if grad is None:
        # grad = torch.tensor([0,])
    return grad

def compute_legality(x, graph):
    num_cells = x.shape[0]
    # penalty = 0.0
    # penalty = torch.tensor([0.0])
    penalties = []  # list to accumulate differentiable tensors
    for i in range(num_cells):
        for j in range(i + 1, num_cells):  # avoid duplicate pairs and self-pairs
            xi, yi = x[i]
            xj, yj = x[j]
            width_i_half = graph.node_attr[i][0] / 2 
            height_i_half = graph.node_attr[i][1] / 2
            width_j_half = graph.node_attr[j][0] / 2
            height_j_half = graph.node_attr[j][1] / 2

            dx = (width_i_half + width_j_half - torch.abs(xi - xj))
            dy = (height_i_half + height_j_half - torch.abs(yi - yj))

            # print("dx:", dx)
            # exit()

            if dx > 0 and dy > 0:
                # overlap_distance_squared = dx**2 + dy**2
                # penalty += overlap_distance_squared
                penalties.append(dx**2 + dy**2)
    
    if penalties:
        # print(f"cell_to_cell penalites found: {len(penalties)}")
        penalty = torch.stack(penalties).sum()
    else:
        # print("#### no cell_to_cell overlap!!!!####")
        # penalty = torch.zeros(1, device=x.device, dtype=x.dtype, requires_grad=True, allow_unused=True)
        penalty = torch.zeros((1,), dtype=x.dtype, device=x.device, requires_grad=True)
        # penalty = torch.zeros((1,), device=x.device, dtype=x.dtype, requires_grad=True, allow_unused=True)
        
    # print("penalty size:", penalty.size())
    return penalty

def compute_macro_legality(x, graph):
    num_cells = x.shape[0]
    num_macros = graph.macro_dims.shape[0]
    # print(f"x.shape:{x.shape}")
    # exit()
    # penalty = 0.0
    # penalty = torch.tensor([0.0])
    penalties = []  # list to accumulate differentiable tensors
    for c in range(num_cells):
        for m in range(num_macros):  # avoid duplicate pairs and self-pairs
            xc, yc = x[c]
            xm, ym = graph.macro_pos[m]
            width_c_half = graph.node_attr[c][0] / 2 
            height_c_half = graph.node_attr[c][1] / 2
            width_m_half = graph.macro_dims[m][0] / 2
            height_m_half = graph.macro_dims[m][1] / 2

            # print(xc.size())
            # print(height_c_half.size())

            dx = (width_c_half + width_m_half - torch.abs(xc - xm))
            dy = (height_c_half + height_m_half - torch.abs(yc - ym))
            # print(dx.size())
            # print(penalty.size())
            # exit()
            if dx > 0 and dy > 0:
                # overlap_distance_squared = dx**2 + dy**2
                # penalty += overlap_distance_squared
                penalties.append(dx**2 + dy**2)
    
    if penalties:
        # print(f"macro_to_cell penalites found: {len(penalties)}")
        penalty = torch.stack(penalties).sum()
    else:
        # print("#### no macro_to_cell overlap!!!!####")
        # penalty = torch.zeros(1, device=x.device, dtype=x.dtype, requires_grad=True, allow_unused=True)
        penalty = torch.zeros((1,), dtype=x.dtype, device=x.device, requires_grad=True)
        # penalty = torch.zeros((1,), device=x.device, dtype=x.dtype, requires_grad=True, allow_unused=True)

    # print("penalty size:", penalty.size())
    # exit()
    return penalty


def compute_boundary_legality(x, graph):
    num_cells = x.shape[0]
    x_min = -1 *    graph.chip_dims[0] / 2
    x_max =         graph.chip_dims[0] / 2
    y_min = -1 *    graph.chip_dims[1] / 2
    y_max =         graph.chip_dims[1] / 2

    # print(x_min, x_max, y_min, y_max)
    # exit()

    penalties = []

    for i in range(num_cells):
        xi, yi = x[i]
        width_half = graph.node_attr[i][0] / 2
        height_half = graph.node_attr[i][1] / 2
        # x < x_min
        left_violation = x_min - (xi - width_half)
        if left_violation > 0:
            penalties.append(left_violation**2)
        # x > x_max
        right_violation = (xi + width_half) - x_max
        if right_violation > 0:
            penalties.append(right_violation**2)
        # y < y_min
        bottom_violation = y_min - (yi - height_half)
        if bottom_violation > 0:
            penalties.append(bottom_violation**2)
        # y > y_max
        top_violation = (yi + height_half) - y_max
        if top_violation > 0:
            penalties.append(top_violation**2)

    if penalties:
        return torch.stack(penalties).sum()
    else:
        return torch.zeros((1,), dtype=x.dtype, device=x.device, requires_grad=True)

def compute_legality_gradient(x0, graph, xt):
    xt = xt.clone().detach().requires_grad_(True)
    legality = compute_legality(x0, graph)
    assert isinstance(legality, torch.Tensor), "legality must be a torch.Tensor"
    assert legality.requires_grad, "legality must require grad"
    grad = torch.autograd.grad(legality, xt, retain_graph=True, allow_unused=True)[0]
    # if grad is None:
        # grad = torch.tensor([0,])
    return grad

def compute_macro_legality_gradient(x0, graph, xt):
    xt = xt.clone().detach().requires_grad_(True)
    legality = compute_macro_legality(x0, graph)
    assert isinstance(legality, torch.Tensor), "legality must be a torch.Tensor"
    assert legality.requires_grad, "legality must require grad"
    grad = torch.autograd.grad(legality, xt, retain_graph=True, allow_unused=True)[0]
    return grad

def compute_boundary_legality_gradient(x0, graph, xt):
    xt = xt.clone().detach().requires_grad_(True)
    legality = compute_boundary_legality(x0, graph)
    assert isinstance(legality, torch.Tensor), "legality must be a torch.Tensor"
    assert legality.requires_grad, "legality must require grad"
    grad = torch.autograd.grad(legality, xt, retain_graph=True, allow_unused=True)[0]
    return grad

class PlacementDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_file:str):
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print("accessing idx: ", idx)
        chip, G = self.data[idx]

        positions = torch.tensor([[cell.x, cell.y] for cell in chip.std_cells], dtype=torch.float)
        
        # print("cell positions shape:", positions.shape) # [# of cells, 2]

        # node features (subject to change)
        node_attr = torch.tensor([[cell.width, cell.height] for cell in chip.std_cells], dtype=torch.float)


        # print(list(G.edges))
        # exit()
        # edge_index = torch.tensor(list(G.edges)).t().contiguous()  # shape [2, E]
        edge_index = torch.tensor(list(G.edges)).t().contiguous()  # shape [2, E]
        # print("edge_index", edge_index)

        # print("edge_index_size:", edge_index.size())
        # example edge attributes (if none, create ones)
        # edge_attr = torch.ones(edge_index.size(1), 4)  # edge_dim=4
        # edge_attr = torch.ones(edge_index.size(1), 4)  # edge_dim=4


        # edge_attr = torch.tensor([[]])
        # desired_keys = ["src_rel_pin", "trgt_rel_pin"]

        # edge_attr = torch.tensor([[chip.cells]])
        edge_attr = []
        for u, v, attr in G.edges(data=True):
            src_pin = chip.std_cells[u].pin_slots[attr["src_rel_pin"]]
            trgt_pin = chip.std_cells[v].pin_slots[attr["trgt_rel_pin"]]
            edge_attr.append([src_pin.dx, src_pin.dy, trgt_pin.dx, trgt_pin.dy])
        edge_attr = torch.tensor(edge_attr)

        # REMOVE LATER
        # edge_feature_dim = 4 # could change
        # num_edges = edge_index.shape[-1]
        # assert(edge_attr.shape == torch.Size([num_edges, edge_feature_dim]))
        # exit()

        # INFERENCE STUFF
        macro_dims = torch.tensor([[macro.width, macro.height] for macro in chip.macros], dtype=torch.float)
        macro_pos = torch.tensor([[macro.x, macro.y] for macro in chip.macros], dtype=torch.float)
        io_pos = torch.tensor([[pin.x, pin.y] for pin in chip.io_pins], dtype=torch.float)


        # Other important stuff
        chip_dims = torch.tensor([chip.width, chip.height], dtype=torch.float)
        nn_chip_dims = torch.tensor([chip.nn_width, chip.nn_height], dtype=torch.int)


        # print("length of attr_list:", len(attr_list))
        # print("shape of edge_index", edge_index.shape)
        # exit()

        data = Data(x=positions, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr,
                    macro_dims=macro_dims, macro_pos=macro_pos, io_pos=io_pos, 
                    chip_dims=chip_dims, nn_chip_dims=nn_chip_dims)
        return data
