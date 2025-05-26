import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch

# UNTESTED
# Sinusoidal Encoding
class SinusoidalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, coords):
        position = coords.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / self.dim))
        pe = torch.zeros(coords.shape[0], coords.shape[1], self.dim)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        return pe # (num_nodes, 2, dim)
        # return statement might be this instead:
        # return pe.view(coords.shape[0], -1)  # output: (num_nodes, 2 * dim)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# features that will be passed (FOR NOW):
# node_in: h and w of cell/cluster
# edge in: relative positions of pins to center of cells/cluster, 
# future for edge: type (ff, non-ff)

class DiffusionModel(nn.Module):
    def __init__(self, node_dim=2, edge_dim=4, hidden_dim=128, block_count=2):
        super().__init__()
        self.block_count = 2
        self.pos_encoder = SinusoidalEncoding(32)
        self.input_mlp = nn.Linear(node_dim + 32, hidden_dim)
        # instantiate blocks
        self.blocks = nn.ModuleList()
        for _ in range(block_count):
            self.blocks.append(nn.ModuleDict({
                'gnn1': GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim), # ccheck args
                'gnn2': GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim), # ccheck args
                'mlp1': MLP(hidden_dim, 4 * hidden_dim, hidden_dim),
                'gnn3': GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim), # ccheck args
                'attn1': nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True),
                'gnn4': GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim), # ccheck args
                'attn2': nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True), # check args
                'mlp2': MLP(hidden_dim, 4 * hidden_dim, hidden_dim),
            })
            )
        # final layer
        self.out = nn.Linear(hidden_dim, 2)  # Predict noise on 2D positions

    def forward(self, x, edge_index, edge_attr):
        pos_enc = self.pos_encoder(x)
        # added
        if pos_enc.dim() == 3 and x.dim() == 2:
            pos_enc = pos_enc.view(-1, pos_enc.size(-1))  # flatten batch and node dims
        # added
        x = torch.cat([x, pos_enc], dim=-1) # how does this make sense?
        x = self.input_mlp(x)
        for block in self.blocks:
            x = block['gnn1'](x, edge_index, edge_attr)
            x = block['gnn2'](x, edge_index, edge_attr)
            x = x + block['mlp1'](x)
            x = block['gnn3'](x, edge_index, edge_attr)
            attn_output, _ = block['attn1'](x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x = attn_output.squeeze(0)
            x = block['gnn4'](x, edge_index, edge_attr)
            attn_output, _ = block['attn2'](x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x = attn_output.squeeze(0)
            x = block['mlp2'](x)
        return self.output_layer(x)