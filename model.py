import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch

# UNTESTED
# Sinusoidal Encoding
class SinusoidalEncoding(nn.Module):
    def __init__(self, pos_dim):
        super().__init__()
        self.pos_dim = pos_dim
        assert(pos_dim % 4 == 0)
    # CHECK AFTER
    def forward(self, coords):
        device = coords.device
        # coords: (num_nodes, 2)
        div_range = torch.arange(0, self.pos_dim // 2, 2, dtype=torch.float32).to(device) # (pos_dim / 4)
        div_term = torch.exp(div_range * -(torch.log(torch.tensor(10000.0)) / (self.pos_dim // 2))) # (pos_dim / 4)
        # print("SinEnc position div term:", div_term.shape)

        # B = coords.shape[-3]
        num_nodes = coords.shape[-2]
        # x_coords = coords[:,:, 0] # (B, num_nodes)
        # y_coords = coords[:,:, 1]
        x_coords = coords[:, 0] # (num_nodes)
        y_coords = coords[:, 1]
        # x coords
        # pe_x = torch.zeros(B, num_nodes, self.pos_dim // 2)
        pe_x = torch.zeros(num_nodes, self.pos_dim // 2).to(device)
        # print(x_coords.unsqueeze(-1).shape, div_term.shape)
        # print("coords shape:", coords.shape)
        # pe_x[:,:, 0::2] = torch.sin(x_coords.unsqueeze(-1) * div_term)
        pe_x[:, 0::2] = torch.sin(x_coords.unsqueeze(-1) * div_term).to(device)
        # pe_x[:,:, 1::2] = torch.cos(x_coords.unsqueeze(-1) * div_term)
        pe_x[:, 1::2] = torch.cos(x_coords.unsqueeze(-1) * div_term).to(device)
        # y coords
        # pe_y = torch.zeros(B, num_nodes, self.pos_dim // 2)
        pe_y = torch.zeros(num_nodes, self.pos_dim // 2).to(device)
        # pe_y[:,:, 0::2] = torch.sin(y_coords.unsqueeze(-1) * div_term)
        pe_y[:, 0::2] = torch.sin(y_coords.unsqueeze(-1) * div_term).to(device)
        # pe_y[:,:, 1::2] = torch.cos(y_coords.unsqueeze(-1) * div_term)
        pe_y[:, 1::2] = torch.cos(y_coords.unsqueeze(-1) * div_term).to(device)

        pe = torch.cat((pe_x, pe_y), dim=-1) # (B, num_nodes, pos_dim)
        # assert(pe.shape == torch.Size([B, num_nodes, self.pos_dim])) # remove later
        assert(pe.shape == torch.Size([num_nodes, self.pos_dim])) # remove later
        return pe

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


""" proposed features:
    edge: 
        relative pin positons (normalized) (size of 4)
    node (not including position):
        size of each cell
"""

class DiffusionModel(nn.Module):
    def __init__(self, node_dim=0, edge_dim=4, pos_dim=64, time_dim=1, hidden_dim=128, block_count=2, x_encode_dim=2):
        super().__init__()
        self.loc_dim = 2 # (x,y) feature
        self.node_dim = node_dim # does not include (x,y) feature
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.block_count = block_count
        self.x_encode_dim = x_encode_dim

        # FOR TESTING
        assert time_dim == hidden_dim
        
        self.pos_encoder = SinusoidalEncoding(pos_dim) # does not have to be 32, can really be anything, so long as divis by 4
        if  time_dim > 1:
            self.time_linear = nn.Sequential(
                nn.Linear(1, time_dim),
                nn.GELU()
            )
        
        self.x_encoder = nn.Sequential(
            nn.Linear(2, x_encode_dim),
            nn.GELU()
        )
        
        # could change input_mlp to actually be MLP
        self.input_mlp = nn.Sequential(
            # nn.Linear(self.loc_dim + node_dim + pos_dim + time_dim, hidden_dim),
            nn.Linear(x_encode_dim + node_dim + pos_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.GELU(),
        )


        # remove unused ones later

        # instantiate blocks
        self.blocks = nn.ModuleList()
        for _ in range(block_count):
            self.blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_dim),
                'gnn1': GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim), # ccheck args
                'ln2': nn.LayerNorm(hidden_dim),
                'gnn2': GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim), # ccheck args
                'ln3': nn.LayerNorm(hidden_dim),
                'mlp1': MLP(hidden_dim, 4 * hidden_dim, hidden_dim),
                'ln4': nn.LayerNorm(hidden_dim),
                'gnn3': GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim), # ccheck args
                'ln5': nn.LayerNorm(hidden_dim),
                # 'attn1': nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True),
                'gnn4': GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim), # ccheck args
                # 'attn2': nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True), # check args
                'mlp2': MLP(hidden_dim, 4 * hidden_dim, hidden_dim),
            })
            )
        # final layer
        self.out = nn.Linear(hidden_dim, 2)  # Predict noise on 2D positions

        """
            steps:
                x: [total_nodes, 2] (specifies positions)
                pos(x): [total_nodes, pos_dim] (pos_dim could be same as hidden_dim)
                node_vec: [total_nodes, node_dim] (other features) (need to create befoer calling forward())
                t: [B, 1] --(optional)-> [B, time_dim] --(expansion)-> [total_nodes, time_dim] (could decide to later to expand before concactting )
                feature_vec: concat(x, pos(x), node_vec, time_dim): [total_nodes, 2 + pos_dim + node_dim + time_dim]
                Linear/MLP(2 + pos_dim + node_dim + time_dim, hidden_size)
        """

    def forward(self, x, edge_index, edge_attr, t, node_attr):
        # NOTE: for comments with "B", imagine we remove B dim and thus are sending in 1 graph only

        assert(len(x.shape) == 2)
        assert(len(edge_index.shape) == 2)
        assert(len(edge_attr.shape) == 2)
        assert(len(node_attr.shape) == 2)

        device = x.device

        # B = x.shape[-3] # this is unneeded
        num_nodes = x.shape[-2]

        if self.time_dim > 1:
            time_emb = self.time_linear(t)
        else:
            time_emb = t
        
        # repeat: [B, time_dim] --> [B, num_nodes, time_dim]
        # time_emb = time_emb.unsqueeze(1).repeat(1, num_nodes, 1)
        time_emb = time_emb.unsqueeze(0).repeat(num_nodes, 1).to(device) # [num_nodes, time_dim]
        # print("time_emb size:", time_emb.shape)
        # exit()


        # not happening right now: # [total_nodes, time_dim]
        # time_emb_tot = time_emb[batch]

        pos_enc = self.pos_encoder(x) # [B, num_nodes, pos_dim]
        assert(pos_enc.shape == torch.Size([num_nodes, self.pos_dim]))

        # expand x
        x = self.x_encoder(x)

        # print("")
        # for name, var in [('x', x), ('pos_enc', pos_enc), ('node_attr', node_attr), ('time_emb', time_emb)]:
            # print(f"{name}: {var.shape}")

        # exit()
        # [B, num_nodes, 2 + pos_dim + node_dim + time_dim]: NOT TRUE since x_encoder
        x = torch.cat([x, pos_enc, node_attr, time_emb], dim=-1)
        # print()

        # print("concatted shape:", x.shape)
        # expected_shape = torch.Size([num_nodes, 2 + self.pos_dim + self.node_dim + self.time_dim])
        # print("expected shape:", expected_shape)
        # assert(x.shape == expected_shape)
        # exit()

        x = F.gelu(self.input_mlp(x))

        # exit()

        for block in self.blocks:
            x = block['gnn1'](x, edge_index, edge_attr) + x + time_emb
            x = block['ln1'](x)
            x = torch.relu(x)
            x = block['gnn2'](x, edge_index, edge_attr) + x  + time_emb
            x = block['ln2'](x)
            x = torch.relu(x)
            x = x + block['mlp1'](x) + time_emb
            x = block['gnn3'](x, edge_index, edge_attr) + time_emb
            x = block['ln3'](x)
            x = torch.relu(x)
            # attn_output, _ = block['attn1'](x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0)) # is this right
            # x = attn_output.squeeze(0) + x # modded
            x = block['gnn4'](x, edge_index, edge_attr) + time_emb
            x = block['ln4'](x)
            x = torch.relu(x)
            # attn_output, _ = block['attn2'](x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            # x = attn_output.squeeze(0) + x # modded
            x = block['mlp2'](x) + time_emb
            x = block['ln5'](x)
            x = torch.relu(x)
        
        return self.out(x)