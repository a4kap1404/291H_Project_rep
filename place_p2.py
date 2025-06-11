"""
    Description:
        Loads output from place_p1.py, i.e. runs inference, outputs placement, and prints time of inference
"""

# this is a non odb script!

import networkx as nx

from py_utils.datagen import *
from py_utils.train_utils import *
import time

def import_odb_data(filename):
    with open(filename, "rb") as f:
        (cell_map, cells, macros, edges, pin_locs, chip_w, chip_h) = pickle.load(f)
    return (cell_map, cells, macros, edges, pin_locs, chip_w, chip_h)
        # cell_map = pickle.load(f)
        # return cell_map
    # return (cell_map, cells, macros, edges, pin_locs, chip_w, chip_h)


def get_macros_cells_and_graph(filename):
    (cell_map, cells, macros, edges, pin_locs, chip_w, chip_h) = import_odb_data(filename)

    # edges_and_pin_locs = []
    # for i in range(len(edges)):
    #     edges_and_pin_locs.append([
    #         edges[i][0], edges[i][1], 
    #         pin_locs[i][0][0], pin_locs[i][0][1],
    #         pin_locs[i][1][0], pin_locs[i][1][1],
    #     ])


    # print(len(edges))

    # filter edges b/c using Digraph in model
    filtered_edges = []
    filtered_pin_locs = []
    keep_indxs = set()
    seen_edges = set()
    for i in range(len(edges)):
        edge = tuple(edges[i])
        if edge not in seen_edges:
            seen_edges.add(edge)
            keep_indxs.add(i)
    # print(len(keep_indxs))
    keep_indxs = sorted(list(keep_indxs))
    print(len(keep_indxs))
    for idx in keep_indxs:
        filtered_edges.append(edges[idx])
        filtered_pin_locs.append(pin_locs[idx])


    # instantiate pins on cells
    filtered_rel_pin_nums = []
    for i in range(len(filtered_edges)):
        src_id = filtered_edges[i][0]
        trgt_id = filtered_edges[i][1]

        src_pin_idx = None
        trgt_pin_idx = None

        # src
        found_pin = False
        for p, pin in enumerate(cells[src_id].pin_slots):
            if (pin.dx, pin.dy) == tuple(filtered_pin_locs[i][0]):
                found_pin = True
                src_pin_idx = p
                break
        if not found_pin:
            cells[src_id].pin_slots.append(PinSlot(
                filtered_pin_locs[i][0][0],
                filtered_pin_locs[i][0][1],
            ))
            src_pin_idx = len(cells[src_id].pin_slots) - 1
        
        # trgt
        found_pin = False
        for p, pin in enumerate(cells[trgt_id].pin_slots):
            if (pin.dx, pin.dy) == tuple(filtered_pin_locs[i][1]):
                found_pin = True
                trgt_pin_idx = p
                break
        if not found_pin:
            cells[trgt_id].pin_slots.append(PinSlot(
                filtered_pin_locs[i][1][0],
                filtered_pin_locs[i][1][1],
            ))
            trgt_pin_idx = len(cells[trgt_id].pin_slots) - 1
        
        filtered_rel_pin_nums.append((src_pin_idx, trgt_pin_idx))

    # create graph
    G = nx.DiGraph()
    for i in range(len(cells)):
        G.add_node(i, pos=(cells[i].x, cells[i].y))
    for i in range(len(filtered_edges)):
        G.add_edge(
            filtered_edges[i][0],
            filtered_edges[i][1],
            src_rel_pin=filtered_rel_pin_nums[i][0],
            trgt_rel_pin=filtered_rel_pin_nums[i][1]
        )
    
    return macros, cells, G, cell_map, chip_h, chip_w


def getInitPlacement(large_net_threshold=10):
    # block = ord.get_db_block()
    macros, cells, G, cell_map, chip_h, chip_w = get_macros_cells_and_graph(large_net_threshold)
    # chip_w = block.getCoreArea().dx() 
    # chip_h = block.getCoreArea().dy()

    max_1_half = max(chip_h, chip_w) / 2
    # normalize all chip cell positons in graph G
    for node in G.nodes:
        x = G.nodes[node]['pos'][0]
        y = G.nodes[node]['pos'][1]
        G.nodes[node]['pos'] = (x/max_1_half, y/max_1_half)

    # normalize everything else
    ios = [] # model does not really incoporate them
    norm_chip_h, norm_chip_w, macros, cells, ios = normalize_dimensions(
    chip_h, chip_w, macros, cells, ios)

    chip = ChipArea(norm_chip_w, norm_chip_h, macros, cells, ios)
    chip = center_normalized_placement(chip)
    
    chip.nn_width = chip_w
    chip.nn_height = chip_h

    return chip, G, cell_map

class PlacementDataset_Inf(torch.utils.data.Dataset):
    def __init__(self, chip, G):
            self.data = ((chip, G),)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print("accessing idx: ", idx)
        chip, G = self.data[idx]

        positions = torch.tensor([[cell.x, cell.y] for cell in chip.std_cells], dtype=torch.float)
        
        # node features (subject to change)
        node_attr = torch.tensor([[cell.width, cell.height] for cell in chip.std_cells], dtype=torch.float)

        edge_index = torch.tensor(list(G.edges)).t().contiguous()  # shape [2, E]

        edge_attr = []
        for u, v, attr in G.edges(data=True):
            src_pin = chip.std_cells[u].pin_slots[attr["src_rel_pin"]]
            trgt_pin = chip.std_cells[v].pin_slots[attr["trgt_rel_pin"]]
            edge_attr.append([src_pin.dx, src_pin.dy, trgt_pin.dx, trgt_pin.dy])
        edge_attr = torch.tensor(edge_attr)

        # INFERENCE STUFF
        macro_dims = torch.tensor([[macro.width, macro.height] for macro in chip.macros], dtype=torch.float)
        macro_pos = torch.tensor([[macro.x, macro.y] for macro in chip.macros], dtype=torch.float)
        io_pos = torch.tensor([[pin.x, pin.y] for pin in chip.io_pins], dtype=torch.float)


        # Other important stuff
        chip_dims = torch.tensor([chip.width, chip.height], dtype=torch.float)
        nn_chip_dims = torch.tensor([chip.nn_width, chip.nn_height], dtype=torch.int)

        data = Data(x=positions, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr,
                    macro_dims=macro_dims, macro_pos=macro_pos, io_pos=io_pos, 
                    chip_dims=chip_dims, nn_chip_dims=nn_chip_dims)
        return data


if __name__ == '__main__':

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    filename = "odb_placement.pkl" # from place_p1.py

    # adjust if needed
    model_dir = "models"
    modelname = "placer_model" # will save model
    model_path = model_dir + "/" + modelname + ".pkl"

    placement_name = "ml_placed_graph.pkl" # will export

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # normalized and centered
    chip, G, cell_map = getInitPlacement(filename)
    dataset = PlacementDataset_Inf(chip, G)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # size of 1 is neccesary for now
    test_iter = iter(dataloader)
    batch = next(test_iter)

    print("batch.x:", batch.x.size())

    # load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    model.to(device)

    # model params (should not change unless retrain)
    # guidance potential gradient weights
    w_hpwl = 1e-4
    w_legality = 1e-4
    w_m_legality = 1e-3 # this one cant be violated either
    w_bound_legality = 1e-3
    grad_w_list = [w_hpwl, w_legality, w_m_legality, w_bound_legality]
    guidance_scale = 1

    beta_start = 1e-4
    beta_end = 0.02
    tanh_threshold = 0.7

    timesteps = 30 # low due to inference
    noise_schedule = LinearNoiseSchedule(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)

    grad_w_list = [w_hpwl, w_legality, w_m_legality, w_bound_legality]

    # run model
    print("beginning model inference...")
    start = time.perf_counter()
    out = guided_sampling(model, noise_schedule, batch, timesteps, grad_w_list, guidance_scale, tanh_threshold)
    end = time.perf_counter()

    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds")
    print("finished model inference...")

    # check to see if constraints are met
    macro_legality = compute_macro_legality(out, batch)
    boundary_legality = compute_boundary_legality(out, batch)
    if (macro_legality > 0):
        print("macro_legality violation")
    if (boundary_legality > 0):
        print("boundary_legality violation")

    # update std_cell locations
    for i in range(len(chip.std_cells)):
        chip.std_cells[i].x = float(out[i][0])
        chip.std_cells[i].y = float(out[i][1])
    
    # undo centering
    chip = undo_centering_on_norm(chip)

    # undo normalization
    chip = undo_normalization(chip)

    # undo on graph
    # max_1_half = max(chip.nn_height, chip.nn_width) / 2
    # for node in G.nodes:
    #     x = G.nodes[node]['pos'][0]
    #     y = G.nodes[node]['pos'][1]
    #     G.nodes[node]['pos'] = (x * max_1_half, y * max_1_half)

    # export result
    chip_cells = [(cell.width, cell.height, cell.x, cell.y) for cell in chip.std_cells]
    with open(placement_name, "wb") as f:
        pickle.dump(chip_cells, f)

# 🐻 its cozy down here

