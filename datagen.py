import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass, field
from typing import List, Tuple
import pickle
import numpy as np
import os
# import 

@dataclass
class PinSlot:
    dx: float
    dy: float

    def __str__(self):
        return f"(dx, dy): ({self.dx}, {self.dy})"

# def PinSlotList_str(list)

@dataclass
class Macro:
    width: float
    height: float
    pin_slots: List[PinSlot]
    x: float = field(init=False, default=None)
    y: float = field(init=False, default=None)

    def bounds(self) -> Tuple[float, float, float, float]:
        half_w, half_h = self.width/2, self.height/2
        return (self.x-half_w, self.y-half_h, self.x+half_w, self.y+half_h)

    def absolute_pins(self) -> List[Tuple[float, float]]:
        return [(self.x+p.dx, self.y+p.dy) for p in self.pin_slots]
    
    def __str__(self):
        # _ = " " * 2
        return f"(width,height):({self.width}, {self.height}), pin_slots:{[str(pin) for pin in self.pin_slots]}, (x,y):({self.x}, {self.y})"

@dataclass
class StandardCell:
    width: float
    height: float
    pin_slots: List[PinSlot]
    x: float = field(init=False, default=None)
    y: float = field(init=False, default=None)

    def bounds(self) -> Tuple[float, float, float, float]:
        half_w, half_h = self.width/2, self.height/2
        return (self.x-half_w, self.y-half_h, self.x+half_w, self.y+half_h)

    def absolute_pins(self) -> List[Tuple[float, float]]:
    # def absolute_pins(self) -> List[List[float, float]]:
        return [(self.x+p.dx, self.y+p.dy) for p in self.pin_slots]

    def __str__(self):
        return f"(width,height):({self.width}, {self.height}), pin_slots:{[str(pin) for pin in self.pin_slots]}, (x,y):({self.x}, {self.y})"

@dataclass
class IOPin:
    width: float
    height: float
    x: float = field(init=False, default=None)
    y: float = field(init=False, default=None)

    def bounds(self) -> Tuple[float, float, float, float]:
        half_w, half_h = self.width/2, self.height/2
        return (self.x-half_w, self.y-half_h, self.x+half_w, self.y+half_h)

    def pin_position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __str__(self):
        return f"(width,height):({self.width}, {self.height}),\t(x,y):({self.x}, {self.y})"

class ChipArea:
    def __init__(self, width, height, macros, std_cells, io_pins, rng=None):
        self.width, self.height = width, height
        self.macros = macros
        self.std_cells = std_cells
        self.io_pins = io_pins
        self.rng = rng or random
        # non_normalized variables (designated with "nn")
        self.nn_width = None
        self.nn_height = None
        

    def overlaps_any(self, b, others):
        x1,y1,x2,y2 = b
        for ox1,oy1,ox2,oy2 in others:
            if not (x2<=ox1 or ox2<=x1 or y2<=oy1 or oy2<=y1):
                return True
        return False

    def place_macros(self, max_retries: int = 500) -> None:
        placed_bounds = []
        placed_macros = []
        for idx, m in enumerate(self.macros[:]):
            placed = False
            for attempt in range(max_retries):
                x = self.rng.uniform(m.width/2, self.width - m.width/2)
                y = self.rng.uniform(m.height/2, self.height - m.height/2)
                b = (x - m.width/2, y - m.height/2, x + m.width/2, y + m.height/2)
                if not self.overlaps_any(b, placed_bounds):
                    m.x, m.y = x, y
                    placed_bounds.append(b)
                    placed_macros.append(m)
                    placed = True
                    break
            if not placed:
                # print(f"Warning: could not place macro #{idx} after {max_retries} attempts; skipping.")
                self.macros.remove(m)
                dummy = None
        self.macros = placed_macros

    def place_io(self) -> None:
        for io in self.io_pins:
            side = self.rng.choice(['left','right','top','bottom'])
            if side=='left':
                io.x, io.y = io.width/2, self.rng.uniform(io.height/2, self.height-io.height/2)
            elif side=='right':
                io.x, io.y = self.width-io.width/2, self.rng.uniform(io.height/2, self.height-io.height/2)
            elif side=='top':
                io.x, io.y = self.rng.uniform(io.width/2, self.width-io.width/2), self.height-io.height/2
            else:
                io.x, io.y = self.rng.uniform(io.width/2, self.width-io.width/2), io.height/2

    def place_cells(self, region: float, dthresh: float, max_retries: int = 100, abs_fails_max: int = 15) -> None:
        placed_cells = []
        placed_bounds = []
        absolute_fails = 0
        for idx, c in enumerate(self.std_cells[:]):
            placed = False
            if (absolute_fails > abs_fails_max):
                break
            for attempt in range(max_retries):
                x = self.rng.uniform(c.width/2, self.width - c.width/2)
                y = self.rng.uniform(c.height/2, self.height - c.height/2)
                b = (x - c.width/2, y - c.height/2, x + c.width/2, y + c.height/2)
                if self.overlaps_any(b, [m.bounds() for m in self.macros]):
                    # print("o_m")
                    continue
                if self.overlaps_any(b, placed_bounds):
                    # print("o_c")
                    continue
                if self.density(region, x, y, placed_cells) > dthresh:
                    # print("d")
                    continue
                c.x, c.y = x, y
                placed_cells.append(c)
                placed_bounds.append(b)
                placed = True
                break
            if not placed:
                # print(f"Warning: could not place cell #{idx} after {max_retries} attempts; skipping.")
                self.std_cells.remove(c)
                absolute_fails += 1
                dummy = None
        self.std_cells = placed_cells
        # check for unplaced cells
        for cell in self.std_cells:
            if cell.x is None:
                raise Exception("unplaced cell in placed_cells[]")

    def density(self, region: float, x: float, y: float, placed: List[StandardCell]) -> float:
        half = region/2
        area = region * region
        occ = 0.0
        for c in placed:
            x1,y1,x2,y2 = c.bounds()
            ix1,ix2 = max(x1, x-half), min(x2, x+half)
            iy1,iy2 = max(y1, y-half), min(y2, y+half)
            if ix1 < ix2 and iy1 < iy2:
                occ += (ix2-ix1) * (iy2-iy1)
        return occ / area

    # def pins(self) -> List[Tuple[float, float, int]]:
    def pins(self) -> List[Tuple[float, float, int, int]]:
        ps = []
        # for m in self.macros:
        #     if m.x is not None:
        #         ps.extend(m.absolute_pins())
        cell_num = 0
        for cell in self.std_cells:
            assert(cell.x is not None)
            if cell.x is not None:
                # t = cell.absolute_pins()[0] + (i,)
                # ps.extend([t,])
                cell_pins = cell.absolute_pins() # [(x,y), (x,y)]
                for pin_idx in range(len(cell_pins)):
                    cell_pins[pin_idx] = cell_pins[pin_idx] + (cell_num, pin_idx)
                    # [(x,y,cell_num,pin_idx), ...]
                ps.extend(cell_pins)
            cell_num += 1
        # for io in self.io_pins:
        #     if io.x is not None:
        #         ps.append(io.pin_position())
        # print("length of ps: ", len(ps))
        return ps


def generate_placement(w: float,
                       h: float,
                       macros: List[Macro],
                       cells: List[StandardCell],
                       ios: List[IOPin],
                       region: float,
                       dthresh: float,
                       steps: int = 100,
                       scale: float = 0.2,
                       seed: int = None,
                       max_macro_retries: int = 500,
                       max_cell_retries: int = 500,
                       edge_sample_ratio: int = 300,
                       edge_to_node_ratio: float = 2.5
                       ) -> Tuple[ChipArea, nx.DiGraph]:
    rng = random.Random(seed) if seed is not None else random
    chip = ChipArea(w, h, macros, cells, ios, rng)
    chip.place_macros(max_retries=max_macro_retries)
    print("placed macros")
    chip.place_io()
    print("placed ios")
    chip.place_cells(region, dthresh, max_retries=max_cell_retries)
    print("placed cells")
    for cell in chip.std_cells:
        if cell.x is None:
            raise Exception("unplaced cell in placed_cells[] 2")
    print(len(chip.std_cells))
    # exit()

    G = nx.DiGraph()
    pins = chip.pins()  # [(x,y,cell_num,pin_idx), ...]

    print("# of pins:", len(pins))

    # random.shuffle(pins) # is this needed if we sort?
    # assuming all std_cells are placed
    # nodes_added = [False] * len(chip.std_cells)
    # for (x, y, cell_num, pin_num) in pins:
        # if (nodes_added)
        # G.add_node(idx, pos=(x, y))
    
    # assuming all std_cells are placed
    for cell_idx, cell in enumerate(chip.std_cells):
        G.add_node(cell_idx, pos=(cell.x, cell.y))

    # print("added placed objects to graph, pins # =", len(pins))

    used_targets = set()
    used_sources = set()

    # Build unique undirected pairs sorted by distance
    random.shuffle(pins)
    pairs = [] # should not pairs of pins from same cells
    N = len(pins)
    edge_step_size = N // edge_sample_ratio
    for i in range(N):
        # print(i)
        for j in range(i+1, N, edge_step_size):
            if pins[i][2] == pins[j][2]: # same cell
                continue
            dist = math.hypot(pins[i][0]-pins[j][0], pins[i][1]-pins[j][1])
            pairs.append((i, j, dist))
    
    
    print("# of pairs:", len(pairs))

    pairs.sort(key=lambda t: t[2])

    print("sorted by distance...")
    # print("total:", total)
    # print(pairs[0])
    # exit()

    # this does not look like paper

    selected_edges = biased_shuffle(pairs, scale)
    print("done with biased shuffling...")
    for k in range(int(len(selected_edges) / edge_to_node_ratio)):
        i = selected_edges[k][0]
        j = selected_edges[k][1]
        cell_num_i = pins[i][2]
        cell_num_j = pins[j][2]
        assert(cell_num_i != cell_num_j)
        rel_pin_num_i = pins[i][3]
        rel_pin_num_j = pins[j][3]

        if (i not in used_targets) and (j not in used_targets and j not in used_sources):
            # i is src, j is target
            G.add_edge(cell_num_i, cell_num_j, src_rel_pin=rel_pin_num_i, trgt_rel_pin=rel_pin_num_j)
            used_sources.add(i)
            used_targets.add(j)
        elif (j not in used_targets) and (i not in used_targets and i not in used_sources):
            # j is src, i is target
            G.add_edge(cell_num_j, cell_num_i, src_rel_pin=rel_pin_num_j, trgt_rel_pin=rel_pin_num_i)
            used_sources.add(j)
            used_targets.add(i)

    print("graph edges: ", len(G.edges))
    print("graph nodes: ", len(G.nodes))
    return chip, G


def plot_full(chip: ChipArea, G: nx.DiGraph, show: bool = True) -> None:
    fig, ax = plt.subplots()
    for m in chip.macros:
        if m.x is not None:
            ax.add_patch(plt.Rectangle((m.x-m.width/2, m.y-m.height/2), m.width, m.height,
                                   facecolor='red', alpha=0.4, edgecolor='black'))
    for c in chip.std_cells:
        if c.x is not None:
            ax.add_patch(plt.Rectangle((c.x-c.width/2, c.y-c.height/2), c.width, c.height,
                                   facecolor='blue', alpha=0.4, edgecolor='black'))
    xs = [io.x for io in chip.io_pins]
    ys = [io.y for io in chip.io_pins]
    ax.scatter(xs, ys, marker='s', facecolor='green', edgecolor='black')
    for u, v in G.edges():
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']
        arrow = FancyArrowPatch(posA=(x1, y1), posB=(x2, y2), arrowstyle='-|>',
                                mutation_scale=7, linewidth=1.5, color='red', alpha=0.9,
                                shrinkA=5, shrinkB=5)
        ax.add_patch(arrow)
        arrow.set_zorder(5)
    ax.set_xlim(0, chip.width)
    ax.set_ylim(0, chip.height)
    ax.set_aspect('equal')
    if show:
        plt.savefig('chip_design.png', bbox_inches='tight', dpi=300)

def biased_shuffle(sorted_list, scale=5.0):
    sorted_list = list(sorted_list)
    n = len(sorted_list)
    result = []
    for _ in range(n):
        indices = np.arange(len(sorted_list))
        weights = np.exp(-indices / scale)
        weights /= weights.sum()
        chosen_idx = np.random.choice(len(sorted_list), p=weights)
        result.append(sorted_list.pop(chosen_idx))
    return result

def print_cells_attr(chip: ChipArea, cell_count, macro_count=0, io_count=0):
    print("Cell Areas")
    # for cell in chip.std_cells:
    cell_count = min(cell_count, len(chip.std_cells))
    macro_count = min(macro_count, len(chip.macros))
    io_count = min(io_count, len(chip.io_pins))
    if (cell_count > 0):
        print("--Cell Data--")
        for i in range(cell_count):
            print(chip.std_cells[i])
        # print(f"{{width: {cell.width}, height: {cell.height}, {cell.x}, {cell.y}}}")
    if (macro_count > 0):
        print("--Macro Data--")
        for i in range(macro_count):
            print(chip.macros[i])
    if (io_count > 0):
        print("--IO Data--")
        for i in range(io_count):
            print(chip.io_pins[i])


def pin_list_good(list, object_h, object_w, chip_y_max, chip_y_min, chip_x_max, chip_x_min):
    for pin in list:
        pin_pos_y = pin.dy + object_h
        pin_pos_x = pin.dx + object_w
        if (pin_pos_y < chip_y_min or pin_pos_y > chip_y_max or 
            pin_pos_x < chip_x_min or pin_pos_x > chip_x_max):
            return False
    return True

def pins_on_object_good(object):
    eps = 0.0000001
    for pin in object.pin_slots:
        x_good = (object.width / 2)  >= abs(pin.dx)
        y_good = (object.height / 2) >= abs(pin.dy)
        if not x_good or not y_good:
            # raise Exception(
                # f"w/2:{object.width/2}, h/2:{object.height/2}, {pin.dx, pin.dy}, {x_diff, y_diff}")
            return False
    return True

def object_pos_good(object, chip_y_max, chip_y_min, chip_x_max, chip_x_min):
        x_min = object.x - object.width/2
        x_max = object.x + object.width/2
        y_min = object.y - object.height/2
        y_max = object.y + object.height/2
        return not (x_min < chip_x_min or y_min < chip_y_min or 
                    x_max > chip_x_max or y_max > chip_y_max)

def chip_good_uncentered(chip: ChipArea, debug=False):
    # checking cells
    for cell in chip.std_cells:
        if not object_pos_good(cell, chip.height, 0, chip.width, 0):
            # print("failed cell ")
            if debug: raise Exception("chip_good_error")
            return False
        if not pin_list_good(cell.pin_slots, cell.height, cell.width, chip.height, 0, chip.width, 0):
            if debug: raise Exception("chip_good_error")
            return False
        if not pins_on_object_good(cell):
            if debug: raise Exception("chip_good_error")
            return False
    # checking macros
    for cell in chip.macros:
        if not object_pos_good(cell, chip.height, 0, chip.width, 0):
            if debug: raise Exception("chip_good_error")
            return False
        if not pin_list_good(cell.pin_slots, cell.height, cell.width, chip.height, 0, chip.width, 0):
            if debug: raise Exception("chip_good_error")
            return False
        if not pins_on_object_good(cell):
            if debug: raise Exception("chip_good_error")
            return False
    # checking pins (only centers)
    for pin in chip.io_pins:
        if (pin.y < 0 or pin.y > chip.height or pin.x < 0 or pin.x > chip.width):
            if debug: raise Exception("chip_good_error")
            return False
    return True


def estimate_num_cells_and_area(density, chip_h, chip_w, c_h, c_w, m_h, m_w):
    # estimate num of macros and cells
    cell_to_macro_total_area_ratio = 8
    area_multiplier = 1 - (1/cell_to_macro_total_area_ratio)
    cell_area = c_h * c_w
    num_cells = int(density * chip_w * chip_h * area_multiplier / cell_area)
    macro_area = m_h * m_w
    if macro_area == 0:
        num_macros = 0
    else:
        num_macros = int(num_cells * cell_area * (1/cell_to_macro_total_area_ratio) / macro_area)
    return (num_cells, num_macros)

def bounded_normal_sample(mean, std_dev, min_val, max_val, size=1):
    samples = np.random.normal(loc=mean, scale=std_dev, size=size)
    clipped = np.clip(samples, min_val, max_val)
    if size == 1:
        return float(clipped[0])
    return clipped.tolist()

def bounded_normal_sample_int(mean, std_dev, min_val, max_val, size=1):
    samples = np.random.normal(loc=mean, scale=std_dev, size=size)
    clipped = np.clip(samples, min_val, max_val)
    rounded = np.round(clipped).astype(int)
    if size == 1:
        return int(rounded[0])
    return rounded.tolist()

def create_macros(h_avg, h_min, h_max, w_avg, w_min, w_max, 
                  pin_num_avg, pin_num_min, pin_num_max, 
                  num_macros, hw_dev_factor=0.1, pin_dev_factor=0.1):
    list = []
    std_dev_h = hw_dev_factor * h_avg
    std_dev_w = hw_dev_factor * w_avg
    std_pin_num = pin_dev_factor * pin_num_avg
    for _ in range(num_macros):
        h = bounded_normal_sample(h_avg, std_dev_h, h_min, h_max)
        w = bounded_normal_sample(w_avg, std_dev_w, w_min, w_max)
        pin_num = bounded_normal_sample_int(pin_num_avg, std_pin_num, pin_num_min, pin_num_max)
        pin_list = [PinSlot(random.uniform(-w/2, w/2), random.uniform(-h/2, h/2)) for _ in range(pin_num)]
        list.append(Macro(w, h, pin_list))
    return list

def create_std_cells(h_avg, h_min, h_max, w_avg, w_min, w_max, 
                  pin_num_avg, pin_num_min, pin_num_max, 
                  num_cells, hw_dev_factor=0.1, pin_dev_factor=0.1):
    list = []
    std_dev_h = hw_dev_factor * h_avg
    std_dev_w = hw_dev_factor * w_avg
    std_pin_num = pin_dev_factor * pin_num_avg
    for _ in range(num_cells):
        h = bounded_normal_sample(h_avg, std_dev_h, h_min, h_max)
        w = bounded_normal_sample(w_avg, std_dev_w, w_min, w_max)
        pin_num = bounded_normal_sample_int(pin_num_avg, std_pin_num, pin_num_min, pin_num_max)
        pin_list = [PinSlot(random.choice([-w/2, w/2]), random.uniform(-h/2, h/2)) for _ in range(pin_num)]
        list.append(StandardCell(w, h, pin_list))
    return list

# define later, but USE THIS by the end
def create_clusters_as_cells():
    pass

def normalize_object(object, max_1_half):
    object.width = object.width / max_1_half
    object.height = object.height / max_1_half
    for pin in object.pin_slots:
        pin.dx = pin.dx / max_1_half
        pin.dy = pin.dy / max_1_half

def normalize_dimensions(chip_h, chip_w, macros: List[Macro], cells: List[StandardCell], ios: List[IOPin]):
    max_1_half = max(chip_h, chip_w) / 2
    # divide all dimensions by max_1_half
    norm_chip_h = chip_h / max_1_half
    norm_chip_w = chip_w / max_1_half
    for macro in macros:
        normalize_object(macro, max_1_half)
    for cell in cells:
        normalize_object(cell, max_1_half)
    for pin in ios:
        pin.width = pin.width / max_1_half
        pin.height = pin.height / max_1_half
    return (norm_chip_h, norm_chip_w, macros, cells, ios)

def shift_norm_object(object, norm_shift_left_amount, norm_shift_down_amount):
    object.x = object.x - norm_shift_left_amount
    object.y = object.y - norm_shift_down_amount

def unshift_norm_object(object, norm_shift_left_amount, norm_shift_down_amount):
    object.x = object.x + norm_shift_left_amount
    object.y = object.y + norm_shift_down_amount

def center_normalized_placement(chip: ChipArea):
    if (chip.height > chip.width):
        norm_shift_left_amount = chip.width / 2
        norm_shift_down_amount = 1
    else:
        norm_shift_left_amount = 1
        norm_shift_down_amount = chip.height / 2
    # center chip
    for macro in chip.macros:
        shift_norm_object(macro, norm_shift_left_amount, norm_shift_down_amount)
    for cell in chip.std_cells:
        shift_norm_object(cell, norm_shift_left_amount, norm_shift_down_amount)
    for pin in chip.io_pins:
        shift_norm_object(pin, norm_shift_left_amount, norm_shift_down_amount)
    return chip

def check_final_positions(chip: ChipArea):
    pass
    # assert(chip)

def undo_centering_on_norm(chip: ChipArea):
    if (chip.height > chip.width):
        norm_shift_left_amount = chip.width / 2
        norm_shift_down_amount = 1
    else:
        norm_shift_left_amount = 1
        norm_shift_down_amount = chip.height / 2
    # uncenter chip
    for macro in chip.macros:
        unshift_norm_object(macro, norm_shift_left_amount, norm_shift_down_amount)
    for cell in chip.std_cells:
        unshift_norm_object(cell, norm_shift_left_amount, norm_shift_down_amount)
    for pin in chip.io_pins:
        unshift_norm_object(pin, norm_shift_left_amount, norm_shift_down_amount)
    return chip

def un_normalize_object(object, max_1_half):
    object.width = object.width * max_1_half
    object.height = object.height * max_1_half
    for pin in object.pin_slots:
        pin.dx = pin.dx * max_1_half
        pin.dy = pin.dy * max_1_half

def undo_normalization(chip: ChipArea):
    max_1_half = max(chip.nn_height, chip.nn_width) / 2
    chip.height = chip.nn_height
    chip.width = chip.nn_width

    for macro in chip.macros:
        un_normalize_object(macro, max_1_half)
    for cell in chip.std_cells:
        un_normalize_object(cell, max_1_half)
    for pin in chip.io_pins:
        pin.width = pin.width * max_1_half
        pin.height = pin.height * max_1_half
    return chip


