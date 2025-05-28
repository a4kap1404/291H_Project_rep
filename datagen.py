import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass, field
from typing import List, Tuple
import pickle
import numpy as np

@dataclass
class PinSlot:
    dx: float
    dy: float

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
        return [(self.x+p.dx, self.y+p.dy) for p in self.pin_slots]

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

class ChipArea:
    def __init__(self, width, height, macros, std_cells, io_pins, rng=None):
        self.width, self.height = width, height
        self.macros = macros
        self.std_cells = std_cells
        self.io_pins = io_pins
        self.rng = rng or random

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

    def pins(self) -> List[Tuple[float, float, int]]:
        ps = []
        # for m in self.macros:
        #     if m.x is not None:
        #         ps.extend(m.absolute_pins())
        i = 0
        for c in self.std_cells:
            if c.x is not None:
                t = c.absolute_pins()[0] + (i,)
                ps.extend([t,])
            i += 1
        # for io in self.io_pins:
        #     if io.x is not None:
        #         ps.append(io.pin_position())
        print("length of ps: ", len(ps))
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
    pins = chip.pins()
    random.shuffle(pins)
    for (x, y, idx) in pins:
        G.add_node(idx, pos=(x, y))

    print("added placed objects to graph, pins # =", len(pins))

    used_targets = set()
    used_sources = set()

    # Build unique undirected pairs sorted by distance
    pairs = []
    N = len(pins)
    edge_step_size = N // edge_sample_ratio
    for i in range(N):
        # print(i)
        for j in range(i+1, N, edge_step_size):
            dist = math.hypot(pins[i][0]-pins[j][0], pins[i][1]-pins[j][1])
            pairs.append((i, j, dist))
    pairs.sort(key=lambda t: t[2])
    total = len(pairs)

    print("sorted by distance")
    # print("total:", total)
    # print(pairs[0])
    # exit()

    selected_edges = biased_shuffle(pairs, scale)
    for k in range(int(len(selected_edges) / edge_to_node_ratio)):
        i = selected_edges[k][0]
        j = selected_edges[k][1]
        if i not in used_sources and j not in used_targets:
            G.add_edge(i, j)
            used_sources.add(i)
            used_targets.add(j)
        elif j not in used_sources and i not in used_targets:
            G.add_edge(j, i)
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
                                mutation_scale=15, linewidth=1.5, color='red', alpha=0.9,
                                shrinkA=-1, shrinkB=-1)
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

# Example use:
if __name__ == '__main__':
    N = 2
    seed = 40
    density = 0.9
    max_macro_retries = 30
    max_cell_retries = 30
    div = 10000
    scale = 4.0

    # From cell metrics
    MIN_C_H = 2800
    MAX_C_H = 2800

    MIN_C_W = 1718.62   # from GCD
    MAX_C_W = 1991.75   # from AES
    # MAX_C_W = 1718.62   # from GCD


    MIN_C_NUM_PINS = 2
    # MAX_C_NUM_PINS = 0  # since all are 0
    MAX_C_NUM_PINS = 10 
    MAX_C_NUM_PINS = 10 


    MIN_C_NUM = 551     # GCD
    MAX_C_NUM = 6000   # manual
    # MAX_C_NUM = 15478   # AES
    # MAX_C_NUM = 15478   # AES
    # MAX_C_NUM = 551     # GCD


    # manually set
    cell_to_macro_size = 8
    # MIN_M_H = 0
    MIN_M_H = int(cell_to_macro_size * MIN_C_H)
    # MAX_M_H = 0
    MAX_M_H = int(cell_to_macro_size * MAX_C_H)
    # MIN_M_W = 0
    MIN_M_W = int(cell_to_macro_size * MIN_C_W)
    # MAX_M_W = 0
    MAX_M_W = int(cell_to_macro_size * MAX_C_W)
    MIN_M_NUM_PINS = int(MIN_C_NUM_PINS * 2)
    MAX_M_NUM_PINS = int(MAX_C_NUM_PINS * 2)
    MIN_M_NUM = 0
    MAX_M_NUM = 10


    # Chip dimensions
    MIN_CHIP_W = 72760     # GCD
    MAX_CHIP_W = 100000    # CUSTOM
    # MAX_CHIP_W = 499700    # AES
    # MAX_CHIP_W = 72760     # GCD
    

    MIN_CHIP_H = 72760     # GCD
    MAX_CHIP_H = 100000    # CUSTOM
    # MAX_CHIP_H = 501200    # AES
    # MAX_CHIP_H = 72760     # GCD


    AVG_IO_PIN = 22


    Data = list()

    for i in range(N):
        random.seed(seed + i)

        m_h = random.randint(MIN_M_H, MAX_M_H)
        m_w = random.randint(MIN_M_W, MAX_M_W)
        m_num_pins = random.randint(MIN_M_NUM_PINS, MAX_M_NUM_PINS)
        m_num = random.randint(MIN_M_NUM, MAX_M_NUM)

        c_h = random.randint(MIN_C_H, MAX_C_H)
        c_w = int(random.uniform(MIN_C_W, MAX_C_W))
        c_num_pins = random.randint(MIN_C_NUM_PINS, MAX_C_NUM_PINS)
        c_num = random.randint(MIN_C_NUM, MAX_C_NUM)

        num_io_pin = AVG_IO_PIN # CHANGE if needed

        chip_w = random.randint(MIN_CHIP_W, MAX_CHIP_W)
        chip_h = random.randint(MIN_CHIP_H, MAX_CHIP_H)

        # div = 100

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

        print("num_cells", num_cells)
        print("num_macros", num_macros)
        # exit()

        macros = [Macro(m_w, m_h, [PinSlot(5, 0) for _ in range(m_num_pins)]) for _ in range(num_macros)]
        cells = [StandardCell(c_w, c_h, [PinSlot(2, 0) for _ in range(c_num_pins)]) for _ in range(num_cells)]
        ios = [IOPin(3, 3) for _ in range(num_io_pin)]
        data_entry = generate_placement(
            chip_w, chip_h, macros, cells, ios, chip_h / div, 
            density, steps=50, scale=scale, seed=seed, 
            max_macro_retries=max_macro_retries, max_cell_retries=max_cell_retries)
        chip, G = data_entry
        for cell in chip.std_cells:
            if cell.x is None or cell.y is None:
                raise Exception("cell has None positions!")
        Data.append(data_entry)
        # if i == 1:
        #     # print(data_entry[0].std_cells[0].x)
        #     # print(data_entry[0].std_cells)
        #     chip, G = data_entry
        #     for u, v in G.edges():
        #         # print("i", end="")
        #         if 'pos' not in G.nodes[u] or 'pos' not in G.nodes[v]:
        #             print(f"Missing position for edge {u}->{v}")
        #             raise Exception("graph problem")
        #     plot_full(chip, G)
        #     # print(dir(G))
        #     print("# of edges: ", len(G.edges))
        #     # print(G.edges)
        #     print("exiting")
        #     exit()

    # save
    with open("Data.pkl", "wb") as f:
        pickle.dump(Data, f)


    # load  
    # classes must be deifned to load
    # with open("Data.pkl", "rb") as f:
        # Data = pickle.load(f)