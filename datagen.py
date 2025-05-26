import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass, field
from typing import List, Tuple
import pickle

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
        for idx, m in enumerate(self.macros):
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

    def place_cells(self, region: float, dthresh: float, max_retries: int = 500) -> None:
        placed_cells = []
        placed_bounds = []
        for idx, c in enumerate(self.std_cells):
            placed = False
            for attempt in range(max_retries):
                x = self.rng.uniform(c.width/2, self.width - c.width/2)
                y = self.rng.uniform(c.height/2, self.height - c.height/2)
                b = (x - c.width/2, y - c.height/2, x + c.width/2, y + c.height/2)
                if self.overlaps_any(b, [m.bounds() for m in self.macros]):
                    continue
                if self.overlaps_any(b, placed_bounds):
                    continue
                if self.density(region, x, y, placed_cells) > dthresh:
                    continue
                c.x, c.y = x, y
                placed_cells.append(c)
                placed_bounds.append(b)
                placed = True
                break
            if not placed:
                # print(f"Warning: could not place cell #{idx} after {max_retries} attempts; skipping.")
                dummy = None
        self.std_cells = placed_cells

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

    def pins(self) -> List[Tuple[float, float]]:
        ps = []
        for m in self.macros:
            ps.extend(m.absolute_pins())
        for c in self.std_cells:
            ps.extend(c.absolute_pins())
        for io in self.io_pins:
            ps.append(io.pin_position())
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
                       max_cell_retries: int = 500) -> Tuple[ChipArea, nx.DiGraph]:
    rng = random.Random(seed) if seed is not None else random
    chip = ChipArea(w, h, macros, cells, ios, rng)
    chip.place_macros(max_retries=max_macro_retries)
    chip.place_io()
    chip.place_cells(region, dthresh, max_retries=max_cell_retries)

    G = nx.DiGraph()
    pins = chip.pins()
    for idx, (x, y) in enumerate(pins):
        G.add_node(idx, pos=(x, y))

    used_targets = set()
    used_sources = set()

    # Build unique undirected pairs sorted by distance
    pairs = []
    N = len(pins)
    for i in range(N):
        for j in range(i+1, N):
            dist = math.hypot(pins[i][0]-pins[j][0], pins[i][1]-pins[j][1])
            pairs.append((i, j, dist))
    pairs.sort(key=lambda t: t[2])
    total = len(pairs)

    # Diffusion-based sampling with unique targets and no chaining
    for t in range(1, steps+1):
        threshold = pairs[int(t/steps*total) - 1][2]
        for i, j, d in pairs:
            if d > threshold:
                break
            if rng.random() < math.exp(-d/scale):
                if i not in used_sources and j not in used_targets:
                    G.add_edge(i, j)
                    used_sources.add(i)
                    used_targets.add(j)
                elif j not in used_sources and i not in used_targets:
                    G.add_edge(j, i)
                    used_sources.add(j)
                    used_targets.add(i)
    return chip, G


def plot_full(chip: ChipArea, G: nx.DiGraph, show: bool = True) -> None:
    fig, ax = plt.subplots()
    for m in chip.macros:
        ax.add_patch(plt.Rectangle((m.x-m.width/2, m.y-m.height/2), m.width, m.height,
                                   facecolor='red', alpha=0.4, edgecolor='black'))
    for c in chip.std_cells:
        ax.add_patch(plt.Rectangle((c.x-c.width/2, c.y-c.height/2), c.width, c.height,
                                   facecolor='blue', alpha=0.4, edgecolor='black'))
    xs = [io.x for io in chip.io_pins]
    ys = [io.y for io in chip.io_pins]
    ax.scatter(xs, ys, marker='s', facecolor='green', edgecolor='black')
    for u, v in G.edges():
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']
        arrow = FancyArrowPatch(posA=(x1, y1), posB=(x2, y2), arrowstyle='-|>',
                                mutation_scale=5, linewidth=0.5, color='gray', alpha=0.7,
                                shrinkA=2, shrinkB=2)
        ax.add_patch(arrow)
    ax.set_xlim(0, chip.width)
    ax.set_ylim(0, chip.height)
    ax.set_aspect('equal')
    if show:
        plt.savefig('chip_design.png', bbox_inches='tight', dpi=300)

# Example use:
if __name__ == '__main__':
    N = 50
    seed = 40
    # Define ranges (you will fill these in)
    MIN_M_H = 0
    MAX_M_H = 0
    MIN_M_W = 0
    MAX_M_W = 0
    MIN_M_NUM_PINS = 0
    MAX_M_NUM_PINS = 0
    MIN_M_NUM = 0
    MAX_M_NUM = 0

    # From cell metrics
    MIN_C_H = 2800
    MAX_C_H = 2800

    MIN_C_W = 1718.62   # from GCD
    MAX_C_W = 1991.75   # from AES

    MIN_C_NUM_PINS = 0
    MAX_C_NUM_PINS = 0  # since all are 0

    MIN_C_NUM = 551     # GCD
    MAX_C_NUM = 15478   # AES

    # Chip dimensions
    MIN_CHIP_W = 72760     # GCD
    MAX_CHIP_W = 499700    # AES

    MIN_CHIP_H = 72760     # GCD
    MAX_CHIP_H = 501200    # AES

    AVG_IO_PIN = 221


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


        macros = [Macro(m_w, m_h, [PinSlot(5, 0)]) for _ in range(m_num_pins)]
        cells = [StandardCell(c_w, c_h, [PinSlot(2, 0)]) for _ in range(c_num_pins)]
        ios = [IOPin(3, 3) for _ in range(num_io_pin)]
        data_entry = generate_placement(chip_w, chip_h, macros, cells, ios, chip_h / 100, 0.4, steps=50, scale=2.5, seed=seed)
        Data.append(data_entry)
        # chip, g = data_entry
        # plot_full(chip, G)
    # save
    with open("Data.pkl", "wb") as f:
        pickle.dump(Data, f)
    # load  
    # classes must be deifned to load
    # with open("Data.pkl", "rb") as f:
        # Data = pickle.load(f)