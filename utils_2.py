from openroad import Design, Tech, Timing
from odb import *
from pathlib import Path
import sys
import argparse
import pdn, odb, utl
import time
import random
# import torch
# from torch_geometric.data import Data

import pickle

# import


# from datagen import *
from typing import List, Tuple

from dataclasses import dataclass, field

@dataclass
class PinSlot:
    dx: float
    dy: float

    def __str__(self):
        return f"(dx, dy): ({self.dx}, {self.dy})"
    
@dataclass
class Macro:
    width: float
    height: float
    pin_slots: List[PinSlot]
    x: float = field(init=False, default=None)
    y: float = field(init=False, default=None)

@dataclass
class StandardCell:
    width: float
    height: float
    pin_slots: List[PinSlot]
    x: float = field(init=False, default=None)
    y: float = field(init=False, default=None)

@dataclass
class IOPin:
    width: float
    height: float
    x: float = field(init=False, default=None)
    y: float = field(init=False, default=None)

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

def is_std_cell(inst):
   isMacro = inst.getMaster().isBlock()
   isFixed = inst.isFixed()
   return not (isMacro or isFixed)

def is_part_of_std_cell(pin):
   isMacro = pin.getInst().getMaster().isBlock()
   isFixed = pin.getInst().isFixed()
   return not (isMacro or isFixed)

def hyperedge_to_edges(src_id, trgt_ids):
    edges = []
    for trgt_id in trgt_ids:
        edges.append((src_id, trgt_id))
    return edges

def get_edges_and_pin_locs(large_net_threshold, inst_map):
#   f = open(file_name, "a")
#   f.write("Nets information (Each line represents one net. The first element in each line is the driver pin.):\n")
    block = ord.get_db_block()
    nets = block.getNets()

    edges = []
    edge_pin_locs = []

    for net in nets:
        if net.getName() == "VDD" or net.getName() == "VSS":
            continue
        if (len(net.getITerms()) + len(net.getBTerms()) >= large_net_threshold):
            continue

        sinkPins = []
        sinkPinPositons = []
        driverId = -1
        dPin = None

        driver_is_not_cell = False

        # check the instance pins
        for p in net.getITerms():
            if p.isOutputSignal() and not is_part_of_std_cell(p):
                driver_is_not_cell = True
                break
            if p.isOutputSignal() and is_part_of_std_cell(p):
                dPin = p
                driverId = inst_map[p.getInst().getName()]
                driver_rel_location = get_rel_location()
            elif not p.isOutputSignal() and is_part_of_std_cell(p):
                sink_id = inst_map[p.getInst().getName()]
                sink_rel_locations = get_rel_location()
                sinkPins.append(sink_id)
                sinkPinPositons.append(sink_rel_locations)
        
        if driver_is_not_cell:
            break
        
        if dPin is None:
            if (net.getName() == "VDD" or net.getName() == "VSS"):
                continue  # ignore power and ground nets
            print("No driver found for net: ",net.getName())
            continue
    
        if (len(sinkPins) + 1 >= large_net_threshold):
            print("Ignore large net: ",net.getName())
            continue

        if (len(sinkPins) == 0):
            continue

        net_edges = hyperedge_to_edges(driverId, sinkPins)
        net_edge_pin_pos = hyperedge_to_edges(driver_rel_location, sinkPinPositons)

        if len(net_edges) > 0:
            edges.extend(net_edges)
            edge_pin_locs.extend(net_edge_pin_pos)
    
    return edges, edge_pin_locs


def get_macros_and_cells():
    block = ord.get_db_block()
    insts = block.getInsts()
#   registers = get_registers(design)

    cell_id = 0
    cell_map = dict()

    macros = []
    cells = []
    # io_pins = []
    # chip_h = None
    # chip_w = None

    # after grabbng list connections, should check that every cell_id matches with one we find here

    for inst in insts:
        instName = inst.getName()
        master = inst.getMaster()
        masterName = master.getName()
        BBox = inst.getBBox()
        isMacro = master.isBlock()
        # isSeq = True if instName in registers else False
        isFixed = True if inst.isFixed() else False
        lx = BBox.xMin()
        ly = BBox.yMin()
        ux = BBox.xMax()
        uy = BBox.yMax()
        width = ux - lx
        height = uy - ly
        x_center = (lx + ux) / 2
        y_center = (ly + uy) / 2
        isFiller = True if master.isFiller() else False
        isTapCell = True if ("TAPCELL" in masterName or "tapcell" in masterName) else False
        #isBuffer = 1 if design.isBuffer(master) else 0
        #isInverter = 1 if design.isInverter(master) else 0
        if (isFiller == True or isTapCell == True):
            continue # ignore filler and tap cells
        
        if (isMacro == True or isFixed == True):
            macro = Macro(width, height, pin_slots=[])
            macro.x = x_center
            macro.y = y_center
            macros.append(macro)
        else: # is cell
            cell_map[instName] = cell_id
            cell_id += 1
            cell = StandardCell(width, height, pin_slots=[])
            cell.x = x_center
            cell.y = y_center
            cells.append(cell)

        # grab relative positions of pins (later)
        # grab all connections and store into a multigraph (later)
    
    assert len(cell_map) == len(cells)
    return cell_map, cells, macros

def get_rel_location(iterm):
    inst_bbox = iterm.getInst().getBBox()
    inst_center_x = (inst_bbox.xMin() + inst_bbox.xMax()) / 2
    inst_center_y = (inst_bbox.yMin() + inst_bbox.yMax()) / 2
    pin_bbox = iterm.getBBox()
    if pin_bbox.xMin() == 0 and pin_bbox.yMin() == 0 and \
        pin_bbox.xMax() == 0 and pin_bbox.yMax() == 0:
            x, y = iterm.getAvgXY()
            pin_center_x = x
            pin_center_y = y
    else:
        pin_center_x = (pin_bbox.xMin() + pin_bbox.xMax()) / 2
        pin_center_y = (pin_bbox.yMin() + pin_bbox.yMax()) / 2
    dx = pin_center_x - inst_center_x
    dy = pin_center_y - inst_center_y
    return (dx, dy)

    # chip_w = block.getCoreArea().dx() 
    # chip_h = block.getCoreArea().dy()

# STOP HERE AND SAVE
def export_odb_data(filename):
    cell_map, cells, macros = get_macros_and_cells()
    large_net_threshold = 10
    edges, pin_locs = get_edges_and_pin_locs(large_net_threshold, cell_map)
    block = ord.get_db_block()
    chip_w = block.getCoreArea().dx() 
    chip_h = block.getCoreArea().dy()

    save_file = (cell_map, cells, macros, edges, pin_locs, chip_w, chip_h)
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(save_file, f)



    


