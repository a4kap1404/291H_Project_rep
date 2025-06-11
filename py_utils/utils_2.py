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

# from utils_3 import *
from py_utils.utils_3 import *


# from datagen import *

# report custom hpwl of ml placement
def getHpwlInMicrons(odb_design, block, tech, dbu_per_micron, tcl_base_dir):
    odb_design.evalTclString(f"source {tcl_base_dir}/report_hpwl.tcl")
    hpwl = odb_design.evalTclString("get_my_hpwl")
    assert hpwl == str(int(hpwl))
    hpwl = int(hpwl)
    rounded_hpwl_in_microns = int(hpwl / dbu_per_micron)
    return rounded_hpwl_in_microns

def load_odb_info(lib_path, odb_path):
    # If we had a separate .lef, we could run tech.readLef but we're using a db here that comes packaged with the library.
    tech = Tech()
    # We do still need to load the Liberty because OpenDB doesn't store enough information for OpenSTA to run correctly.
    tech.readLiberty(lib_path)
    odb_design = Design(tech) # Every Design has to be associated with a Tech, even if it's empty.
    odb_design.readDb(odb_path)
    library = odb_design.getDb().getLibs()[0]
    dbu_per_micron = library.getDbUnitsPerMicron()
    block = odb_design.getBlock()

    return odb_design, block, tech, dbu_per_micron

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

def get_edges_and_pin_locs(large_net_threshold, inst_map, block):
#   f = open(file_name, "a")
#   f.write("Nets information (Each line represents one net. The first element in each line is the driver pin.):\n")
    # block = ord.get_db_block()
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
                driver_rel_location = get_rel_location(p)
            elif not p.isOutputSignal() and is_part_of_std_cell(p):
                sink_id = inst_map[p.getInst().getName()]
                sink_rel_locations = get_rel_location(p)
                sinkPins.append(sink_id)
                sinkPinPositons.append(sink_rel_locations)
        
        if driver_is_not_cell:
            break
        
        if dPin is None:
            if (net.getName() == "VDD" or net.getName() == "VSS"):
                continue  # ignore power and ground nets
            # print("No driver found for net: ",net.getName())
            continue
    
        if (len(sinkPins) + 1 >= large_net_threshold):
            # print("Ignore large net: ",net.getName())
            continue

        if (len(sinkPins) == 0):
            continue

        net_edges = hyperedge_to_edges(driverId, sinkPins)
        net_edge_pin_pos = hyperedge_to_edges(driver_rel_location, sinkPinPositons)

        if len(net_edges) > 0:
            edges.extend(net_edges)
            edge_pin_locs.extend(net_edge_pin_pos)
    
    return edges, edge_pin_locs


def get_macros_and_cells(block):
    # block = ord.get_db_block()
    insts = block.getInsts()
#   registers = get_registers(design)

    cell_id = 0
    cell_map = dict()

    # print(f"len of insts: {len(block.getInsts())}")

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
            cell.x = float(x_center)
            cell.y = float(y_center)
            cells.append(cell)

        # grab relative positions of pins (later)
        # grab all connections and store into a multigraph (later)
    
    # assert len(cell_map) == len(cells)
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
def export_odb_data(filename_1, filename_2, block, odb_design, tech):

    print("getting macros and cells...")
    cell_map, cells, macros = get_macros_and_cells(block)
    large_net_threshold = 10
    print("edges and pin locations...")
    edges, pin_locs = get_edges_and_pin_locs(large_net_threshold, cell_map, block)
    # block = ord.get_db_block()
    chip_w = block.getCoreArea().dx() 
    chip_h = block.getCoreArea().dy()

    print(f"cell amount:{len(cells)}")
    print(f"macro amount:{len(macros)}")
    print(f"unfiltered edges amount:{len(edges)}")
    print(f"unfiltered associated pin (not i/o) amount:{len(pin_locs)}")
    print(f"chip dims (x,y):{chip_w, chip_h}")

    ml_save_file = (cell_map, cells, macros, edges, pin_locs, chip_w, chip_h)
    # ml_save_file = cells

    # print([type(c) for c in ml_save_file])
    # print("ml_save_file:", ml_save_file)
    with open(filename_1, "wb") as f:
        pickle.dump(ml_save_file, f)

    with open(filename_2, "wb") as f:
        pickle.dump(cell_map, f)


    


