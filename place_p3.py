# this is an odb script!


from openroad import Design, Tech, Timing
from odb import *
from pathlib import Path
from utils import *
from utils_2 import *

import sys
import argparse
import pdn, odb, utl
import time

import pickle

# If we had a separate .lef, we could run tech.readLef but we're using a db here that comes packaged with the library.
tech = Tech()
# We do still need to load the Liberty because OpenDB doesn't store enough information for OpenSTA to run correctly.
tech.readLiberty("./corner_libs/NangateOpenCellLibrary_typical.lib")

design = Design(tech) # Every Design has to be associated with a Tech, even if it's empty.

design.readDb("./odbs/3_2_place_iop.odb")
# assert(utils.lib_unit_consistency(design))
library = design.getDb().getLibs()[0]
dbu_per_micron = library.getDbUnitsPerMicron()
cam_vertical_offset = library.getSites()[0].getHeight()
block = design.getBlock()

print(dbu_per_micron)

# load placement
placement_name = "ml_placed_graph.pkl"
with open(placement_name, "rb") as f:
    cell_placement = pickle.load(f)

# load cell map
cell_map_filename = "odb_placement_cell_map.pkl"
with open(cell_map_filename, "rb") as f:
    cell_map = pickle.load(f)


# move cells
# for i, cell in enumerate(cell_placement):
# block = ord.get_db_block()
insts = block.getInsts()
for inst in insts:
    if not inst.getName() in cell_map:
        continue
    cell_id = cell_map[inst.getName()]
    width = cell_placement[cell_id][0]
    height = cell_placement[cell_id][1]
    x = cell_placement[cell_id][2]
    y = cell_placement[cell_id][3]
    assert x > 0 and y > 0
    x_placement_pos = x - width/2
    y_placement_pos = y - height/2
    inst.setLocation(int(x_placement_pos), int(y_placement_pos))

run_incremental_placement(design)

    
   





