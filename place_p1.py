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


print("starting python script...")

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

# assume 3_2_place_iop has been done and loaded
filename = "odb_placement"
cell_map_filename = "odb_placement_cell_map"
export_odb_data(filename, cell_map_filename, block)
