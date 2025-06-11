"""
    Description:
        Reads odb and outputs data for ML model, aswell as for later when mapping
        python data structure of "cells" back to their odb cell counterparts
"""

# this is an odb script!

from openroad import Design, Tech, Timing
from odb import *
from pathlib import Path
from py_utils.utils import *
from py_utils.utils_2 import *

# adjust if neccesary
lib_dir = "corner_libs"
ofrs_dir = "ofrs_deliv"

design = "gcd"
process="nangate45"

lib_map = {
    "nangate45": f"{lib_dir}/NangateOpenCellLibrary_typical.lib",
    "asap7": f"{lib_dir}/merged.lib" # not sure if correct
}
lib_path = lib_map[process]
odb_path = f"{ofrs_dir}/{process}/{design}/3_2_place_iop.odb"

block, dbu_per_micron = load_odb_info(lib_path, odb_path)

# assume 3_2_place_iop has been done and loaded
filename = "odb_placement.pkl"
cell_map_filename = "odb_placement_cell_map.pkl"
export_odb_data(filename, cell_map_filename, block)

