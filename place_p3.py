# this is an odb script!

"""
    Important Note: 
        After generating the 3_5 detailed placement odb,
        we report hpwl of generated placement using a algorithm taken 
        from the openroad github repo. This mainly due to me not being able to 
        isolate the exact tcl code to easily get the HPWL from an post-detailed placement
        odb. The results are NOT exactly the same,
        but seemingly "close enough" given how off the ml_placement is from
        the original. For a sanity check however, we will report the original
        HPWL estimate from the 3_5_place_dp.json in ANOTHER file (report_og_placement.py)

    Custom HPWL Algorithm taken from:
        https://github.com/The-OpenROAD-Project/OpenROAD/blob/master/src/gpl/test/report_hpwl.tcl
    
    Metrics Reported in this script:
        CUSTOM_HPWL(ml_placement)

    Metrics Report in report_og_placement.py
        CUSTOM_HPWL(original_placement)
        ORFS_HPWL(original_placement)
"""

from openroad import Design, Tech, Timing
from odb import *
from pathlib import Path
from py_utils.utils import *
from py_utils.utils_2 import *

import pickle
import sys

# adjust if neccesary
odb_dir_base = "odbs" # output directory
ofrs_dir = "ofrs_deliv"
tcl_base_dir = "misc"
lib_dir = "corner_libs"

design = "gcd"
process="nangate45"
if len(sys.argv) < 3:
    print("\nError: openroad -python -exit place_p3.py design process")
    sys.exit(1)
design = sys.argv[1]
process = sys.argv[2]


lib_map = {
    "nangate45": f"{lib_dir}/NangateOpenCellLibrary_typical.lib",
    "asap7": f"{lib_dir}/merged.lib" # not sure if correct
}
lib_path = lib_map[process]

odb_path = f"{ofrs_dir}/{process}/{design}/3_2_place_iop.odb"
odb_dir = f"{odb_dir_base}/{process}/{design}"

odb_design, block, tech, dbu_per_micron = load_odb_info(lib_path, odb_path)

# load placement
placement_name = f"{odb_dir}/ml_placed_graph.pkl"
with open(placement_name, "rb") as f:
    cell_placement = pickle.load(f)

# load cell map
cell_map_filename = f"{odb_dir}/odb_placement_cell_map.pkl"
with open(cell_map_filename, "rb") as f:
    cell_map = pickle.load(f)

# move cells
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

# timing rest of 
start = time.perf_counter()
run_incremental_placement(odb_design, block, tech, odb_dir)
end = time.perf_counter()
elapsed = end - start
print(f"Elapsed time of post-intialization placement: {elapsed:.6f} seconds")

# report custom hpwl of ml placement
hpwl_in_micro = getHpwlInMicrons(odb_design, block, tech, dbu_per_micron, tcl_base_dir)
print(f"original placememt custom estimation of hpwl: {hpwl_in_micro} um")