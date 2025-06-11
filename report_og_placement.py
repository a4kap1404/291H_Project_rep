"""
    Estimates HPWL using algorithm taken from: the open_road repo
        - link: https://github.com/The-OpenROAD-Project/OpenROAD/blob/master/src/gpl/test/report_hpwl.tcl
    
    Important Note: Custom HPWL is a bit off from OFRS one in 3_5_place_dp.json.
    Thus, we will report both in case of massive divergence.

    Metrics Report in report_og_placement.py
        CUSTOM_HPWL(original_placement)
        ORFS_HPWL(original_placement)
"""

from openroad import Design, Tech, Timing
from odb import *
from pathlib import Path
from py_utils.utils import *
from py_utils.utils_2 import *

# adjust if neccesary
lib_dir = "corner_libs"
ofrs_dir = "ofrs_deliv"
tcl_base_dir = "misc" # adjust if needed

design = "gcd"
process="nangate45"

lib_map = {
    "nangate45": f"{lib_dir}/NangateOpenCellLibrary_typical.lib",
    "asap7": f"{lib_dir}/merged.lib" # not sure if correct
}

lib_path = lib_map[process]
odb_path = f"{ofrs_dir}/{process}/{design}/3_5_place_dp.odb"

odb_design, block, dbu_per_micron = load_odb_info(lib_path, odb_path, giveOdbDesign=True)

# custom hpwl of original placement
hpwl_in_micro = getHpwlInMicrons(odb_design, dbu_per_micron, tcl_base_dir)
print(f"original placememt custom estimation of hpwl: {hpwl_in_micro} um")

# ofrs hpwl
# REPORT THIS