# this is an odb script!

from openroad import Design, Tech, Timing
from odb import *
from pathlib import Path
from utils import *
from utils_2 import *
from train_utils import *

import sys
import argparse
import pdn, odb, utl
import time

import model



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

exit()

# assume 3_2_place_iop has been done and loaded

# import as graph and chip (normalized and centered)

filename = "odb_placement.pkl"
# export_odb_data(filename) # restore


# make dataset
# dataset = PlacementDataset()

# load model

# launch model

# push placed model back into the odb (can stop and vizualize for testing)

# call run_incremental_place to run global and detailed placement (skips resizing)

# grab HPWL (somehow) (apparently under "detailedplace__route__wirelength__estimated" in 3_5 json)


# 


# export currently placed hypergraph into files

# grab newely placed hypergraph and do rest of placement (but no resizing for some reason?)

# grab metrics of hpwl

# example of how to hightlight a certain cell: select -type Inst -name _486_ -filter Master=INV_X1

print("Please use the generated 3_3_place_gp.def and 3_3_place_gp.odb files for remaining flows.")





# print(len(block.getNets()))
nets = block.getNets()
# print(dir(nets))
# print(len)
# print(nets[0].getITerms())
# print(nets[0].getBTerms())
# for i in range(len(nets)):
    # print(nets[i].getName())
    # print(nets[i].getITerms())
    # print(nets[i].getBTerms())


print("finished python script...")

parser = argparse.ArgumentParser(description="Example script to perform global placement initialization using OpenROAD.")
parser.add_argument("-d", default="ibex", help="Give the design name")
parser.add_argument("-t", default="nangate45", help="Give the technology node")
parser.add_argument("-large_net_threshold", default="1000", help="Large net threshold. We should remove global nets like reset.")
parser.add_argument("-flow_path", default="./of/flow", help="path of flow directory")

args = parser.parse_args()

tech_node = args.t
design = args.d
large_net_threshold = int(args.large_net_threshold)
hg_file_name = str(design) + "_" + str(tech_node) + ".txt"
f = open(hg_file_name, "w")
f.close()

flow_path = args.flow_path
path = flow_path + "/results/" + tech_node + "/" + design + "/base"
floorplan_odb_file = path + "/3_2_place_iop.odb"
sdc_file = path + "/2_floorplan.sdc"

# Load the design
tech, design = load_design(tech_node, floorplan_odb_file, sdc_file)

# transcibe cells into matrix that stores height and width of each cell
# N = # of nodes/cells; E = # of 2 pin edges
nodes = ... # N x 2
edge_attr = ... # E x 4  (relative positions of pin locaitons (x,y) for tail and head)
edge_indices = ... # E x 2

# load trained model

# run model

# NOTE: REMEMBER TO UNDO THE NORMALIZATION

# iterate through each row of [N x 2] matrix, and that will represent the positions of each cell


