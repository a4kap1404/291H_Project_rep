#!/bin/bash

# Description: Used to generate odb and json files from ofrs, and copy files to a local directory

# modify these to generate specific design
# process="nangate45"
process="asap7"
# design="aes"
# design="gcd"
# design="ibex"
# design="ariane136"
design="jpeg"
# design="swerv_wrapper" # fails to generate with nangate45, can try just using regular Makefile


# change to whatever fits your setup
ofrs_dir="../of" # location of open road flow scripts
ofrs_deliv_base="ofrs_deliv"

###################################

working_dir=$(pwd)

# ensure directories exist before running them
mkdir -p corner_libs
mkdir -p ${ofrs_deliv_base}
mkdir -p odbs # outputted ml placements
mkdir -p models # holds trained models

# generate 3_2 (and the rest) odb files using openroad (need to do this only once per design) (comment-in if you want to run ofrs)
# cd ${ofrs_dir}/flow
# if [ ! -e "${working_dir}/GenTestAndTrain.mk" ]; then
#     echo "Moving custom makefile (GenTestAndTrain.mk) into ${ofrs_dir}/flow"
#     cp GenTestAndTrain.mk ${ofrs_dir}/flow
# fi
# echo "######## Launching OFRS to generate design(${design} on process(${process})) ########"
# DESIGN_CONFIG=./designs/${process}/${design}/config.mk  make -f GenTestAndTrain.mk
# cd ${working_dir}

# copy data (to not accidentally modify)
echo "######## Copying files from design(${design} on process(${process})) ########"
ofrs_deliv=${ofrs_deliv_base}/${process}/${design}
mkdir -p ${ofrs_deliv}
cp ${ofrs_dir}/flow/results/${process}/${design}/base/3_2_place_iop.odb ${ofrs_deliv} # for initial placement
cp ${ofrs_dir}/flow/results/${process}/${design}/base/3_5_place_dp.odb ${ofrs_deliv} # for comparsion to ml placement, will be used to get hpwl using custom algo
cp ${ofrs_dir}/flow/logs/${process}/${design}/base/3_5_place_dp.json ${ofrs_deliv} # could use for getting hpwl, but we estimate it ourselves
# for getting timing
cp ${ofrs_dir}/flow/logs/${process}/${design}/base/3_3_place_dp.log ${ofrs_deliv} # could use for getting hpwl, but we estimate it ourselves
cp ${ofrs_dir}/flow/logs/${process}/${design}/base/3_4_place_dp.log ${ofrs_deliv} # could use for getting hpwl, but we estimate it ourselves
cp ${ofrs_dir}/flow/logs/${process}/${design}/base/3_5_place_dp.log ${ofrs_deliv} # could use for getting hpwl, but we estimate it ourselves
