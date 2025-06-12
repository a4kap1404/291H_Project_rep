#!/bin/bash
set -e

# launch using "bash gen_conversions.sh"

# Description: Ignore this file (and get_final_times.sh) if you can use CUDA with pytorch, and ORFS on the same container. 
# If so, just use place.sh to get benchmarks
# this file will convert designs and store them in odbs/{process}/{design}, then will push to github, then will pull on cuda container to run model and get times
# Will measusre times, then push to github, then on ORFS container will use get get_final_times.sh to get final times

ml_placement_dir="odbs"

# process="nangate45"
# # process="asap7"
# # design="aes"
# design="gcd"
# # design="ibex"
# # design="ariane136"
# # design="jpeg"
# # design="swerv_wrapper"

# for parsing

processes=("nangate45" "asap7")
designs=("aes" "gcd" "ibex" "jpeg")

for process in "${processes[@]}"; do
  for design in "${designs[@]}"; do

    echo "Running for process=$process, design=$design"
    mkdir -p ${ml_placement_dir}/${process}/${design}

    echo "\n--- Launching place_p1.py ---"
    openroad -python -no_splash -exit place_p1.py ${design} ${process}

  done
done
