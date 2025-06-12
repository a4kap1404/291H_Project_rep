#!/bin/bash
set -e

# launch using "gen_gpu_ml_placements.sh"

# Description: Ignore this file if you can use CUDA with pytorch, and ORFS on the same container. 
# If so, just use place.sh to get benchmarks

ml_placement_dir="odbs"

processes=("nangate45" "asap7")
designs=("aes" "gcd" "ibex" "jpeg")

# for parsing
ufloat="\K[0-9]+.[0-9]+"

for process in "${processes[@]}"; do
  for design in "${designs[@]}"; do

    echo "Running for process=$process, design=$design"

    echo "\n--- Launching place_p2.py ---"
    p2_output=$(python place_p2.py ${design} ${process} | tee /dev/tty)
    ml_gp_init_time=$(echo "${p2_output}" | grep -oP "Elapsed time: ${ufloat}") # seconds
    
    current_date=$(date)
    echo "Design: ${design}, Process: ${process} | ${current_date}" >> gpu_ml_placements.log
    echo "time (s): ${ml_gp_init_time}" >> gpu_ml_placements.log
  done
done

