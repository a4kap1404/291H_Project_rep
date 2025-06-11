#!/bin/bash

# runtime entire placement inference flow, and records file in results.log

ml_placement_dir="odbs"
ofrs_deliv_base="ofrs_deliv"


process="nangate45"
# process="asap7"
# design="aes"
design="gcd"
# design="ibex"
# design="ariane136"
# design="jpeg"
# design="swerv_wrapper"

mkdir -p ${ml_placement_dir}/${process}/${design}

# note: "og" stands for original (vanilla openroad)

# for parsing
ufloat="\K[0-9]+.[0-9]+"
uint="\K[0-9]+"

echo "\n--- Launching place_p1.py ---"
openroad -python -no_splash -exit place_p1.py ${design} ${process}

echo "\n--- Launching place_p2.py ---"
# p2_output=$(python place_p2.py ${design} ${process} | tee /dev/tty)
ml_gp_init_time=$(echo "${p2_output}" | grep -oP "Elapsed time: ${ufloat}") # seconds

echo "\n--- Launching place_p3.py ---"
# p3_output=$(openroad -python -no_splash -exit place_p3.py ${design} ${process} | tee /dev/tty)
ml_placement_hpwl=$(echo "${p3_output}" | grep -oP "custom estimation of hpwl: ${uint}") # um
ml_placement_gp_and_dp_time=$(echo "${p3_output}" | grep -oP "post-intialization placement: ${ufloat}") # seconds

echo "\n--- Launching report_og_placement.py ---"
# og_report=$(openroad -python -no_splash -exit report_og_placement.py ${design} ${process} | tee /dev/tty)
og_custom_hpwl=$(echo ${og_report} | grep -oP "original placememt custom estimation of hpwl: ${uint}")
og_orfs_hpwl=$(echo ${og_report} | grep -oP "original placememt orfs estimation of hpwl: ${uint}")


echo "\n-----Summary-----\nDesign: ${design}, Process: ${process}"
echo "ml_gp_initialization time: ${ml_gp_init_time} s"
echo "ml_placement hpwl: ${ml_placement_hpwl} um"
echo "ml_placement_gp_and_dp time: ${ml_placement_gp_and_dp_time} s"
echo "og_custom hpwl: ${og_custom_hpwl} um"
echo "og_orfs hpwl: ${og_orfs_hpwl} um"

total_ml_runtime=$(awk "BEGIN {print $ml_gp_init_time + $ml_placement_gp_and_dp_time}")
echo "total runtime (given 3_2_place_iop, finshing with detailed placement): $total_ml_runtime s"

# computing final metrics

# grab og results
orfs_deliv=${ofrs_deliv_base}/${process}/${design}
time_3_3=$(awk '/Elapsed time:/ { time=$3 } END { split(time, parts, ":"); seconds=parts[1]*60 + parts[2]; print seconds }' ${orfs_deliv}/3_3*.log)
time_3_4=$(awk '/Elapsed time:/ { time=$3 } END { split(time, parts, ":"); seconds=parts[1]*60 + parts[2]; print seconds }' ${orfs_deliv}/3_4*.log)
time_3_5=$(awk '/Elapsed time:/ { time=$3 } END { split(time, parts, ":"); seconds=parts[1]*60 + parts[2]; print seconds }' ${orfs_deliv}/3_5*.log)
total_og_runtime=$(awk "BEGIN {print $time_3_3 + $time_3_4 + $time_3_5}")
total_og_runtime_no_resize=$(awk "BEGIN {print $time_3_3 + $time_3_5}")
# echo "total og runtime (including resize): ${total_og_runtime}"
echo "total og runtime (no resize, with detailed detailed placement): ${total_og_runtime_no_resize}"

# assumption: using no resize, but do include detail placement in timing

# comparison: compute speedups
runtime_ratio=$(awk "BEGIN {print $total_og_runtime_no_resize / $total_ml_runtime}")
custom_hpwl_speedup=$(awk "BEGIN {print $og_custom_hpwl / $ml_placement_hpwl}")
orfs_hpwl_speedup=$(awk "BEGIN {print $og_orfs_hpwl / $ml_placement_hpwl}")

# saving results
echo "Design: ${design}, Process: ${Process}" >> results.logs
echo "ml_runtime: ${total_ml_runtime}, og_runtime_no_resize: ${total_og_runtime_no_resize}, time_ratio: ${runtime_ratio}" >> results.log
echo "ml_hpwl: ${ml_placement_hpwl}, og_custom hpwl: ${og_custom_hpwl}, og_orfs_hpwl: ${og_orfs_hpwl}" >> results.log
echo "custom_hpwl_speedup: ${custom_hpwl_speedup}, orfs_hpwl_speedup: ${orfs_hpwl_speedup}\n" >> results.log
