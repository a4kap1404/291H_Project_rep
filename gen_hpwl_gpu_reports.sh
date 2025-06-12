#!/bin/bash
set -e

# launch with bash

# Description: After running gen_gpu_ml_placements.sh to get gpu timings and placements from pytorch model, run place_p3 and report_og.placement, 
# and write them into a log file



ml_placement_dir="odbs"
ofrs_deliv_base="ofrs_deliv"
gpu_log_file="gpu_ml_placements.log"
output_log_file="results.log"

processes=("nangate45" "asap7")
designs=("aes" "gcd" "ibex" "jpeg")



# note: "og" stands for original (vanilla openroad)

echo "---------------------------------------------" >> "${output_log_file}"

# for parsing
ufloat="\K[0-9]+.[0-9]+"
uint="\K[0-9]+"

for process in "${processes[@]}"; do
  for design in "${designs[@]}"; do

    echo "Running for process=$process, design=$design"


    # grab gpu timing from log file
    ml_gp_init_time=$(tac "$gpu_log_file" | awk -v design="$design" -v process="$process" '
    $0 ~ "Design: " design ", Process: " process {
        found = 1
        next
    }
    found && $0 ~ /^time \(s\):/ {
        print $3
        exit
    }
    ')

    echo "\n--- Launching place_p3.py ---"
    p3_output=$(openroad -python -no_splash -exit place_p3.py ${design} ${process} | tee /dev/tty)
    ml_placement_hpwl=$(echo "${p3_output}" | grep -oP "custom estimation of hpwl: ${uint}") # um
    ml_placement_gp_and_dp_time=$(echo "${p3_output}" | grep -oP "post-intialization placement: ${ufloat}") # seconds

    echo "\n--- Launching report_og_placement.py ---"
    og_report=$(openroad -python -no_splash -exit report_og_placement.py ${design} ${process} | tee /dev/tty)
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
    echo "total og runtime (no resize, with detailed detailed placement): ${total_og_runtime_no_resize} s"

    # assumption: using no resize, but do include detail placement in timing

    # comparison: compute speedups
    runtime_speedup=$(awk "BEGIN {print $total_og_runtime_no_resize / $total_ml_runtime}")
    custom_hpwl_speedup=$(awk "BEGIN {print $og_custom_hpwl / $ml_placement_hpwl}")
    orfs_hpwl_speedup=$(awk "BEGIN {print $og_orfs_hpwl / $ml_placement_hpwl}")

    echo "time_speedup: ${runtime_speedup}"
    echo "custom_hpwl_speedup: ${custom_hpwl_speedup}"
    echo "orfs_hpwl_speedup: ${orfs_hpwl_speedup}\n"

    # saving results
    current_date=$(date)
    echo "Design: ${design}, Process: ${process} | ${current_date}" >> "$output_log_file"
    echo "ml_runtime: ${total_ml_runtime}, og_runtime_no_resize: ${total_og_runtime_no_resize}, runtime_speedup: ${runtime_speedup}" >> "$output_log_file"
    echo "ml_hpwl: ${ml_placement_hpwl}, og_custom hpwl: ${og_custom_hpwl}, og_orfs_hpwl: ${og_orfs_hpwl}" >> "$output_log_file"
    echo "custom_hpwl_speedup: ${custom_hpwl_speedup}, orfs_hpwl_speedup: ${orfs_hpwl_speedup}\n" >> "$output_log_file"

    done
done