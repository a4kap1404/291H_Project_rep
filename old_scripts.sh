#!/bin/bash
# cp of/flow/designs

process="nangate45"
# process="asap7"
design="aes"
# design="gcd"

cp of/flow/results/${process}/${design}/base/2_floorplan.odb ./odbs
cp of/flow/results/${process}/${design}/base/2_1_floorplan.odb ./odbs
cp of/flow/results/${process}/${design}/base/2_2_floorplan_macro.odb ./odbs
cp of/flow/results/${process}/${design}/base/2_3_floorplan_tapcell.odb ./odbs
cp of/flow/results/${process}/${design}/base/2_4_floorplan_pdn.odb ./odbs

cp of/flow/results/${process}/${design}/base/3_place.odb ./odbs
cp of/flow/results/${process}/${design}/base/3_1_place_gp_skip_io.odb ./odbs
cp of/flow/results/${process}/${design}/base/3_2_place_iop.odb ./odbs
cp of/flow/results/${process}/${design}/base/3_3_place_gp.odb ./odbs
cp of/flow/results/${process}/${design}/base/3_4_place_resized.odb ./odbs
cp of/flow/results/${process}/${design}/base/3_5_place_dp.odb ./odbs

# input sdc
cp of/flow/results/${process}/${design}/base/2_1_floorplan.sdc ./sdcs


# for viewing
# openroad -gui scripts/show.tcl

# cd of/flow/results/${process}/${design}/base
# # List all files in the current directory sorted by high-precision mtime
# for f in *; do
#     if [[ -f "$f" ]]; then
#         # Get nanosecond-precision mtime using stat
#         # This works on GNU stat (Linux). BSD/macOS uses different options.
#         printf "%s\t%s\n" "$(stat --format='%Y.%N' "$f")" "$f"
#     fi
# done | sort -nr | awk -F'\t' '{print $2}'
# cd /work/CSE291/draft