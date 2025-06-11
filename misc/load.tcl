utl::set_metrics_stage "detailedplace__{}"

read_db odbs/3_5_place_dp.odb
# report_wirelength
report_metrics

report_metrics 3 "detailed place" true false


exit