#!/bin/bash
# This script is intended to be run to perform code profiling
# Before this code is run a flamegraph .log file has to be made
# This requires use as seen in: https://gist.github.com/joshdover/1b9c86ef427b1506b7a5dd2509309674

flamegraph_script_path=$1
flamegraph_log_path=$2
flamegraph_image_path=$3
$flamegraph_script_path --title "Flamegraph Stack Traces" $flamegraph_log_path > $flamegraph_image_path
gio open $flamegraph_image_path