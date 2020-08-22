#!/bin/bash
# This script is intended to be run to perform code profiling
# Before this code is run a flamegraph .log file has to be made
# This requires use as seen in: https://gist.github.com/joshdover/1b9c86ef427b1506b7a5dd2509309674

# Only look at certain slice
grep "planning" perf.log > perfPlanning.log
grep "search" perf.log > perfSearch.log

mv perf.log ~/FlameGraph/
mv perfPlanning.log ~/FlameGraph/
mv perfSearch.log ~/FlameGraph/

cd ~/FlameGraph
./flamegraph.pl --title "Overall Runtime" perf.log > perfPlanning.svg 
./flamegraph.pl --title "Planning Runtime" perfPlanning.log > perfPlanning.svg 
./flamegraph.pl --title "Filtered for Search" perfSearch.log > perfSearch.svg 

gio open perfPlanning.svg
gio open perfSearch.svg