#!/bin/bash

# python3 -m flamegraph -o perf.log trial.py

# Only look at certain slice
grep "planning" perf.log > perfPlanning.log
grep "search" perf.log > perfSearch.log

# declare -a excludeMethods=("Linux" "Fedora" "Red" "Ubuntu" "Debian")
# # Exclude Certain Methods
# for val in ${includeMethods[@]}; do
# 	grep -v waiting_method perf.log > perf.log
# done


mv perf.log ~/FlameGraph/
mv perfPlanning.log ~/FlameGraph/
mv perfSearch.log ~/FlameGraph/

cd ~/FlameGraph
./flamegraph.pl --title "Overall Runtime" perf.log > perfPlanning.svg 
./flamegraph.pl --title "Planning Runtime" perfPlanning.log > perfPlanning.svg 
./flamegraph.pl --title "Filtered for Search" perfSearch.log > perfSearch.svg 

gio open perfPlanning.svg
gio open perfSearch.svg