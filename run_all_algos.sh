#!/bin/bash

algorithms=("bf" "bndm" "cosinescreendq" "cosinescreendqlin" "kmp" "kr")

for algo in "${algorithms[@]}"; do
    echo "=================================="
    echo "Running: $algo"
    echo "=================================="
    
    gcc source/algos/${algo}.c -O3 -g -msse4 -lm -o source/bin/${algo}
    ./select -none ${algo}
    valgrind --tool=cachegrind --cache-sim=yes ./smart -text genome -pset 50 -plen 24 24 2>&1 | tee ${algo}_cachegrind.log
    
    echo "----- Miss rates for $algo -----"
    grep "miss rate" ${algo}_cachegrind.log
    echo ""
done

echo "All done. Log files created:"
ls *_cachegrind.log
