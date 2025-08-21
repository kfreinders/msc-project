#!/bin/bash

for i in $(seq 0.2 0.2 2.0)
do
    echo "--------------------------------------------------------------------------------"
    echo " epsilon = $i"
    echo "--------------------------------------------------------------------------------"
    PYTHONPATH=python python3 -m analysis.abc --method rejection --epsilon "$i"
    echo
done
