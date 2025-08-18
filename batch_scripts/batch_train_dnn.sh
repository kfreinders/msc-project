#!/bin/bash

declare -a paths=(
    "data/scarce_stats/scarce_0.00.csv"
    "data/scarce_stats/scarce_0.05.csv"
    "data/scarce_stats/scarce_0.10.csv"
    "data/scarce_stats/scarce_0.15.csv"
    "data/scarce_stats/scarce_0.20.csv"
    "data/scarce_stats/scarce_0.25.csv"
    "data/scarce_stats/scarce_0.30.csv"
    "data/scarce_stats/scarce_0.35.csv"
    "data/scarce_stats/scarce_0.40.csv"
    "data/scarce_stats/scarce_0.45.csv"
    "data/scarce_stats/scarce_0.50.csv"
    "data/scarce_stats/scarce_0.55.csv"
    "data/scarce_stats/scarce_0.60.csv"
    "data/scarce_stats/scarce_0.65.csv"
    "data/scarce_stats/scarce_0.70.csv"
    "data/scarce_stats/scarce_0.75.csv"
    "data/scarce_stats/scarce_0.80.csv"
)

for level in "${paths[@]}"
do
    PYTHONPATH=python python3 -m pipelines.train_on_scarcity \
      --scarce-data-path "$level" \
      --master-path "data/nosoi/master.csv" \
      --splits-path "data/splits/$level" \
      --hparamspace-path "python/pipelines/hparamspace.json" \
      --n-trials 100 \
      --max-epochs 100 \
      --seed 42
done

