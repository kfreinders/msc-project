#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=2GB

echo -e '################################################################################'
echo -e '#                              NOSOI SIMULATIONS                               #'
echo -e '#                          PARALLEL PROCESSING SCRIPT                          #'
echo -e '################################################################################'

cd "$(dirname "$0")/.."  # Move to project root
Rscript R/main.R
