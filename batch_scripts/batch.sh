#!/bin/bash
#SBATCH --job-name=nosoi_parallel
#SBATCH --time=02:00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=4GB
#SBATCH --output=nosoi_parallel_%j.out
#SBATCH --error=nosoi_parallel_%j.err

echo -e '################################################################################'
echo -e '#                              NOSOI SIMULATIONS                               #'
echo -e '#                          PARALLEL PROCESSING SCRIPT                          #'
echo -e '################################################################################'

cd "$(dirname "$0")/.."  # Move to project root
echo "Working directory: $(pwd)"

# Purge all modules and load R
module purge
module load GDAL/3.9.0-foss-2023b
module load R/4.4.1-gfbf-2023b

Rscript R/main.R
