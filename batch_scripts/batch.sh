#!/bin/bash
#SBATCH --job-name=nosoi_parallel
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=26GB
#SBATCH --output=nosoi_parallel_%j.out
#SBATCH --error=nosoi_parallel_%j.err

echo -e '################################################################################'
echo -e '#                              NOSOI SIMULATIONS                               #'
echo -e '#                          PARALLEL PROCESSING SCRIPT                          #'
echo -e '################################################################################'

# Move to project root
cd /home4/s3919323/msc-project
echo -e "\nWorking directory: $(pwd)"

# Purge all modules and load R
module purge
module load GDAL/3.9.0-foss-2023b
module load R/4.4.1-gfbf-2023b

Rscript R/main.R
