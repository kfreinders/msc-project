#!/bin/bash
#SBATCH --job-name=create_scarce_data
#SBATCH --time=02-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --output=create_scarce_data_%j.out
#SBATCH --error=create_scarce_data_%j.err

# Move to project root
cd /scratch/s3919323/msc-project

# Purge all modules and Python
module purge
module load Python/3.10.8-GCCcore-12.2.0

# Set 'python' scripts directory as pythonpath and run main script
PYTHONPATH=python python3 -m pipelines.create_scarce_data
