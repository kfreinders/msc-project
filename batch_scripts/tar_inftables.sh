#!/bin/bash
#SBATCH --job-name=make_tarball
#SBATCH --time=00-00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=make_tarball_%j.out
#SBATCH --error=make_tarball_%j.err

# Move to project root
cd /home4/s3919323/msc-project
echo -e "\nWorking directory: $(pwd)"

find data/nosoi/ -name "inftable_*.parquet" -print0 | \
  tar --null -cf - --files-from=- | pigz -p $SLURM_CPUS_PER_TASK > nosoi_inftables.tar.gz

