# Inferring *nosoi* Simulation Parameters from Summary Statistics

This project explores the ability of deep neural networks (DNNs) to recover the
original parameters of *nosoi* agent-based simulations from a set of summary
statistics. The focus is on how data scarcity impacts the quality of parameter
inference, with future work extending toward graph-based learning using Graph
Neural Networks (GNNs).

---

## Project Goals

| Task                                                                  | Status         |
|-----------------------------------------------------------------------|----------------|
| Implement optimal parameter space exploration for *nosoi* simulations | ✅ Done        |
| Set up a parallel processing pipeline for *nosoi* simulations         | ✅ Done        |
| Effectively compress full *nosoi* transmission chains                 | ✅ Done        |
| Build a DNN to predict simulation parameters from summary statistics  | ✅ Done        |
| Hyperparameter tuning of the DNN to improve performance               | ✅ Done        |
| Further optimize model performance by transforming skewed parameters  | ✅ Done        |
| Create modular framework for different data degradation strategies    | ✅ Done        |
| Quantify performance under various scenarios of data scarcity         | 🔄 In progress |
| Investigate how simulation size affects parameter inference           | ⏳ Planned     |
| Parameter sensitivity analysis for summary statistics                 | ⏳ Planned     |
| Compute SHAP-values for summary statistics                            | ⏳ Planned     |
| Explore the use of Graph Neural Networks                              | ⏳ Planned     |
| Explore the use of Approximate Bayesian Computation                   | ⏳ Planned     |


## Usage

---

### 1. Set *nosoi* Simulation Settings (R)

Edit [`R/config.R`](R/config.R) to set:

* The simulation parameters:
    * The number of simulations
    * The maximum simulation length in days
    * The maximum allowed number of infected individuals
    * The number of initially infected individuals
    * The target output directory

* Lower and upper bounds for each parameter of interest:
    * Mean incubation time
    * Standard deviation in incubation time
    * Mean number of contacts per individual per time step
    * Probability of transmission
    * Probability of death
    * Recovery time

Latin Hypercube Sampling (LHS) will be used to sample the specified parameter
space, to ensure as many different parameter combinations as possible are
simulated.


### 2. Run Parallel *nosoi* Simulations

The *nosoi* simulations batch script
[`batch_scripts/batch.sh`](batch_scripts/batch.sh) is written for use on a
cluster with SLURM installed. It will run one *nosoi* simulation per allocated
CPU core. If the environmental variable `SLURM_CPUS_ON_NODE` is not found, for
example when doing local testing the script will automatically use all
available cores except one. Make sure to take a look at [`R/main.R`](R/main.R)
for the full pipeline.

Finally, make sure to set the working directory in
[`batch_scripts/batch.sh`](batch_scripts/batch.sh) to where you cloned this
repo and adjust the job name or allocated resources. Then, submit the job with:

```bash
sbatch batch_scripts/batch.sh
```

With current settings and resources, this script takes about 60 hours. It will
also write a number of files to [`data/nosoi`](data/nosoi):

| File                                 | Purpose                                                           |
|--------------------------------------|-------------------------------------------------------------------|
| master.csv                           | Contains simulation seeds and used *nosoi* parameters             |
| summary_stats_export.csv             | Contains summary statistics computed over the transmission chains |
| summary_statistics_distributions.pdf | Plots of all summary statistic distributions                      |
| inftable_xxxxxxxxxx_mapped.parquet   | Full transmission chain, compressed and stored as Apache Parquet  |

Note that the script outputs a single Parquet file per transmission chain,
meaning that for 400,000 simulations you will get as many Parquet files. To
download them from a cluster, it is more I/O friendly to first archive them
with:

```bash
sbatch batch_scripts/tar_inftables.sh
```
