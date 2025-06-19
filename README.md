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


### 3. Degrade Transmission Chains and Compute Summary Statistics

Real-world epidemiological data is seldom complete. To simulate this, we
introduce artificial data scarcity in the transmission chains using
RandomNodeDrop:
* Randomly sample a certain percentage of nodes in the transmission chain and delete them
* Children of deleted nodes are reconnected to their parent, simulating undocumented infections

Settings like which percentages of RandomNodeDrop to apply can be set in
[`python/create_scarce_data.py`](python/create_scarce_data.py), and the SLURM
job name and resources can be set in
[`batch_scripts/create_scarce_data.sh`](batch_scripts/create_scarce_data.sh).
Finally, to submit the job, run:

```bash
sbatch batch_scripts/create_scarce_data.sh
```

With current settings and resources, this script takes about 48 hours. It will
not save the full mutated transmission chains (but is fully reproducible with a
seed) but instead saves the recomputed summary statistics for each level of
scarcity.


### 4. Evaluate DNN Performance Under Data Scarcity

Once the summary statistics have been generated for all scarcity levels, we
train a deep neural network (DNN) to predict the original *nosoi* simulation
parameters from these degraded features. The output paths and hyperparameter
search space can be adjusted in
[`python/pipelines/evaluate_scarcity.py/`](batch_scripts/create_scarce_data.sh).

Finally, launch the tuning and training and training pipeline:

```bash
python -m pipelines.train_on_scarcity
```

This will:

* Load the recomputed summary statistics and matching parameters
* Split data into train/validation/test sets, if not already done
* Perform hyperparameter tuning using Optuna
* Train a DNN for each scarcity level (e.g., 5% to 50% RandomNodeDrop)
* Save the best model and performance metrics for each level

Each trained model and associated metrics are stored in [`data/dnn`](data/dnn)
under a directory named after its scarcity level (e.g., `scarce_0.20/`),
including:

---


## License

MIT License.
