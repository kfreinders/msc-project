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
