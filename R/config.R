nosoi_settings <- list(
  n_sim = 4e5,            # Number of simulations to run
  length = 100,           # Simulation length in days
  max_infected = 10000,   # Maximum number of infected individuals
  init_individuals = 1    # Initial number of infected individuals
)

param_bounds <- list(
  mean_t_incub  = c(2, 21),
  stdv_t_incub  = c(1, 4),
  mean_nContact = c(0.1, 5),
  p_trans       = c(0.01, 1),
  p_fatal       = c(0.01, 0.5),
  t_recovery    = c(10, 30)
)

output_folder         <- "data/nosoi"
paramsets_file        <- file.path(output_folder, "master.csv")
paramsets_plot_file   <- file.path(output_folder, "parameter_distributions.pdf")
db_name               <- file.path(output_folder, "simulation_results.db")
ss_filename           <- file.path(output_folder, "summary_stats_export.csv")
parquet_file          <- file.path(output_folder, "nosoi_inftables.parquet")

