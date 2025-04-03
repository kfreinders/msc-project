#------------------------------------------------------------------------------#
#    LIBRARIES                                                                 #
#------------------------------------------------------------------------------#

suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(DBI)
  library(dplyr)
  library(ggplot2)
  library(igraph)
  library(lhs)
  library(nosoi)
  library(parallel)
  library(purrr)
  library(RSQLite)
  library(tidyr)
  library(truncnorm)
  library(viridis)
  library(viridisLite)
})

source("R/config.R")
source("R/sample_parameters.R")
source("R/nosoi_sim.R")
source("R/summary_statistics.R")
source("R/sqlite.R")
source("R/run_nosoi_parallel.R")
source("R/utils.R")

#------------------------------------------------------------------------------#
#    SAMPLE PARAMETER SPACE                                                    #
#------------------------------------------------------------------------------#

# Time current process
start_time <- Sys.time()

# Get full parameter sets to simulate or remaining parameter sets when resuming
df <- resume_or_generate_parameters(
  nosoi_settings$n_sim, param_bounds, output_folder, paramsets_file, paramsets_plot_file
)

#------------------------------------------------------------------------------#
#    RUN SIMULATIONS IN PARALLEL                                               #
#------------------------------------------------------------------------------#

print_section("RUNNING NOSOI SIMULATIONS")

# Create a table in an SQLite database to store generated summary statistics
initialize_db(db_name)
cat("SQLite database connection successfully established\n")

# Get the number of available cores
num_cores <- if (Sys.getenv("SLURM_CPUS_ON_NODE") != "") {
  as.numeric(Sys.getenv("SLURM_CPUS_ON_NODE"))  # Running as a SLURM job
} else {
  detectCores() - 1  # When running locally, leave one core for stability
}

# Start and time the simulations
cat(sprintf(
  "Running simulations on %d cores with dynamic task allocation...\n\n",
  num_cores
))
mc_stats <- run_nosoi_parallel(
  df, db_name, output_folder, num_cores, nosoi_settings
)

# Export the summary_statistics to a csv
export_db_to_csv(db_name, ss_filename)

end_time <- Sys.time()  # End timing
elapsed_time <- round(difftime(end_time, start_time, units = "secs"), 2)

#------------------------------------------------------------------------------#
#    FINAL SUMMARY                                                             #
#------------------------------------------------------------------------------#

print_run_summary(
  df, paramsets_file, ss_filename, output_folder, mc_stats, elapsed_time
)

