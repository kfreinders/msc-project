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

df <- resume_or_generate_parameters(
  n_sim, param_bounds, output_folder, paramsets_file, paramsets_plot_file
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
cat(sprintf("Running simulations on %d cores with dynamic task allocation...\n\n", num_cores))
start_time <- Sys.time()  # Start timing
output_files <- run_nosoi_parallel(paramsets_file, db_name, output_folder, num_cores)
end_time <- Sys.time()  # End timing

# Compute and format total elapsed time in seconds
formatted_time <- format_elapsed_time(
  round(difftime(end_time, start_time, units = "secs"), 2)
)

#------------------------------------------------------------------------------#
#    FINAL SUMMARY                                                             #
#------------------------------------------------------------------------------#

print_section("SUMMARY")

valid_files <- output_files[!sapply(output_files, is.null)]
successful_runs <- length(valid_files)

print_run_summary(successful_runs, n_sim, db_name, ss_filename, parquet_file)

