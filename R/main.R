#------------------------------------------------------------------------------#
#    LIBRARIES                                                                 #
#------------------------------------------------------------------------------#

suppressPackageStartupMessages({
  library(data.table)
  library(DBI)
  library(dplyr)
  library(ggplot2)
  library(igraph)
  library(nosoi)
  library(parallel)
  library(purrr)
  library(RSQLite)
  library(tidyr)
  library(truncnorm)
  library(viridis)
  library(viridisLite)
})

source("../R/sample_parameters.R")
source("../R/nosoi_sim.R")
source("../R/summary_statistics.R")
source("../R/sqlite.R")
source("../R/run_nosoi_parallel.R")
source("../R/utils.R")

#------------------------------------------------------------------------------#
#    CONFIGURATION                                                             #
#------------------------------------------------------------------------------#

# nosoi parameter bounds
n_sim <- 1e1                    # Number of simulations to run
param_bounds <- list(
  mean_t_incub  = c(2, 21),     # Mean incubation time bounds
  stdv_t_incub  = c(1, 4),      # Incubation time standard deviation bounds
  mean_nContact = c(0.1, 5),    # Mean number of contacts bounds
  p_trans       = c(0.01, 1),   # Transmission probability bounds
  p_fatal       = c(0.01, 0.5)  # Fatality probability bounds
)

# Parallel processing and file names
num_cores <- detectCores()
output_folder <- "../data/nosoi/"
paramsets_file <- "master.csv"
paramsets_plot_file <- "parameter_distributions.pdf"
db_name <- "simulation_results.db"
ss_filename <- "summary_stats_export.csv"

#------------------------------------------------------------------------------#
#    SAMPLE PARAMETER SPACE                                                    #
#------------------------------------------------------------------------------#

# Set file save location to output_folder
paramsets_file <- paste(output_folder, "master.csv", sep = "")
paramsets_plot_file <- paste(output_folder, "parameter_distributions.pdf", sep = "")
db_name <- paste(output_folder, "simulation_results.db", sep = "")
ss_filename <- paste(output_folder, "summary_stats_export.csv", sep = "")

print_section("GENERATING PARAMETER DISTRIBUTIONS")

# Sample parameter sets
print_param_bounds(param_bounds)
df <- generate_parameters(n_sim, param_bounds)
validate_parameters(df, param_bounds)
cat(sprintf("Successfully generated %d unique parameter sets\n", n_sim))

# Save the parameter sets to a CSV
if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)
write.csv(df, paramsets_file, row.names = FALSE)
cat(sprintf("Saved parameter sets to '%s'\n", paramsets_file))

# Save a plot of the parameter sampling
plot_parameter_distributions(df, paramsets_plot_file)
cat(sprintf("Saved plot of parameter sets distribution to '%s'\n", paramsets_plot_file))

#------------------------------------------------------------------------------#
#    RUN SIMULATIONS IN PARALLEL                                               #
#------------------------------------------------------------------------------#

print_section("RUNNING NOSOI SIMULATIONS")

# Create a table in an SQLite database to store generated summary statistics
initialize_db(db_name)
cat("SQLite database connection successfully established\n")

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

if (successful_runs > 0) {
  # Export summary statistics if at least one simulation was successful
  export_db_to_csv(db_name, ss_filename)
  
  # Check if any simulations failed and print batch statistics
  if (successful_runs == n_sim) {
    cat("All simulations completed successfully\n")
  } else {
    cat(sprintf("Simulations successful: %d\n", successful_runs))
    cat(sprintf("Simulations failed: %d\n", n_sim - successful_runs))
  }
  
  cat("Total runtime:", formatted_time, "\n")
  cat(sprintf("Full infection tables saved in: %s\n", output_folder))
  cat(sprintf("Summary statistics exported to: %s\n\n", ss_filename))
} else {
  cat("No valid results were generated\n\n")
}