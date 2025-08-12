#!/usr/bin/env Rscript

#------------------------------------------------------------------------------#
#    LIBRARIES                                                                 #
#------------------------------------------------------------------------------#

suppressPackageStartupMessages({
  library(argparse)
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
  library(tidyr)
  library(truncnorm)
  library(viridis)
  library(viridisLite)
})

#------------------------------------------------------------------------------#
#    LOAD DEFAULTS (config.R)                                                  #
#------------------------------------------------------------------------------#

source("R/config.R")               # provides nosoi_settings, param_bounds, and path vars
source("R/sample_parameters.R")
source("R/nosoi_sim.R")
source("R/run_nosoi_parallel.R")
source("R/utils.R")

#------------------------------------------------------------------------------#
#    CLI                                                                       #
#------------------------------------------------------------------------------#

`%||%` <- function(a, b) if (!is.null(a)) a else b
parse_range <- function(x, name) {
  parts <- as.numeric(strsplit(x, ",", fixed = TRUE)[[1]])
  if (length(parts) != 2 || any(is.na(parts))) {
    stop(sprintf("Flag --%s must be two comma-separated numbers, e.g. 2,21", name), call. = FALSE)
  }
  parts
}

parser <- ArgumentParser(prog = "nosoi-main", description = "Run nosoi simulations with CLI overrides")

# nosoi settings
parser$add_argument("--n-sim", type = "double", help = "Number of simulations (e.g. 400000)")
parser$add_argument("--length", type = "integer", help = "Simulation length in days")
parser$add_argument("--max-infected", type = "integer", help = "Max infected individuals")
parser$add_argument("--init-individuals", type = "integer", help = "Initial infected individuals")

# Param bounds
parser$add_argument("--mean-t-incub", help = "Range e.g. 2,21")
parser$add_argument("--stdv-t-incub", help = "Range e.g. 1,4")
parser$add_argument("--mean-ncontact", help = "Range e.g. 0.1,5")
parser$add_argument("--p-trans", help = "Range e.g. 0.01,1")
parser$add_argument("--p-fatal", help = "Range e.g. 0.01,0.5")
parser$add_argument("--mean-t-recovery", help = "Range e.g. 10,30")

# Paths
parser$add_argument("--out", dest = "output_folder", help = "Output folder (default from config.R)")
parser$add_argument("--paramsets-file", help = "Path to master.csv")
parser$add_argument("--paramsets-plot-file", help = "Path to parameter_distributions.pdf")
parser$add_argument("--parquet-file", help = "Path to nosoi_inftables.parquet")

args <- parser$parse_args()

# Apply overrides to nosoi_settings
nosoi_settings$n_sim            <- args$n_sim            %||% nosoi_settings$n_sim
nosoi_settings$length           <- args$length           %||% nosoi_settings$length
nosoi_settings$max_infected     <- args$max_infected     %||% nosoi_settings$max_infected
nosoi_settings$init_individuals <- args$init_individuals %||% nosoi_settings$init_individuals

# Apply overrides to param_bounds
if (!is.null(args$mean_t_incub))    param_bounds$mean_t_incub    <- parse_range(args$mean_t_incub,    "mean-t-incub")
if (!is.null(args$stdv_t_incub))    param_bounds$stdv_t_incub    <- parse_range(args$stdv_t_incub,    "stdv-t-incub")
if (!is.null(args$mean_ncontact))   param_bounds$mean_nContact   <- parse_range(args$mean_ncontact,   "mean-ncontact")
if (!is.null(args$p_trans))         param_bounds$p_trans         <- parse_range(args$p_trans,         "p-trans")
if (!is.null(args$p_fatal))         param_bounds$p_fatal         <- parse_range(args$p_fatal,         "p-fatal")
if (!is.null(args$mean_t_recovery)) param_bounds$mean_t_recovery <- parse_range(args$mean_t_recovery, "mean-t-recovery")

# Apply overrides to paths
output_folder       <- args$output_folder       %||% output_folder
paramsets_file      <- args$paramsets_file      %||% paramsets_file
paramsets_plot_file <- args$paramsets_plot_file %||% paramsets_plot_file
parquet_file        <- args$parquet_file        %||% parquet_file

if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

#------------------------------------------------------------------------------#
#    SAMPLE PARAMETER SPACE                                                    #
#------------------------------------------------------------------------------#

start_time <- Sys.time()

df <- resume_or_generate_parameters(
  nosoi_settings$n_sim, param_bounds, output_folder, paramsets_file, paramsets_plot_file
)

#------------------------------------------------------------------------------#
#    RUN SIMULATIONS IN PARALLEL                                               #
#------------------------------------------------------------------------------#

print_section("RUNNING NOSOI SIMULATIONS")

num_cores <- if (Sys.getenv("SLURM_CPUS_ON_NODE") != "") {
  as.numeric(Sys.getenv("SLURM_CPUS_ON_NODE"))
} else {
  max(1, detectCores() - 1)
}

cat(sprintf("Running simulations on %d cores with dynamic task allocation...\n\n", num_cores))

mc_stats <- run_nosoi_parallel(
  df, output_folder, num_cores, nosoi_settings
)

end_time <- Sys.time()
elapsed_time <- round(difftime(end_time, start_time, units = "secs"), 2)

#------------------------------------------------------------------------------#
#    FINAL SUMMARY                                                             #
#------------------------------------------------------------------------------#

print_run_summary(
  df, paramsets_file, output_folder, mc_stats, elapsed_time
)

