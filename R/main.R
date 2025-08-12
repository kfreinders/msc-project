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
#    IMPORTS                                                                   #
#------------------------------------------------------------------------------#

source("R/defaults.R")
source("R/sample_parameters.R")
source("R/nosoi_sim.R")
source("R/run_nosoi_parallel.R")
source("R/utils.R")

#------------------------------------------------------------------------------#
#    CLI                                                                       #
#------------------------------------------------------------------------------#

`%||%` <- function(a, b) if (!is.null(a)) a else b
parse_range <- function(x, name) {
  parts <- strsplit(x, ",", fixed = TRUE)[[1]]
  parts <- trimws(parts)
  nums  <- suppressWarnings(as.numeric(parts))

  if (length(nums) == 1 && !is.na(nums[1])) {
    return(c(nums[1], nums[1]))
  }
  if (length(nums) == 2 && all(!is.na(nums))) {
    return(as.numeric(nums))
  }

  stop(sprintf("Flag --%s must be a number or two comma-separated numbers (e.g. 5  or  2,21).", name),
       call. = FALSE)
}

parser <- ArgumentParser(prog = "nosoi-main", description = "Run nosoi simulations with CLI overrides")

# nosoi settings
parser$add_argument("--n-sim", type = "double", help = "Number of simulations (e.g. 400000)")
parser$add_argument("--length", type = "integer", help = "Simulation length in days")
parser$add_argument("--max-infected", type = "integer", help = "Max infected individuals")
parser$add_argument("--init-individuals", type = "integer", help = "Initial infected individuals")

# Param bounds
parser$add_argument("--mean-t-incub",    help = "Number (fixed) or range like 2,21")
parser$add_argument("--stdv-t-incub",    help = "Number (fixed) or range like 1,4")
parser$add_argument("--mean-ncontact",   help = "Number (fixed) or range like 0.1,5")
parser$add_argument("--p-trans",         help = "Number (fixed) or range like 0.01,1")
parser$add_argument("--p-fatal",         help = "Number (fixed) or range like 0.01,0.5")
parser$add_argument("--mean-t-recovery", help = "Number (fixed) or range like 10,30")


# Paths
parser$add_argument("--out", dest = "output_folder", help = "Output folder")

args <- parser$parse_args()

#------------------------------------------------------------------------------#
#    BUILD CONTEXT                                                             #
#------------------------------------------------------------------------------#

config <- defaults

# overrides: nosoi_settings
config$nosoi_settings$n_sim            <- args$n_sim            %||% config$nosoi_settings$n_sim
config$nosoi_settings$length           <- args$length           %||% config$nosoi_settings$length
config$nosoi_settings$max_infected     <- args$max_infected     %||% config$nosoi_settings$max_infected
config$nosoi_settings$init_individuals <- args$init_individuals %||% config$nosoi_settings$init_individuals

# overrides: param_bounds
if (!is.null(args$mean_t_incub))    config$param_bounds$mean_t_incub    <- parse_range(args$mean_t_incub,    "mean-t-incub")
if (!is.null(args$stdv_t_incub))    config$param_bounds$stdv_t_incub    <- parse_range(args$stdv_t_incub,    "stdv-t-incub")
if (!is.null(args$mean_ncontact))   config$param_bounds$mean_nContact   <- parse_range(args$mean_ncontact,   "mean-ncontact")
if (!is.null(args$p_trans))         config$param_bounds$p_trans         <- parse_range(args$p_trans,         "p-trans")
if (!is.null(args$p_fatal))         config$param_bounds$p_fatal         <- parse_range(args$p_fatal,         "p-fatal")
if (!is.null(args$mean_t_recovery)) config$param_bounds$mean_t_recovery <- parse_range(args$mean_t_recovery, "mean-t-recovery")

config$paths$output_folder <- args$output_folder %||% config$paths$output_folder

# Hardcode filenames inside output folder
config$paths$paramsets_file      <- file.path(config$paths$output_folder, "master.csv")
config$paths$paramsets_plot_file <- file.path(config$paths$output_folder, "parameter_distributions.pdf")

if (!dir.exists(config$paths$output_folder)) {
  dir.create(config$paths$output_folder, recursive = TRUE, showWarnings = FALSE)
}


if (!is.null(args$seed)) set.seed(args$seed)

#------------------------------------------------------------------------------#
#    SAMPLE PARAMETER SPACE                                                    #
#------------------------------------------------------------------------------#

start_time <- Sys.time()

df <- resume_or_generate_parameters(
  config$nosoi_settings$n_sim,
  config$param_bounds,
  config$paths$output_folder,
  config$paths$paramsets_file,
  config$paths$paramsets_plot_file
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
  df,
  config$paths$output_folder,
  num_cores,
  config$nosoi_settings
)

end_time <- Sys.time()
elapsed_time <- round(difftime(end_time, start_time, units = "secs"), 2)

#------------------------------------------------------------------------------#
#    FINAL SUMMARY                                                             #
#------------------------------------------------------------------------------#

print_run_summary(
  df,
  config$paths$paramsets_file,
  config$paths$output_folder,
  mc_stats,
  elapsed_time
)

