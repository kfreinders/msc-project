#------------------------------------------------------------------------------#
#    PARAMETER SET GENERATION                                                  #
#------------------------------------------------------------------------------#

normalize_param_bounds <- function(param_bounds) {
  lapply(param_bounds, function(x) {
    if (length(x) == 1) rep(x, 2) else x
  })
}

generate_parameters <- function(amount, param_bounds) {
  # Ensure `amount` is a valid positive integer
  if (!is.numeric(amount) || amount <= 0 || amount != as.integer(amount)) {
    stop("Error: `amount` must be a positive integer.")
  }

  # Validate bounds
  validate_bounds(param_bounds)

  # Identify variable and fixed parameters
  is_fixed <- vapply(param_bounds, function(x) x[1] == x[2], logical(1))
  variable_bounds <- param_bounds[!is_fixed]
  fixed_values <- lapply(param_bounds[is_fixed], function(x) rep(x[1], amount))

  # Latin Hypercube Sampling for variable parameters
  if (length(variable_bounds) > 0) {
    num_vars <- length(variable_bounds)
    lhs_matrix <- randomLHS(amount, num_vars)

    # Scale each column to its corresponding bounds
    scaled_vars <- Map(
      function(col, bounds) col * diff(bounds) + bounds[1],
      as.data.frame(lhs_matrix), variable_bounds
    )

    scaled_vars <- as.data.frame(scaled_vars)
    names(scaled_vars) <- names(variable_bounds)
  } else {
    scaled_vars <- data.frame()
  }

  # Combine with fixed parameters
  df_fixed <- if (length(fixed_values) > 0) {
    as.data.frame(fixed_values)
  } else {
    data.frame(row.names = seq_len(amount))  # empty df with correct row count
  }

  df <- cbind(scaled_vars, df_fixed)

  # Ensure original parameter order is preserved
  df <- df[names(param_bounds)]

  # Add unique seed column
  df <- cbind(seed = sample(.Machine$integer.max, amount, replace = FALSE), df)

  return(df)
}

#------------------------------------------------------------------------------#
#    PARAMETER SET VALIDATION FUNCTION                                         #
#------------------------------------------------------------------------------#

validate_bounds <- function(param_bounds) {
  for (param in names(param_bounds)) {
    bounds <- param_bounds[[param]]

    # Check: numeric and length 2
    if (!is.numeric(bounds) || length(bounds) != 2) {
      stop(sprintf(
        "Error: parameter `%s` must be a numeric vector of length 2. Use c(x, x) for fixed values.",
        param
      ))
    }

    # Check: left bound <= right bound
    if (bounds[1] > bounds[2]) {
      stop(sprintf(
        "Error: for parameter `%s`, lower bound (%g) is greater than upper bound (%g).",
        param, bounds[1], bounds[2]
      ))
    }
  }
}

validate_parameters <- function(df, param_bounds) {
  if (any(duplicated(df$seed))) {
    stop("Error: Duplicate seeds detected! Re-generate seeds with unique values.")
  }


  # Ensure no NA values
  if (any(is.na(df))) {
    stop("Error: NA values in sampled parameters.")
  }

  # Ensure all values are within expected ranges
  for (param in names(param_bounds)) {
    bounds <- param_bounds[[param]]

    min_val <- bounds[1]
    max_val <- bounds[2]

    if (min_val == max_val) {
      # Fixed parameter
      if (any(df[[param]] != min_val)) {
        stop(sprintf("Error: Parameter `%s` should be fixed at %g, but has deviating values.", param, min_val))
      }
    } else {
      # Ranged parameter
      if (any(df[[param]] < min_val | df[[param]] > max_val)) {
        stop(sprintf("Error: Parameter `%s` has values outside the range [%g, %g].", param, min_val, max_val))
      }
    }
  }
}

#------------------------------------------------------------------------------#
#    RESUMING PRODUCTION RUNS                                                  #
#------------------------------------------------------------------------------#

find_remaining <- function(n_sim, output_folder, paramsets_file) {
  # Resume simulations by comparing existing Parquet files with the seeds in
  # `paramsets_file`, if it exists
  cat(sprintf("Existing master file found at '%s'\n", paramsets_file))
  df <- fread(paramsets_file)

  existing_files <- list.files(output_folder, pattern = "^inftable_\\d{10}_mapped\\.parquet$")
  completed_seeds <- as.integer(
    sub("^inftable_(\\d{10})_mapped\\.parquet$", "\\1", existing_files)
  )
  df <- df[!(df$seed %in% completed_seeds), ]

  if (nrow(df) > 0) {
    cat(sprintf(
      "Resuming run with %s remaining simulations\n",
      format(n_sim, big.mark = ",", decimal.mark = ".", scientific = FALSE)
    ))
  } else {
    cat("All simulations already completed. Nothing to do.\n")
    quit(save = "no")
  }

  return(df)
}

generate_parameters <- function(
  n_sim,
  param_bounds,
  output_folder,
  paramsets_file,
  plot_file
) {

  # If `paramsets_file` does not exist, create a new parameter set
  print_section("GENERATING PARAMETER DISTRIBUTIONS")
  # Normalize param_bounds to 2-element vector to handle fixed values
  param_bounds <- normalize_param_bounds(param_bounds)
  df <- generate_parameters(n_sim, param_bounds)
  print_param_bounds(param_bounds)
  validate_parameters(df, param_bounds)
  cat(sprintf(
    "Successfully generated %s unique parameter sets\n",
    format(n_sim, big.mark = ",", decimal.mark = ".", scientific = FALSE)
  ))

  if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)
  write.csv(df, paramsets_file, row.names = FALSE)
  cat(sprintf("Saved parameter sets to '%s'\n", paramsets_file))

  plot_parameter_distributions(df, plot_file)
  cat(sprintf("Saved plot of parameter sets distribution to '%s'\n", plot_file))

  return(df)
}

