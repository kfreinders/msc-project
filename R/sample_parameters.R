#------------------------------------------------------------------------------#
#    PARAMETER SET GENERATION                                                  #
#------------------------------------------------------------------------------#

generate_parameters <- function(amount, param_bounds) {
  # Ensure `amount` is a valid positive integer
  if (!is.numeric(amount) || amount <= 0 || amount != as.integer(amount)) {
    stop("Error: `amount` must be a positive integer.")
  }

  validate_bounds(param_bounds)

  # Identify variable and fixed parameters
  is_fixed <- map_lgl(param_bounds, ~ .x[1] == .x[2])
  variable_bounds <- param_bounds[!is_fixed]
  fixed_values <- map(param_bounds[is_fixed], ~ rep(.x[1], amount))

  # Latin Hypercube Sampling for variable parameters
  if (length(variable_bounds) > 0) {
    num_vars <- length(variable_bounds)
    lhs_matrix <- randomLHS(amount, num_vars)

    # Scale each column to its corresponding bounds
    scaled_vars <- map2_dfc(
      as.data.frame(lhs_matrix),
      variable_bounds,
      ~ .x * diff(.y) + .y[1]
    )

    names(scaled_vars) <- names(variable_bounds)
  } else {
    scaled_vars <- data.frame()
  }

  # Combine with fixed parameters
  df_fixed <- as.data.frame(fixed_values)
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

    # Make sure we have both a left and right bound, also for fixed values
    if (length(bounds) != 2) {
      stop(sprintf(
        "Error: missing bounds for parameter `%s`. If you intended a fixed value, set left bound = right bound.",
        param
      ))
    }

    # Make sure left bound <= right bound
    if (bounds[1] > bounds[2]) {
      stop("Error: left bound greater than right bound.")
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

    # Handle range
    if (bounds[1] != bounds[2]) {
      min_val <- bounds[1]
      max_val <- bounds[2]
      if (any(df[[param]] < min_val | df[[param]] > max_val)) {
        stop(paste("Error: Values out of range for", param))
      }
    # Handle fixed value
    } else {
      fixed_val <- bounds[1]
      if (any(df[[param]] != fixed_val)) {
        stop(paste("Error: Values out of range for", param))
      }
    }
  }
}

