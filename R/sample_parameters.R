#------------------------------------------------------------------------------#
#    PARAMETER SET GENERATION                                                  #
#------------------------------------------------------------------------------#

generate_parameters <- function(amount, param_bounds) {
  # Ensure `amount` is a valid positive integer
  if (!is.numeric(amount) || amount <= 0 || amount != as.integer(amount)) {
    stop("Error: `amount` must be a positive integer.")
  }

  validate_bounds(param_bounds)
  
  # Sample parameters from uniform distributions
  df <- as.data.frame(
    map(param_bounds, function(bounds) {
      if (bounds[1] != bounds[2]) {
        # If the bounds differ, sample in that range
        runif(amount, bounds[1], bounds[2])
      } else {
        # If both bounds are the same, return a constant vector
        rep(bounds[1], amount)
      }
    })
  )
  
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

