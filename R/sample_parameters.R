#------------------------------------------------------------------------------#
#    PARAMETER SET GENERATION                                                  #
#------------------------------------------------------------------------------#

generate_parameters <- function(amount, param_bounds) {
  # Ensure `amount` is a valid positive integer
  if (!is.numeric(amount) || amount <= 0 || amount != as.integer(amount)) {
    stop("Error: `amount` must be a positive integer.")
  }
  
  # Sample parameters from uniform distributions
  df <- as.data.frame(
    map(param_bounds, ~ runif(amount, .x[1], .x[2]))
  )
  
  # Add unique seed column
  df <- cbind(seed = sample(.Machine$integer.max, amount, replace = FALSE), df)
  
  return(df)
}

#------------------------------------------------------------------------------#
#    PARAMETER SET VALIDATION FUNCTION                                         #
#------------------------------------------------------------------------------#

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
    min_val <- param_bounds[[param]][1]
    max_val <- param_bounds[[param]][2]
    if (any(df[[param]] < min_val | df[[param]] > max_val)) {
      stop(paste("Error: Values out of range for", param))
    }
  }
}

