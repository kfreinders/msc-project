#------------------------------------------------------------------------------#
#    PRINT LOGGING SECTION HEADERS                                             #
#------------------------------------------------------------------------------#

print_section <- function(s, termwidth = 80) {
  # Calculate padding lengths
  leftlen <- floor((termwidth - nchar(s)) / 2) - 1
  rightlen <- ceiling((termwidth - nchar(s)) / 2) - 1
  
  # Construct the formatted section header
  full <- paste0("\n", strrep("=", leftlen), " ", s, " ", strrep("=", rightlen), "\n") 
  
  # Print the output
  cat(full, "\n")
}

#------------------------------------------------------------------------------#
#    PRINT PARAMATER BOUNDS TABLE                                              #
#------------------------------------------------------------------------------#

print_param_bounds <- function(param_bounds) {
  cat("Parameter sampling space:\n")
  cat("----------------------------------\n")
  for (param in names(param_bounds)) {
    range <- param_bounds[[param]]
    cat(sprintf(" %-14s : %6.3f - %6.3f\n", param, range[1], range[2]))
  }
  cat("----------------------------------\n\n")
}

#------------------------------------------------------------------------------#
#    PARAMETER SET DISTRIBUTION PLOTTING                                       #
#------------------------------------------------------------------------------#

plot_parameter_distributions <- function(df, path) {
  # Convert data to long format for faceting
  df_long <- df %>% select(-seed) %>%
    pivot_longer(cols = everything(), names_to = "Parameter", values_to = "Value")
  
  # Convert Parameter column to factor
  df_long$Parameter <- factor(df_long$Parameter, levels = c(
    "mean_t_incub", "stdv_t_incub", "mean_nContact", "p_trans", "p_fatal"
  ))
  
  # Define proper expression labels
  parameter_labels <- as_labeller(c(
    mean_t_incub = "mu[incub]",
    stdv_t_incub = "sigma[incub]",
    mean_nContact = "italic(n)[contact]",
    p_trans = "italic(p)[trans]",
    p_fatal = "italic(p)[fatal]"
  ), label_parsed)
  
  # Define custom theme
  theme_pub <- theme_classic(base_size = 14) +  
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),  
      strip.text = element_text(face = "bold", size = 14),
      axis.title = element_text(size = 14, face = "bold"),
      axis.text = element_text(size = 12),
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
  
  # Save a plot of the parameter distributions
  plot_dist <- ggplot(df_long, aes(x = Value, fill = Parameter)) + 
    geom_histogram(binwidth = function(x) diff(range(x)) / 40, 
                   alpha = 0.8, color = "black") +
    facet_wrap(~Parameter, scales = "free_x", labeller = parameter_labels, ncol = 2) +
    scale_fill_viridis_d() +
    labs(
      title = "Distribution of Sampled Parameters",
      x = "Value",
      y = "Count"
    ) +
    theme_pub
  ggsave(path, plot = plot_dist, width = 8, height = 6)
  
  # Ensure plot was written successfully
  if (!file.exists(path)) {
    stop("Error: Parameter distribution plot was not created successfully.")
  }
}

#------------------------------------------------------------------------------#
#    ELAPSED TIME FORMATTING                                                   #
#------------------------------------------------------------------------------#

format_elapsed_time <- function(elapsed_time) {
  elapsed_secs <- as.numeric(elapsed_time)  # Convert difftime object to numeric (seconds)
  
  if (elapsed_time < 60) {
    return(sprintf("%.2f sec", elapsed_time))
  } else if (elapsed_time < 3600) {  # Less than 1 hour
    minutes <- floor(elapsed_time / 60)
    seconds <- round(elapsed_time %% 60, 2)
    return(sprintf("%d min %.2f sec", minutes, seconds))
  } else if (elapsed_time < 86400) {  # Less than 24 hours
    hours <- floor(elapsed_time / 3600)
    minutes <- round((elapsed_time %% 3600) / 60, 2)
    return(sprintf("%d hr %.2f min", hours, minutes))
  } else {  # More than 24 hours
    days <- floor(elapsed_time / 86400)
    hours <- round((elapsed_time %% 86400) / 3600, 2)
    return(sprintf("%d days %.2f hr", days, hours))
  }
}