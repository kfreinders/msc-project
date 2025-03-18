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
  
  if (elapsed_secs < 60) {
    return(sprintf("%.2f sec", elapsed_secs))
  } else if (elapsed_secs < 3600) {  # Less than 1 hour
    minutes <- floor(elapsed_secs / 60)
    seconds <- round(elapsed_secs %% 60, 2)
    return(sprintf("%d min %.2f sec", minutes, seconds))
  } else if (elapsed_secs < 86400) {  # Less than 24 hours
    hours <- floor(elapsed_secs / 3600)
    minutes <- round((elapsed_secs %% 3600) / 60, 2)
    return(sprintf("%d hr %.2f min", hours, minutes))
  } else {  # More than 24 hours
    days <- floor(elapsed_secs / 86400)
    hours <- round((elapsed_secs %% 86400) / 3600, 2)
    return(sprintf("%d days %.2f hr", days, hours))
  }
}


#------------------------------------------------------------------------------#
#    COMPRESS NOSOI INFECTION TABLE                                            #
#------------------------------------------------------------------------------#

reconstruct_hosts_ID <- function(df) {
  # Ensure df is a data frame
  df <- as.data.frame(df)

  # Create a new hosts.ID column based on row number
  df$hosts.ID <- seq_len(nrow(df))

  # Convert inf.by back to original host IDs
  if ("inf.by" %in% names(df)) {
    valid_indices <- !is.na(df$inf.by) & df$inf.by > 0 & df$inf.by <= nrow(df)
    df$inf.by[valid_indices] <- df$hosts.ID[df$inf.by[valid_indices]]
  }

  # Reconstruct 'active' column: 1 if out.time is NA, otherwise 0
  if ("out.time" %in% names(df)) {
    df$active <- as.integer(is.na(df$out.time))
  }

  # Ensure the correct column order
  correct_order <- c("hosts.ID", "inf.by", "inf.time", "out.time", "active", "tIncub")
  existing_columns <- intersect(correct_order, names(df))  # Keep only existing columns
  df <- df[, existing_columns, drop = FALSE]  # Reorder and drop unnecessary columns

  return(df)
}

sort_replace_datatypes <- function(df) {
  # Sort by infection time
  df <- df[order(df$inf.time), ]

  # Convert hosts.ID: Remove "H-" and convert to integer
  df$hosts.ID <- as.integer(sub("^H-", "", as.character(df$hosts.ID)))

  # Convert inf.by: Remove "H-" and convert to integer, handling NA cases
  df$inf.by <- sub("^H-", "", as.character(df$inf.by))  
  df$inf.by <- suppressWarnings(as.integer(df$inf.by))
  df$inf.by[is.na(df$inf.by)] <- 0  # Ensure patient zero is 0

  # Convert out.time: Ensure proper NA replacement and integer conversion
  df$out.time[df$out.time == "nan"] <- NA
  df$out.time <- suppressWarnings(as.integer(df$out.time))

  return(df)
}

# ------------------------------------------------------------------------------
# INF.BY MAPPING EXPLANATION:
# The `inf.by` column initially stores the host ID of the infector. By
# replacing each entry with the row index of its corresponding infector, we
# transform it into a parent-pointer structure. This fully encodes the graph
# structure, allowing us to drop the `hosts.ID` column to reduce the file size.
# ------------------------------------------------------------------------------
  
save_inftable_compressed <- function(df, output_folder, seed) {
  # Sort the df and use more efficient data types
  df <- sort_replace_datatypes(df)

  formatted_seed <- sprintf("%010d", seed)

  # Debugging: Also write original table.hosts as CSV
  fwrite(df, file.path(output_folder, paste0("inftable_", formatted_seed, ".csv")))

  # Apply mapping
  df$inf.by <- match(df$inf.by, df$hosts.ID, nomatch = NA)  # Replace with row indices
  
  # Drop hosts.ID column after mapping, as we can reconstruct it
  df$hosts.ID <- NULL  

  # Hosts with no out.time (NA) are still active, therefore this column is
  # redundant
  df$active <- NULL  

  # Save as Parquet
  filename <- paste0("inftable_", formatted_seed, "_mapped.parquet")
  write_parquet(df, file.path(output_folder, filename))

  # Debugging: write as reconstructed CSV also
  rec <- reconstruct_hosts_ID(df)
  fwrite(rec, file.path(output_folder, paste0("inftable_", formatted_seed, "_rec.csv")))

  return(file.path(output_folder, filename))
}

