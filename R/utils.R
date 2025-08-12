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
  cat("--------------------------------------------------\n")

  # Find the longest parameter name for alignment
  max_name_length <- max(nchar(names(param_bounds)))

  for (param in names(param_bounds)) {
    range <- param_bounds[[param]]

    if (range[1] < range[2]) {
      # Regular parameter with a range
      cat(sprintf(" %-*s : %6.3f - %6.3f\n", max_name_length, param, range[1], range[2]))
    } else {
      # Fixed parameter
      cat(sprintf(" %-*s : %6.3f\n", max_name_length, param, range[1]))
    }
  }

  cat("--------------------------------------------------\n\n")
}

#------------------------------------------------------------------------------#
#    PARAMETER SET DISTRIBUTION PLOTTING                                       #
#------------------------------------------------------------------------------#

plot_parameter_distributions <- function(df, path) {
  # Remove 'seed' column and reshape the data to long format
  df_long <- df %>%
    select(-seed) %>%
    pivot_longer(cols = everything(), names_to = "Parameter", values_to = "Value")

  # Identify fixed parameters (those with a single unique value)
  fixed_params <- df_long %>%
    group_by(Parameter) %>%
    summarize(unique_values = n_distinct(Value), .groups = "drop") %>%
    filter(unique_values == 1) %>%
    pull(Parameter)

  # Filter out fixed parameters for plotting
  df_plot <- df_long %>%
    filter(!(Parameter %in% fixed_params))

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
    p_fatal = "italic(p)[fatal]",
    t_recovery = "t[recovery]"
  ), label_parsed)

  # Print fixed parameters that were not plotted, if any
  if (length(fixed_params) > 0) {
    cat("Not plotting fixed parameters: ")
    for (param in fixed_params) {
      cat(sprintf("%s ", param))
    }
    cat("\n")
  }

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

  # Plot only non-fixed parameters
  plot_dist <- ggplot(df_plot, aes(x = Value, fill = Parameter)) + 
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

  # Save the plot
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
#    POST-SIMULATION SUMMARY                                                   #
#------------------------------------------------------------------------------#

print_mclapply_stats <- function(mc_stats) {
  # Extract first element if mc_stats is a list
  stats <- mc_stats[[1]]

  # Format and print each row manually
  cat("Memory usage (mclapply stats):\n")
  for (i in seq_len(nrow(stats))) {
    row_name <- rownames(stats)[i]
    values <- format(stats[i, ], digits = 3, justify = "right")
    cat(sprintf("  %-7s %10s %6s %11s %6s %9s %6s\n",
                row_name,
                values["used"], "(Mb)",
                values["gc trigger"], "(Mb)",
                values["max used"], "(Mb)"))
  }
  cat("\n")
}

print_run_summary <- function(
  df_remaining, paramsets_file, output_folder, mc_stats, elapsed_time
) {
  print_section("SUMMARY")

  # Load all seeds from master
  df_all <- fread(paramsets_file)
  all_seeds <- unique(df_all$seed)
  n_total <- length(all_seeds)

  # Determine whether we resumed a run
  resumed <- nrow(df_remaining) < n_total
  seeds_remaining <- df_remaining$seed

  # Find all parquet output files and parse seeds
  parquet_files <- list.files(
    path = output_folder,
    pattern = "^inftable_\\d+_mapped\\.parquet$",
    full.names = TRUE
  )

  # Validate file existence and parse seed numbers
  existing_files <- parquet_files[file.exists(parquet_files)]
  seeds_in_files <- as.integer(gsub(".*inftable_(\\d+)_mapped\\.parquet$", "\\1", existing_files))
  seeds_in_files <- unique(seeds_in_files)
  n_completed <- length(seeds_in_files)

  # Compare seed sets
  seeds_missing <- setdiff(all_seeds, seeds_in_files)
  seeds_unexpected <- setdiff(seeds_in_files, all_seeds)
  has_duplicates <- any(duplicated(seeds_in_files))

  # Print general progress
  if (n_completed == 0) {
    cat("No valid results were generated\n\n")
    return()
  }

  # Print parallel processing stats
  print_mclapply_stats(mc_stats)

  # Print no. completed simulations if we've resumed a run
  if (resumed) {
    completed_this_session <- n_completed - (n_total - nrow(df_remaining))
    cat(sprintf(
      "Resumed run: %d simulations completed in this session\n",
      completed_this_session
    ))
  }

  # Print total no. completed simulations
  cat(sprintf(
    "Total completed simulations: %d / %d (%.2f%%)\n",
    n_completed, n_total, 100 * n_completed / n_total
  ))

  # Print current run and total runtime
  cat(sprintf("Runtime: %s\n", format_elapsed_time(elapsed_time)))

  # File integrity diagnostics
  if (length(seeds_missing) > 0) {
    cat(sprintf("%d expected output files are missing\n", length(seeds_missing)))
    cat("Missing seeds: ", paste(head(seeds_missing, 5), collapse = ", "))
    if (length(seeds_missing) > 5) cat(" ... (truncated)\n")
  }

  if (length(seeds_unexpected) > 0) {
    cat(sprintf(
      "%d unexpected seed files found. %s should be clean when starting a new run\n",
      length(seeds_unexpected), output_folder
    ))
    cat("Unexpected seeds:", paste(head(seeds_unexpected, 5), collapse = ", "))
    if (length(seeds_unexpected) > 5) cat(" ... (truncated)\n")
    cat("\n")
  }

  if (has_duplicates) {
    cat("Duplicate seed files found in output\n")
  }

  # Final note
  cat(sprintf("Full infection tables saved to: %s\n\n", output_folder))
}

#------------------------------------------------------------------------------#
#    COMPRESS NOSOI INFECTION TABLE                                            #
#------------------------------------------------------------------------------#

# Function to determine fate post-simulation
# Mapping:
#   0 -> still active
#   1 -> deceased
#   2 -> recovered

determine_fate <- function(df) {
  df$fate <- NA  # Initialize fate column

  # Compute individual recovery thresholds
  recovery_threshold <- df$inf.time + df$tIncub + df$tRecov + runif(nrow(df), min = 0, max = 1)

  # Recovered: survived beyond recovery threshold
  df$fate[df$out.time >= recovery_threshold] <- 2

  # Deceased: exited before recovery threshold
  df$fate[df$out.time < recovery_threshold & !is.na(df$out.time)] <- 1

  # Still active at end of simulation
  df$fate[is.na(df$out.time)] <- 0

  return(df)
}

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
  
save_inftable_compressed <- function(df, output_folder, seed, simtime) {
  # Sort the df and use more efficient data types
  df <- sort_replace_datatypes(df)

  formatted_seed <- sprintf("%010d", seed)

  # Debugging: Also write original table.hosts as CSV
  # fwrite(df, file.path(output_folder, paste0("inftable_", formatted_seed, ".csv")))

  # Apply mapping
  df$inf.by <- match(df$inf.by, df$hosts.ID, nomatch = NA)  # Replace with row indices
  
  # Drop hosts.ID column after mapping, as we can reconstruct it
  df$hosts.ID <- NULL  

  # Hosts with no out.time (NA) are still active, therefore this column is
  # redundant
  df$active <- NULL  

  # Convert metadata to schema
  table <- Table$create(df)
  metadata <- list(simtime = as.character(simtime))
  schema <- table$schema$WithMetadata(metadata)
  table <- table$cast(schema)

  # Save as Parquet with metadata
  filename <- paste0("inftable_", formatted_seed, "_mapped.parquet")
  write_parquet(table, file.path(output_folder, filename))

  # Debugging: write as reconstructed CSV also
  # rec <- reconstruct_hosts_ID(df)
  # fwrite(rec, file.path(output_folder, paste0("inftable_", formatted_seed, "_rec.csv")))

  return(file.path(output_folder, filename))
}

