#------------------------------------------------------------------------------#
# Helper Functions                                                             #
#------------------------------------------------------------------------------#

safe_mean <- function(x, default = 0) {
  ifelse(is.na(mean(x, na.rm = TRUE)), default, mean(x, na.rm = TRUE))
}

safe_median <- function(x, default = 0) {
  ifelse(is.na(median(x, na.rm = TRUE)), default, median(x, na.rm = TRUE))
}

safe_var <- function(x, default = 0) {
  ifelse(is.na(var(x, na.rm = TRUE)), default, var(x, na.rm = TRUE))
}

#------------------------------------------------------------------------------#
# Compute Summary Statistics                                                   #
#------------------------------------------------------------------------------#

compute_summary_statistics <- function(SimulationSingle, nosoi_settings) {
  Data <- SimulationSingle
  All_Data_A <- getTableHosts(Data)
  Data_A <- All_Data_A[, c("hosts.ID", "inf.by")]

  # Ensure Data_A exists
  if (is.null(Data_A) || nrow(Data_A) == 0) {
    warning("No host data found in simulation.")
    return(data.table(seed = NA))  # Return a placeholder to prevent errors
  }

  # Basic Statistics
  ss_noninf <- sum(!Data_A$hosts.ID %in% Data_A$inf.by)  # Number of non-infecting hosts
  frequency_table <- table(Data_A$inf.by)
  result_table <- data.frame(hosts.ID = names(frequency_table), Frequency = as.numeric(frequency_table))

  ss_mean_secinf <- safe_mean(result_table$Frequency)
  ss_med_secinf <- safe_median(result_table$Frequency)
  ss_var_secinf <- safe_var(result_table$Frequency)

  # Pseudo H - Fraction of infecting hosts needed to infect 50% of the hosts
  result_table <- result_table[order(-result_table$Frequency), ]
  result_table$Cumulative <- cumsum(result_table$Frequency)
  ss_fractop50 <- which(result_table$Cumulative >= (0.5 * sum(result_table$Frequency)))[1] / length(result_table$Frequency)

  ss_hostspertime <- length(Data_A$hosts.ID) / Data$total.time  # Hosts per time

  # Infection Time Statistics
  inf_time <- All_Data_A$out.time - All_Data_A$inf.time
  inf_time <- inf_time[!is.na(inf_time)]

  ss_mean_inftime <- safe_mean(inf_time)
  ss_med_inftime <- safe_median(inf_time)
  ss_var_inftime <- safe_var(inf_time)

  ss_prop_infectors <- length(frequency_table) / length(All_Data_A$hosts.ID)  # Proportion of infecting hosts

  ss_active_final <- sum(All_Data_A$active)  # Number of active hosts at the end
  ss_hosts_total <- length(All_Data_A$hosts.ID)  # Total hosts over the simulation
  ss_frac_active_final <- ifelse(ss_hosts_total != 0, ss_active_final / ss_hosts_total, 0)  # Ratio of active hosts to total

  # Infection Timing Metrics
  result_td <- merge(All_Data_A, All_Data_A, by.x = "inf.by", by.y = "hosts.ID", suffixes = c("", "_infected_by"))
  result_td$inf_time_diff <- result_td$inf.time - result_td$inf.time_infected_by
  result_td <- result_td[, c("hosts.ID", "inf.by", "inf.time", "inf.time_infected_by", "inf_time_diff")]

  ss_mean_inflag <- safe_mean(result_td$inf_time_diff, Data$total.time + 1)
  ss_min_inflag <- ifelse(nrow(result_td) <= 1, Data$total.time + 1, min(aggregate(inf_time_diff ~ inf.by, result_td, min)$inf_time_diff))
  ss_med_inflag <- safe_median(result_td$inf_time_diff)
  ss_var_inflag <- safe_var(result_td$inf_time_diff)

  ss_frac_runtime <- Data$total.time / nosoi_settings$length  # Ratio of simulation run time to max time

  # Network Structure Analysis
  edges <- All_Data_A[, c("inf.by", "hosts.ID")]
  graph <- graph_from_data_frame(edges, directed = TRUE)

  ss_g_degree <- safe_mean(degree(graph))                   # Degree distribution
  ss_g_clustcoef <- transitivity(graph, type = "global")    # Clustering coefficient
  ss_g_density <- edge_density(graph)                       # Network density
  ss_g_diam <- diameter(graph)                              # Network diameter
  ss_g_meanego <- safe_mean(ego_size(graph))                # Mean neighborhood size
  ss_g_radius <- radius(graph)                              # Network radius
  ss_g_meanalpha <- safe_mean(alpha_centrality(graph))      # Network mean alpha centrality
  ss_g_effglob <- global_efficiency(graph)                  # Network global efficiency

  # Return all summary statistics in a df
  summary_stats <- data.frame(
    ss_noninf, ss_mean_secinf, ss_med_secinf, ss_var_secinf, ss_fractop50,
    ss_hostspertime, ss_mean_inftime, ss_med_inftime, ss_var_inftime,
    ss_prop_infectors, ss_active_final, ss_hosts_total, ss_frac_active_final,
    ss_mean_inflag, ss_min_inflag, ss_med_inflag, ss_var_inflag,
    ss_frac_runtime, ss_g_degree, ss_g_clustcoef, ss_g_density, ss_g_diam,
    ss_g_meanego, ss_g_radius, ss_g_meanalpha, ss_g_effglob
  )

  return(summary_stats)
}

