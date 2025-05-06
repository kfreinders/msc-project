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

compute_summary_statistics <- function(sim_result, hosts_table, nosoi_settings) {
  # Basic Statistics
  ss_noninf <- sum(!hosts_table$hosts.ID %in% hosts_table$inf.by)  # Number of non-infecting hosts
  frequency_table <- table(hosts_table$inf.by)
  result_table <- data.frame(hosts.ID = names(frequency_table), Frequency = as.numeric(frequency_table))

  ss_mean_secinf <- safe_mean(result_table$Frequency)
  ss_med_secinf <- safe_median(result_table$Frequency)
  ss_var_secinf <- safe_var(result_table$Frequency)

  # Pseudo H - Fraction of infecting hosts needed to infect 50% of the hosts
  result_table <- result_table[order(-result_table$Frequency), ]
  result_table$Cumulative <- cumsum(result_table$Frequency)
  ss_fractop50 <- which(result_table$Cumulative >= (0.5 * sum(result_table$Frequency)))[1] / length(result_table$Frequency)

  ss_hostspertime <- length(hosts_table$hosts.ID) / sim_result$total.time  # Hosts per time

  # Infection Time Statistics
  inf_time <- hosts_table$out.time - hosts_table$inf.time
  inf_time <- inf_time[!is.na(inf_time)]

  ss_mean_inftime <- safe_mean(inf_time)
  ss_med_inftime <- safe_median(inf_time)
  ss_var_inftime <- safe_var(inf_time)

  ss_prop_infectors <- length(frequency_table) / length(hosts_table$hosts.ID)  # Proportion of infecting hosts

  ss_active_final <- sum(hosts_table$active)  # Number of active hosts at the end
  ss_hosts_total <- length(hosts_table$hosts.ID)  # Total hosts over the simulation
  ss_frac_active_final <- ifelse(ss_hosts_total != 0, ss_active_final / ss_hosts_total, 0)  # Ratio of active hosts to total

  # Infection Timing Metrics
  result_td <- merge(hosts_table, hosts_table, by.x = "inf.by", by.y = "hosts.ID", suffixes = c("", "_infected_by"))
  result_td$inf_time_diff <- result_td$inf.time - result_td$inf.time_infected_by
  result_td <- result_td[, c("hosts.ID", "inf.by", "inf.time", "inf.time_infected_by", "inf_time_diff")]

  ss_mean_inflag <- safe_mean(result_td$inf_time_diff, sim_result$total.time + 1)
  ss_min_inflag <- ifelse(nrow(result_td) <= 1, sim_result$total.time + 1, min(aggregate(inf_time_diff ~ inf.by, result_td, min)$inf_time_diff))
  ss_med_inflag <- safe_median(result_td$inf_time_diff)
  ss_var_inflag <- safe_var(result_td$inf_time_diff)

  ss_frac_runtime <- sim_result$total.time / nosoi_settings$length  # Ratio of simulation run time to max time

  # Network Structure Analysis
  edges <- hosts_table[, c("inf.by", "hosts.ID")]
  graph <- graph_from_data_frame(edges, directed = TRUE)

  ss_g_degree <- safe_mean(degree(graph))                   # Degree distribution
  ss_g_clustcoef <- transitivity(graph, type = "global")    # Clustering coefficient
  ss_g_density <- edge_density(graph)                       # Network density
  ss_g_diam <- diameter(graph)                              # Network diameter
  ss_g_meanego <- safe_mean(ego_size(graph))                # Mean neighborhood size
  ss_g_radius <- radius(graph)                              # Network radius
  ss_g_meanalpha <- safe_mean(alpha_centrality(graph))      # Network mean alpha centrality
  ss_g_effglob <- global_efficiency(graph)                  # Network global efficiency

  # Mortality-related statistics from host fate
  if ("fate" %in% colnames(hosts_table)) {
    # Mapping:
    #   0 -> still active
    #   1 -> deceased
    #   2 -> recovered
    death_mask <- hosts_table$fate == 1
    recovery_mask <- hosts_table$fate == 2

    time_to_death <- hosts_table$out.time[death_mask] - hosts_table$inf.time[death_mask]

    ss_deaths <- sum(death_mask)                            # Total no. deaths
    ss_mean_deaths <- safe_mean(death_mask)                 # Proportion of hosts that died
    ss_mean_ttd <- safe_mean(time_to_death)                 # Mean time to death
    ss_med_ttd <- safe_median(time_to_death)                # Median time to death
    ss_var_ttd <- safe_var(time_to_death)                   # Variance of time to death
    ss_death_recov_ratio <- ifelse(
      sum(recovery_mask) != 0,
      ss_deaths / sum(recovery_mask),                       # Death-to-recovery ratio
      NA
    )
  } else {
    ss_deaths <- ss_mean_deaths <- ss_mean_ttd <- ss_med_ttd <- ss_var_ttd <- ss_death_recov_ratio <- NA
  }

  # Return all summary statistics in a df
  summary_stats <- data.frame(
    ss_noninf, ss_mean_secinf, ss_med_secinf, ss_var_secinf, ss_fractop50,
    ss_hostspertime, ss_mean_inftime, ss_med_inftime, ss_var_inftime,
    ss_prop_infectors, ss_active_final, ss_hosts_total, ss_frac_active_final,
    ss_mean_inflag, ss_min_inflag, ss_med_inflag, ss_var_inflag,
    ss_frac_runtime, ss_g_degree, ss_g_clustcoef, ss_g_density, ss_g_diam,
    ss_g_meanego, ss_g_radius, ss_g_meanalpha, ss_g_effglob, ss_deaths,
    ss_mean_deaths, ss_mean_ttd, ss_med_ttd, ss_var_ttd, ss_death_recov_ratio
  )

  return(summary_stats)
}

