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
  Data_A <- getTableHosts(Data)
  All_Data_A <- Data_A
  Data_A <- subset(Data_A, select = c(hosts.ID, inf.by))
  
  # Ensure Data_A exists
  if (is.null(Data_A) || nrow(Data_A) == 0) {
    warning("No host data found in simulation.")
    return(data.table(seed = NA))  # Return a placeholder to prevent errors
  }
  
  # Basic Statistics
  SS_00 <- sum(!Data_A$hosts.ID %in% Data_A$inf.by)  # Number of non-infecting hosts
  frequency_table <- table(Data_A$inf.by)
  result_table <- data.frame(hosts.ID = names(frequency_table), Frequency = as.numeric(frequency_table))
  
  SS_01 <- safe_mean(result_table$Frequency)
  SS_02 <- safe_median(result_table$Frequency)
  SS_03 <- safe_var(result_table$Frequency)
  
  # Pseudo H - Fraction of infecting hosts needed to infect 50% of the hosts
  result_table <- result_table[order(-result_table$Frequency), ]
  result_table$Cumulative <- cumsum(result_table$Frequency)
  SS_04 <- which(result_table$Cumulative >= (0.5 * sum(result_table$Frequency)))[1] / length(result_table$Frequency)
  
  SS_05 <- length(Data_A$hosts.ID) / Data$total.time  # Hosts per time
  
  # Infection Time Statistics
  inf_time <- All_Data_A$out.time - All_Data_A$inf.time
  inf_time <- inf_time[!is.na(inf_time)]
  
  SS_06 <- safe_mean(inf_time)
  SS_07 <- safe_median(inf_time)
  SS_08 <- safe_var(inf_time)
  
  SS_09 <- length(frequency_table) / length(All_Data_A$hosts.ID)  # Proportion of infecting hosts
  
  SS_10 <- sum(All_Data_A$active)  # Number of active hosts at the end
  SS_11 <- length(All_Data_A$hosts.ID)  # Total hosts over the simulation
  SS_12 <- ifelse(SS_11 != 0, SS_10 / SS_11, 0)  # Ratio of active hosts to total
  
  # Infection Timing Metrics
  result_td <- merge(All_Data_A, All_Data_A, by.x = "inf.by", by.y = "hosts.ID", suffixes = c("", "_infected_by"))
  result_td$inf_time_diff <- result_td$inf.time - result_td$inf.time_infected_by
  result_td <- result_td[, c("hosts.ID", "inf.by", "inf.time", "inf.time_infected_by", "inf_time_diff")]
  
  SS_13 <- safe_mean(result_td$inf_time_diff, Data$total.time + 1)
  SS_14 <- ifelse(nrow(result_td) <= 1, Data$total.time + 1, min(aggregate(inf_time_diff ~ inf.by, result_td, min)$inf_time_diff))
  SS_15 <- safe_median(result_td$inf_time_diff)
  SS_16 <- safe_var(result_td$inf_time_diff)

  SS_17 <- Data$total.time / nosoi_settings$length  # Ratio of simulation run time to max time

  # Network Structure Analysis
  edges <- All_Data_A[, c("inf.by", "hosts.ID")]
  graph <- graph_from_data_frame(edges, directed = TRUE)

  SS_18 <- safe_mean(degree(graph))  # Degree distribution
  SS_19 <- safe_mean(transitivity(graph, type = "global"))  # Clustering coefficient
  SS_20 <- safe_mean(edge_density(graph))  # Network density
  SS_21 <- safe_mean(diameter(graph))  # Network diameter
  SS_22 <- safe_mean(ego_size(graph))  # Mean neighborhood size
  SS_23 <- safe_mean(radius(graph))  # Network radius
  SS_24 <- safe_mean(alpha_centrality(graph))  # Network mean alpha centrality
  SS_25 <- safe_mean(global_efficiency(graph))  # Network global efficiency

  # Return all summary statistics in a df
  summary_stats <- data.frame(
    SS_00, SS_01, SS_02, SS_03, SS_04, SS_05, SS_06, SS_07, SS_08, SS_09, SS_10,
    SS_11, SS_12, SS_13, SS_14, SS_15, SS_16, SS_17, SS_18, SS_19, SS_20, SS_21,
    SS_22, SS_23, SS_24, SS_25
  )
  
  return(summary_stats)
}

