#------------------------------------------------------------------------------#
#    LOGGING MESSAGES                                                          #
#------------------------------------------------------------------------------#

print_log<- function(seed, hosts_infected, elapsed_time) {
  # Determine formatting for Hosts Infected column
  if (hosts_infected > 99999) {
    host_str <- sprintf("%10.2e", hosts_infected)  # Scientific notation for large numbers
  } else {
    host_str <- sprintf("%10d", hosts_infected)   # Regular integer formatting
  }
  
  # Print aligned output
  cat(sprintf("FINISHED: seed %010d | Hosts Infected: %s | Duration: %6.2f sec\n", 
              seed, host_str, elapsed_time))
}

#------------------------------------------------------------------------------#
#    WORKER FACTORY                                                            #
#------------------------------------------------------------------------------#

create_worker <- function(db_name, output_folder, nosoi_settings) {
  function(params) {
    tryCatch({
      seed <- params$seed
      start_time <- Sys.time()  # Start timing
      
      # Run the simulation
      sim_result <- run_nosoi_simulation(params, nosoi_settings)
      hosts_table <- getTableHosts(sim_result)
      simtime = sim_result$total.time
      
      if (!is.null(hosts_table) && nrow(hosts_table) > 0) {
        # Assign host fates
        hosts_table <- determine_fate(hosts_table)

        # Compute summary statistics
        summary_statistics <- compute_summary_statistics(
          hosts_table, nosoi_settings, simtime
        )

        # Save the infection table to a Parquet file
        save_inftable_compressed(hosts_table, output_folder, seed, simtime)
        
        end_time <- Sys.time()  # End timing
        elapsed_time <- round(difftime(end_time, start_time, units = "secs"), 2)
        
        # Store summary statistics in SQLite
        db <- dbConnect(RSQLite::SQLite(), db_name)
        write_summary_statistics(db, seed, summary_statistics) 
        dbDisconnect(db)  # Close connection after writing
        
        # Logging
        print_log(seed, nrow(hosts_table), elapsed_time)

        # Explicitly clear memory
        rm(sim_result, hosts_table)
        gc()
        
      } else {
        message("WARNING: seed ", seed, " finished but no infections recorded.")
      }
    }, error = function(e) {
      message("ERROR: seed ", params$seed, " failed. Cause: ", e$message)
    })
  }
}

#------------------------------------------------------------------------------#
#    PARALLELIZATION                                                           #
#------------------------------------------------------------------------------#

run_nosoi_parallel <- function(
  param_df, db_name, output_folder, num_cores, nosoi_settings
) {
  # Create worker
  params_list <- split(param_df, seq(nrow(param_df)))
  worker <- create_worker(db_name, output_folder, nosoi_settings)
  
  # Run simulations in parallel.
  mc_stats <- mclapply(
    params_list, worker, mc.cores = num_cores, mc.preschedule = FALSE
  )

  return(mc_stats)
}

