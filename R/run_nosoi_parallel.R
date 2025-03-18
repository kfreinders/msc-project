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

create_worker <- function(db_name, output_folder) {
  function(params) {
    tryCatch({
      seed <- params$seed
      start_time <- Sys.time()  # Start timing
      
      # Run the simulation
      sim_result <- run_nosoi_simulation(params)
      hosts_table <- getTableHosts(sim_result)
      summary_statistics <- compute_summary_statistics(sim_result)
      
      if (!is.null(hosts_table) && nrow(hosts_table) > 0) {
        # Save the infection table to a Parquet file
        output_file <- save_inftable_compressed(hosts_table, output_folder, seed)
        
        end_time <- Sys.time()  # End timing
        elapsed_time <- round(difftime(end_time, start_time, units = "secs"), 2)
        
        # Store summary statistics in SQLite
        db <- dbConnect(RSQLite::SQLite(), db_name)
        write_summary_statistics(db, seed, summary_statistics) 
        dbDisconnect(db)  # Close connection after writing
        
        # Logging
        print_log(seed, nrow(hosts_table), elapsed_time)
        
        return(output_file)  # Return file names
      } else {
        message("WARNING: seed ", seed, " finished but no infections recorded.")
        return(NULL)
      }
    }, error = function(e) {
      message("ERROR: seed ", params$seed, " failed. Cause: ", e$message)
      return(NULL)
    })
  }
}

#------------------------------------------------------------------------------#
#    PARALLELIZATION                                                           #
#------------------------------------------------------------------------------#

run_nosoi_parallel <- function(input_file, db_name, output_folder, num_cores) {
  # Load parameter sets
  df <- fread(input_file)
  params_list <- split(df, seq(nrow(df)))
  
  # Create worker with access to db_name and output_folder
  worker <- create_worker(db_name, output_folder)
  
  # Run in parallel
  output_files <- mcmapply(worker, params_list,
                           mc.cores = num_cores, mc.preschedule = FALSE)
  
  return(output_files)
}
