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
        # Save the infection table to a csv
        output_file <- file.path(output_folder, paste0("nosoi_inftable_seed_", seed, ".csv"))
        fwrite(hosts_table, output_file, append = TRUE, col.names = TRUE)
        
        end_time <- Sys.time()  # End timing
        elapsed_time <- round(difftime(end_time, start_time, units = "secs"), 2)
        
        # Store summary statistics in SQLite
        db <- dbConnect(RSQLite::SQLite(), db_name)
        write_summary_statistics(db, seed, summary_statistics) 
        dbDisconnect(db)  # Close connection after writing
        
        # Logging
        cat(sprintf("FINISHED: seed %d | Hosts Infected: %d | Duration: %.02f sec\n", seed, nrow(hosts_table), elapsed_time))
        
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
