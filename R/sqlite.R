#------------------------------------------------------------------------------#
#    DATABASE SETUP                                                            #
#------------------------------------------------------------------------------#

initialize_db <- function(db_name) {
  db <- dbConnect(RSQLite::SQLite(), db_name)
  # Create a table if it doesn't exist. Use invisible to suppress printing output 
  invisible(dbExecute(db,"
    CREATE TABLE IF NOT EXISTS summary_stats (
      seed INTEGER PRIMARY KEY,
      ss_noninf REAL,
      ss_mean_secinf REAL,
      ss_med_secinf REAL,
      ss_var_secinf REAL,
      ss_fractop50 REAL,
      ss_hostspertime REAL,
      ss_mean_inftime REAL,
      ss_med_inftime REAL,
      ss_var_inftime REAL,
      ss_prop_infectors REAL,
      ss_active_final REAL,
      ss_hosts_total REAL,
      ss_frac_active_final REAL,
      ss_mean_inflag REAL,
      ss_min_inflag REAL,
      ss_med_inflag REAL,
      ss_var_inflag REAL,
      ss_frac_runtime REAL,
      ss_g_degree REAL,
      ss_g_clustcoef REAL,
      ss_g_density REAL,
      ss_g_diam REAL,
      ss_g_meanego REAL,
      ss_g_radius REAL,
      ss_g_meanalpha REAL,
      ss_g_effglob REAL,
      ss_deaths REAL,
      ss_mean_deaths REAL,
      ss_mean_ttd REAL,
      ss_med_ttd REAL,
      ss_var_ttd REAL,
      ss_death_recov_ratio REAL
    )
  "))
  dbDisconnect(db) 
}

#------------------------------------------------------------------------------#
#    SAFELY WRITING TO SQLITE DB                                               #
#------------------------------------------------------------------------------#

# Function to write summary statistics
write_summary_statistics <- function(db, seed, summary_stats) {
  attempt <- 1 
  max_attempts <- 5
    
  # Retry writing to db until successful or until we exceed max_attempts
  while (attempt <= max_attempts) {
    tryCatch({
    # Improve concurrent writing
    dbExecute(db, "PRAGMA journal_mode=WAL;")
    
    # Construct SQL query
    query <- paste0(
      "INSERT OR REPLACE INTO summary_stats (seed, ", 
      paste(names(summary_stats), collapse = ", "), 
      ") VALUES (? , ", 
      paste(rep("?", length(summary_stats)), collapse = ", "), 
      ")"
    )
    
    # Execute the query using prepared statements
    dbExecute(db, query, params = c(seed, as.numeric(summary_stats[1, ])))
    
    # Success: Exit loop
    return(TRUE)
    }, error = function(e) {
      if (grepl("database is locked", e$message)) {
        # FIXME: fixed time and make backoff actually exponential
        Sys.sleep(runif(1, 0.1, 0.5) * attempt)  # Exponential backoff
        attempt <- attempt + 1
      } else {
        message("ERROR: Failed to write summary stats for seed ", seed, " | Cause: ", e$message)
        return(FALSE)
      }
    })
  }
  message("ERROR: Giving up on seed ", seed, " after ", max_attempts, " attempts (database locked).")
  return(FALSE)
}

#------------------------------------------------------------------------------#
#    EXPORTING DB TO CSV                                                       #
#------------------------------------------------------------------------------#

export_db_to_csv <- function(db_name, path) {
  db <- dbConnect(RSQLite::SQLite(), db_name)
  df <- dbReadTable(db, "summary_stats") 
  fwrite(df, path) 
  dbDisconnect(db) 
}

