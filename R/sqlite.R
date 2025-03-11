#------------------------------------------------------------------------------#
#    DATABASE SETUP                                                            #
#------------------------------------------------------------------------------#

initialize_db <- function(db_name) {
  db <- dbConnect(RSQLite::SQLite(), db_name)
  # Create a table if it doesn't exist. Use invisible to suppress printing output 
  invisible(dbExecute(db,"
    CREATE TABLE IF NOT EXISTS summary_stats (
      seed INTEGER PRIMARY KEY,
      SS_00 REAL,
      SS_01 REAL,
      SS_02 REAL,
      SS_03 REAL,
      SS_04 REAL,
      SS_05 REAL,
      SS_06 REAL,
      SS_07 REAL,
      SS_08 REAL,
      SS_09 REAL,
      SS_10 REAL,
      SS_11 REAL,
      SS_12 REAL,
      SS_13 REAL,
      SS_14 REAL,
      SS_15 REAL,
      SS_16 REAL,
      SS_17 REAL,
      SS_18 REAL,
      SS_19 REAL,
      SS_20 REAL,
      SS_21 REAL,
      SS_22 REAL,
      SS_23 REAL,
      SS_24 REAL,
      SS_25 REAL
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
