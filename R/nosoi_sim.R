#------------------------------------------------------------------------------#
#    NOSOI CORE FUNCTION FACTORIES                                             #
#------------------------------------------------------------------------------#

# Function factory: incubation time 
create_tIncubFunc <- function(mean_tIncub, stdv_tIncub) {
  function(t) {
    return(rtruncnorm(1, a = 2, mean = mean_tIncub, sd = stdv_tIncub))
  }
}

# Function factory: transmission probability
create_pTransFunc <- function(p_trans) {
  function(t, tIncub) {
    if (t <= tIncub) {
      return(0)
    } else {
      return(p_trans)
    }
  }
}

# Function factory: exit probability (death / recovery)
create_pExitFunc <- function(p_fatal) {
  function(t, tIncub) {
    t_recovery <- 20  # Days before recovery
    if (t <= tIncub) {
      return(0)
    } else if ((t > tIncub) & (t < (tIncub + t_recovery))) {
      return(p_fatal)
    } else if (t > (tIncub + t_recovery)) {
      return(1)
    }
  }
}

# Function factory: number of contacts
create_nContactFunc <- function(mean_nContact) {
  function(t) {
    return(rpois(1, lambda = mean_nContact))  # Poisson-distributed contacts  
  }
}

#------------------------------------------------------------------------------#
#    SIMULATION LOGIC                                                          #
#------------------------------------------------------------------------------#

run_nosoi_simulation <- function(params) {
  # Convert parameters to numeric to avoid issues
  mean_tIncub   <- as.numeric(params$mean_t_incub)
  stdv_tIncub   <- as.numeric(params$stdv_t_incub)
  mean_nContact <- as.numeric(params$mean_nContact)
  p_trans       <- as.numeric(params$p_trans)
  p_fatal       <- as.numeric(params$p_fatal)
  seed          <- as.numeric(params$seed)
  
  # Create core functions
  tIncubFunc <- create_tIncubFunc(mean_tIncub, stdv_tIncub)
  pTransFunc <- create_pTransFunc(p_trans)
  nContactFunc <- create_nContactFunc(mean_nContact)
  pExitFunc <- create_pExitFunc(p_fatal)
  
  # Parameter lists
  param_pTrans <- list(tIncub = tIncubFunc)
  param_pExit <- list(tIncub = tIncubFunc)
  
  # Set seed for reproducing results
  set.seed(seed)
  
  # Start the simulation
  suppressMessages(
    simulation <-  nosoiSim(
      type = "single",        # Single-host system
      length = 365,           # Simulation length in days
      popStructure = "none",  # No population structure
      max.infected = 10000,   # Maximum number of infected individuals
      init.individuals = 1,   # Initial number of infected individuals
      
      # pTrans
      pTrans = pTransFunc,
      param.pTrans = param_pTrans,
      
      # nContact
      nContact = nContactFunc,
      param.nContact = NA,
      
      # pExit
      pExit = pExitFunc,
      param.pExit = param_pExit,
      
      # Closing parameters
      prefix.host = "H", 
      print.progress = FALSE
    )
  )
  
  return(simulation)
}

