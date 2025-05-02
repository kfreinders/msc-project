#------------------------------------------------------------------------------#
#    NOSOI CORE FUNCTION FACTORIES                                             #
#------------------------------------------------------------------------------#

# Function factory: incubation time 
create_tIncubFunc <- function(mean_tIncub, stdv_tIncub) {
  function(t) {
    return(rtruncnorm(1, a = 2, mean = mean_tIncub, sd = stdv_tIncub))
  }
}

# Funcion factory: recovery time
create_tRecoveryFunc <- function(mean_tRecovery) {
  function(t) {
    return(rtruncnorm(1, a = 2, mean = mean_tRecovery, sd = 1))
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
  function(t, tIncub, tRecov) {
    if (t <= tIncub) {
      return(0)
    } else if ((t > tIncub) & (t < (tIncub + tRecov))) {
      return(p_fatal)
    } else if (t > (tIncub + tRecov)) {
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

run_nosoi_simulation <- function(param_bounds, nosoi_settings) {
  # Convert parameters to numeric to avoid issues
  mean_tIncub       <- as.numeric(param_bounds$mean_t_incub)
  stdv_tIncub       <- as.numeric(param_bounds$stdv_t_incub)
  mean_nContact     <- as.numeric(param_bounds$mean_nContact)
  p_trans           <- as.numeric(param_bounds$p_trans)
  p_fatal           <- as.numeric(param_bounds$p_fatal)
  mean_tRecovery    <- as.numeric(param_bounds$mean_t_recovery)
  seed              <- as.numeric(param_bounds$seed)

  # Do the same for the nosoi simulation settings
  length            <- as.numeric(nosoi_settings$length)
  max_infected      <- as.numeric(nosoi_settings$max_infected)
  init_individuals  <- as.numeric(nosoi_settings$init_individuals)
  
  # Create core functions
  tIncubFunc        <- create_tIncubFunc(mean_tIncub, stdv_tIncub)
  tRecovFunc        <- create_tRecoveryFunc(mean_tRecovery)
  pTransFunc        <- create_pTransFunc(p_trans)
  nContactFunc      <- create_nContactFunc(mean_nContact)
  pExitFunc         <- create_pExitFunc(p_fatal)
  
  # Parameter lists
  param_pTrans      <- list(tIncub = tIncubFunc)
  param_pExit       <- list(tIncub = tIncubFunc, tRecov = tRecovFunc)
  
  # Set seed for reproducing results
  set.seed(seed)
  
  # Start the simulation
  suppressMessages(
    simulation <-  nosoiSim(
      type = "single",                      # Single-host system
      length = length,                      # Simulation length in days
      popStructure = "none",                # No population structure
      max.infected = max_infected,          # Maximum number of infected individuals
      init.individuals = init_individuals,  # Initial number of infected individuals
      
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
      print.progress = FALSE
    )
  )
  
  return(simulation)
}

