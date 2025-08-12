defaults <- list(
  nosoi_settings = list(
    n_sim            = 4e5,    # Number of simulations to run
    length           = 100,    # Simulation length in days
    max_infected     = 50000,  # Maximum number of infected individuals
    init_individuals = 1       # Initial number of infected individuals
  ),
  param_bounds = list(
    mean_t_incub     = c(2, 21),
    stdv_t_incub     = c(1, 4),
    mean_nContact    = c(0.1, 5),
    p_trans          = c(0.01, 1),
    p_fatal          = c(0.01, 0.5),
    mean_t_recovery  = c(10, 30)
  ),
  paths = list(
    output_folder    = "data/nosoi"
  )
)
