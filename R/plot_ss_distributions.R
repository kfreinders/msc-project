#------------------------------------------------------------------------------#
#    LIBRARIES                                                                 #
#------------------------------------------------------------------------------#

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
})

#------------------------------------------------------------------------------#
#    PROCESS AND LABEL DATA                                                    #
#------------------------------------------------------------------------------#

# Load the summary statistics CSV
df <- fread("data/nosoi/summary_stats_export.csv")

# Filter out simulations with only patient zero
if ("SS_11" %in% names(df)) {
  df <- df %>% filter(SS_11 != 1)
}

# Reshape to long format
df_long <- df %>%
  # select(-seed) %>%
  pivot_longer(cols = everything(), names_to = "Statistic", values_to = "Value")

# Define order and labels
ss_levels <- paste0("SS_", sprintf("%02d", 0:25))

ss_labels <- c(
  SS_00 = "00. Non-infecting hosts",
  SS_01 = "01. Mean sec. infections",
  SS_02 = "02. Median sec. infections",
  SS_03 = "03. Var. sec. infections",
  SS_04 = "04. Fraction causing 50%",
  SS_05 = "05. Hosts per time",
  SS_06 = "06. Mean infection time",
  SS_07 = "07. Median infection time",
  SS_08 = "08. Var. infection time",
  SS_09 = "09. Proportion infecting",
  SS_10 = "10. Active hosts at end",
  SS_11 = "11. Total hosts",
  SS_12 = "12. Active / total ratio",
  SS_13 = "13. Mean transmission delay",
  SS_14 = "14. Min delay per infector",
  SS_15 = "15. Median delay",
  SS_16 = "16. Variance in delay",
  SS_17 = "17. Time ratio",
  SS_18 = "18. Mean degree",
  SS_19 = "19. Clustering",
  SS_20 = "20. Edge density",
  SS_21 = "21. Diameter",
  SS_22 = "22. Mean ego size",
  SS_23 = "23. Radius",
  SS_24 = "24. Alpha centrality",
  SS_25 = "25. Global efficiency"
)

# Set ordered factor with labels
df_long$Statistic <- factor(
  df_long$Statistic, levels = ss_levels, labels = ss_labels
)

#------------------------------------------------------------------------------#
#    GENERATE PLOTS                                                            #
#------------------------------------------------------------------------------#

# Save multi-page PDF
pdf("summary_statistics_distributions.pdf", width = 12, height = 10)

plots <- list()

# Loop over statistics
for (i in seq_along(levels(df_long$Statistic))) {
  stat <- levels(df_long$Statistic)[i]
  data_stat <- df_long %>% filter(Statistic == stat)

  # Freedman-Diaconis calculation
  IQR_value <- IQR(data_stat$Value, na.rm = TRUE)
  n <- sum(!is.na(data_stat$Value))
  
  binwidth_fd <- ifelse(
    IQR_value == 0,
    (max(data_stat$Value, na.rm = TRUE) - min(data_stat$Value, na.rm = TRUE)) / 30,
    2 * IQR_value / n^(1/3)
  )
  
  # Spread
  spread <- max(data_stat$Value, na.rm = TRUE) - min(data_stat$Value, na.rm = TRUE)
  
  # Default to FD binning
  binwidth <- binwidth_fd
  
  # Only if spread is very wide, check integer closeness
  if (spread > 20) {
    dist_to_int <- abs(data_stat$Value - round(data_stat$Value))
    dist_to_half <- abs(data_stat$Value - (floor(data_stat$Value) + 0.5))
  
    prop_close_to_int <- mean(dist_to_int < 0.1, na.rm = TRUE)
    prop_close_to_half <- mean(dist_to_half < 0.1, na.rm = TRUE)
  
    if (prop_close_to_half > 0.8) {
      binwidth <- 0.5
    } else if (prop_close_to_int > 0.8) {
      binwidth <- 1
    }
  }

  range_vals <- range(data_stat$Value, na.rm = TRUE)
  data_range <- diff(range_vals)
  n_bins <- ceiling(data_range / binwidth)
  
  # Safe histogram computation
  max_count <- tryCatch({
    if (n_bins <= 1) {
      1  # Default fallback if bins are invalid
    } else {
      hist_data <- hist(data_stat$Value, plot = FALSE, breaks = n_bins)
      max(hist_data$counts, na.rm = TRUE)
    }
  }, error = function(e) {
    1  # If hist() fails, fallback max_count
  })

  # Calculate an adaptive upper limit
  y_upper <- if (max_count <= 100) {
    max_count * 1.5  # Gentle expansion for low counts
  } else if (max_count <= 1000) {
    max_count * 1.2  # Moderate expansion
  } else {
    max_count * 1.05  # Small expansion for very large counts
  }

  p <- ggplot(data_stat, aes(x = Value)) +
    geom_histogram(binwidth = binwidth, fill = "steelblue", color = "black", boundary = 0) +
    scale_x_continuous(limits = range(data_stat$Value, na.rm = TRUE)) +
    scale_y_continuous(limits = c(0, min(max_count * 1.1, 8000))) +
    labs(title = stat, x = NULL, y = NULL) +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5),
    )
  
  plots[[stat]] <- p
}

#------------------------------------------------------------------------------#
#    ARRANGE PLOTS WITH PATCHWORK                                              #
#------------------------------------------------------------------------------#

plots_per_page <- 9
n_pages <- ceiling(length(plots) / plots_per_page)

for (page in 1:n_pages) {
  idx_start <- (page - 1) * plots_per_page + 1
  idx_end <- min(page * plots_per_page, length(plots))
  plots_subset <- plots[idx_start:idx_end]

  combined_plot <- patchwork::wrap_plots(plots_subset, ncol = 3, nrow = 3, byrow = TRUE) +
    plot_annotation(
      caption = NULL,
      theme = theme(
        plot.caption = element_blank()
      )
    )

  final_plot <- combined_plot &
    theme(
      plot.margin = margin(5, 5, 5, 5),
      axis.title = element_blank()
    )

  print(final_plot)
}

dev.off()

