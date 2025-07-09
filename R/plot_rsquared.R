library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)

# Read the CSV file
df <- read_csv("R/r2_scores.csv")

# Convert to long format
df_long <- df %>%
  pivot_longer(
    cols = -level,
    names_to = "parameter",
    values_to = "r2"
  )

# Create the plot
p = ggplot(df_long, aes(x = level, y = r2, color = parameter, shape = parameter)) +
  geom_line() +
  geom_point(size = 3) +
  scale_x_continuous(breaks = seq(0, 0.5, by = 0.05)) +
  labs(
    title = "R² Score per Parameter Across Scarcity Levels",
    x = "Scarcity Level",
    y = expression(R^2 ~ "Score"),
    color = "Parameter",
    shape = "Parameter"
  ) +
  theme_bw() +
  theme(
    text = element_text(size = 12),
    legend.position = "bottom"
  )

ggsave(p,
  filename = "rsquared.pdf",
  device = "pdf",
  height = 20, width = 30, units = "cm"
)
