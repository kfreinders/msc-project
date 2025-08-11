# Load libraries
library(ggplot2)
library(dplyr)

# ABC data
abc_df <- data.frame(
  n = c(50, 100, 150, 200, 250, 300, 350, 400, 450, 500),
  avg_time = c(1.53333449363709, 2.33834900856018, 4.38914813995361,
               5.62229118347168, 5.40343616008759, 7.23700640201569,
               7.43134422302246, 8.33237960338593, 11.8551999092102,
               11.998906826973),
  std_time = c(0.066926003386527, 0.376475644775661, 0.284781540227084,
               0.552257729467502, 1.24308483485766, 1.26867436280923,
               1.63365275998704, 1.66393855046442, 0.830055193003004,
               2.13681356448545),
  method = "ABC"
)

# DNN data
dnn_df <- data.frame(
  n = c(50, 100, 150, 200, 250, 300, 350, 400, 450, 500),
  avg_time = c(0.004420814000013706, 0.006530833799979518,
               0.00665922289995251, 0.008907359800014092,
               0.00997673440001563, 0.009681118200023774,
               0.01345133810000334, 0.014318010200031494,
               0.01605967020004755, 0.01742212980007025),
  std_time = c(0.008312170042578301, 0.008900351235030567,
               0.009105318352805413, 0.008915145578824835,
               0.008690949727250262, 0.008923624497200428,
               0.008612799258026452, 0.008386512607143687,
               0.009139547916049926, 0.008774806683781828),
  method = "DNN"
)

# Combine data
combined_df <- rbind(abc_df, dnn_df)

# Plot
ggplot(combined_df, aes(x = n, y = avg_time, fill = method, color = method)) +
  geom_ribbon(aes(ymin = avg_time - std_time, ymax = avg_time + std_time), alpha = 0.2, color = NA) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", linetype = "dashed", se = FALSE, linewidth = 0.8) +
  facet_wrap(~ method, scales = "free_y") +
  scale_x_continuous(breaks = seq(50, 500, 50), limits = c(50, 500), expand = c(0, 0)) +
  scale_color_manual(values = c("ABC" = "#2A9D8F", "DNN" = "#440154")) +
  scale_fill_manual(values = c("ABC" = "#2A9D8F", "DNN" = "#440154")) +
  labs(
    x = "Number of predictions",
    y = "Average compute time (s)"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")

