# write.csv(as.data.frame(anova_table), file = "C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_pMatch_100_results.csv", row.names = FALSE)

############# One-way ANOVA #######################

library(tidyverse)
library(ggpubr)
library(rstatix)
library(datarium)
library(lme4)
library(readODS)
library(reshape2)
library(ggpubr)

# Convert NumPy arrays to data frames in R
# data = read_ods('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_predictive_accuracy.ods', sheet = 'data50_25percentile')
data = read_ods('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/GLM_weights.ods', sheet = '50')
data <- melt(data, variable.name = "group", value.name = "score")

anova_model = res.aov <- aov(score ~ group, data = data)
summary(res.aov)
tukey_results <- TukeyHSD(res.aov)

# Extract significant pairs and their p-values
significant_pairs <- as.data.frame(tukey_results$group)
significant_pairs <- significant_pairs[significant_pairs$`p adj` < 0.05, ]

# Create comparison list for geom_signif
comparison_list <- lapply(rownames(significant_pairs), function(x) unlist(strsplit(x, "-")))

# Plot with ggplot2 and manually add significant comparisons
p <- ggplot(data, aes(x = group, y = score)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Boxplot of Scores by Group with Significant Comparisons",
       x = "Group",
       y = "Score")

# Add significance annotations
p + geom_signif(comparisons = comparison_list, 
                map_signif_level = TRUE, 
                y_position = max(data$score) * 1.1)
