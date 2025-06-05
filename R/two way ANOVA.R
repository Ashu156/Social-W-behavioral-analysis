# Two-way ANOVA

# Load the necessary R packages
# install.packages(c("tidyverse", "lme4"))  # Install if not already installed
library(tidyverse)
library(ggpubr)
library(rstatix)
library(datarium)
library(lme4)
library(readODS)
library(ggsignif)
library(car)
library(afex)

# Convert NumPy arrays to data frames in R
file_path = 'C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_transitions_per_minute.ods'
WT = read_ods(file_path, sheet = 'WT50')
# FX = read_ods(file_path, sheet = 'FX50')
MixedWT = read_ods(file_path, sheet = 'Mixed100_WT')
MixedFX = read_ods(file_path, sheet = 'Mixed100_FX')

clip_length = 21

# Transpose the data frame
if (ncol(WT) > nrow(WT)) {
  WT = data.frame(t(as.matrix(WT)))
  # FX = data.frame(t(as.matrix(FX)))
  MixedWT = data.frame(t(as.matrix(MixedWT)))
  MixedFX = data.frame(t(as.matrix(MixedFX)))
}




# MixedWT = na.omit(MixedWT)
# MixedFX = na.omit(MixedFX)



WT <- lapply(WT, function(col) head(col[!is.na(col)], clip_length))  # Extract first 36 non-NA values per column
WT <- as.data.frame(WT)  # Convert to dataframe

# FX <- lapply(FX, function(col) head(col[!is.na(col)], clip_length))  # Extract first 36 non-NA values per column
# FX <- as.data.frame(FX)  # Convert to dataframe

MixedWT <- lapply(MixedWT, function(col) head(col[!is.na(col)], clip_length))  # Extract first 36 non-NA values per column
MixedWT <- as.data.frame(MixedWT)  # Convert to dataframe

MixedFX <- lapply(MixedFX, function(col) head(col[!is.na(col)], clip_length))  # Extract first 36 non-NA values per column
MixedFX <- as.data.frame(MixedFX)  # Convert to dataframe



WT_rows = nrow(WT)*ncol(WT)
# FX_rows = nrow(FX)*ncol(FX)
MixedWT_rows = nrow(MixedWT)*ncol(MixedWT)
MixedFX_rows = nrow(MixedFX)*ncol(MixedFX)

dataWT <- data.frame(score = character(WT_rows), stringsAsFactors = FALSE)
dataWT$id = rep(paste0("WT", 1:ncol(WT)), each = nrow(WT))
dataWT$genotype = rep(paste0("WT"), each = nrow(WT))
dataWT$time <- rep(paste0(1:nrow(WT)), length.out = nrow(dataWT))
dataWT$score <- unlist(WT, use.names = FALSE)




# dataFX <- data.frame(score = character(FX_rows), stringsAsFactors = FALSE)
# dataFX$id = rep(paste0("FX", 1:ncol(FX)), each = nrow(FX))
# dataFX$genotype = rep(paste0("FX"), each = nrow(FX))
# dataFX$time <- rep(paste0(1:nrow(FX)), length.out = nrow(dataFX))
# dataFX$score <- unlist(FX, use.names = FALSE)


dataMixedWT <- data.frame(score = character(MixedWT_rows), stringsAsFactors = FALSE)
dataMixedWT$id = rep(paste0("MixedWT", 1:ncol(MixedWT)), each = nrow(MixedWT))
dataMixedWT$genotype = rep(paste0("MixedWT"), each = nrow(MixedWT))
dataMixedWT$time <- rep(paste0(1:nrow(MixedWT)), length.out = nrow(dataMixedWT))
dataMixedWT$score <- unlist(MixedWT, use.names = FALSE)

dataMixedFX <- data.frame(score = character(MixedFX_rows), stringsAsFactors = FALSE)
dataMixedFX$id = rep(paste0("MixedFX", 1:ncol(MixedFX)), each = nrow(MixedFX))
dataMixedFX$genotype = rep(paste0("MixedFX"), each = nrow(MixedFX))
dataMixedFX$time <- rep(paste0(1:nrow(MixedFX)), length.out = nrow(dataMixedFX))
dataMixedFX$score <- unlist(MixedFX, use.names = FALSE)


data = rbind(dataWT, dataMixedWT, dataMixedFX) # dataFX,

# data <- na.omit(data)

result = lm(formula = score ~ genotype + time + time:genotype, data = data)
# result = lm(formula = score ~ time + genotype + time:genotype, data = data)
# plot(result, which = 2, add.smooth = FALSE)
# plot(result, which = 3, add.smooth = FALSE)
tdat <- data.frame(predicted=predict(result), residual = residuals(result))
ggplot(tdat,aes(sample=residual)) + stat_qq() + stat_qq_line()
ggplot(tdat,aes(x=residual)) + geom_histogram(bins=20, color="black")

m1 = anova(result)
m1

m2 = car::Anova(result, type = 2)
m2

m3 = car::Anova(result, type = 3)
m3

m4 = aov_ez(
  id = "id",          # Repeated measure: Subject pair
  dv = "score",   # Dependent variable
  within = "time", # Repeated factor: Condition (50 vs opq)
  between = "genotype", # Between factor: Genotype (WT vs FX)
  data = data
)

# Print ANOVA results
print(m4$anova_table)

# Visualization
bxp <- ggboxplot(
  data, x = "time", y = "score",
  color = "genotype", palette = "aaas"
)
bxp


# Extract ANOVA table
anova_table <- anova(result)

# Save ANOVA table to a CSV file
