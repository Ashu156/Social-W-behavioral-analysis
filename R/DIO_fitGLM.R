library(readODS)
library(glmnet)
library(clipr)

data <- read.csv("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/CohortAS3/cohortAS3_rat_1_50.csv")
all_perf <- read_ods('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_perf_pMatch.ods', sheet = 'WT50')
ratData <- all_perf$AS3_WT1
sessions <- which(!is.na(ratData))
perf <- ratData[sessions]

# Define the percentile (e.g., 0.1 for the 10th percentile)
nth_percentile <- 0.25

# Calculate the percentile thresholds
lower_threshold <- quantile(perf, nth_percentile)
upper_threshold <- quantile(perf, 1 - nth_percentile)

# Find the indices for the bottom nth percentile
bottom_indices <- sessions[perf <= lower_threshold]
bottom_values <- ratData[bottom_indices]

# Find the indices for the top nth percentile
top_indices <- sessions[perf >= upper_threshold]
top_values <- ratData[top_indices]

# Bottom nth percentile
cat("Bottom nth percentile indices and values:\n")
print(data.frame(Index = bottom_indices, Value = bottom_values))

# Top nth percentile
cat("Top nth percentile indices and values:\n")
print(data.frame(Index = top_indices, Value = top_values))


# Filter the dataframe for bottom session values
bottom_df <- data[data$session %in% bottom_indices, ]

# Filter the dataframe for top session values
top_df <- data[data$session %in% top_indices, ]


# data = na.omit(data)
# choices = data$thiswell
# predictor = data$hiswell
# # random_predictor = sample(1:3, length(choices), replace = T)
# random_predictor = sample(predictor)

# n = length(choices)



# x_pred = cbind(predictor, rep(1, length(choices)))
# x_pred = cbind(random_predictor, rep(1, length(choices)))

# y = choices[1:length(choices)]


dataset = bottom_df
choices = dataset$thiswell
predictor = dataset$hiswell
n = length(choices)
x_pred = cbind(predictor, rep(0, length(choices)))
y = choices[1:length(choices)]

non_na_rows = complete.cases(x_pred)

# Subset x_pred and y based on non-NA rows
x_pred = x_pred[non_na_rows, ]
y = y[non_na_rows]

############### For overall predictions ##########################

# train = sample(seq(length(y)),ceiling(0.7*length(choices)),replace=FALSE)
# fit3 = glmnet(x_pred[train,], y[train], family = "multinomial")
# confusion.glmnet(fit3, newx = x_pred[-train, ], newy = y[-train], s = 0.01)
#  
#  
# fit3c = cv.glmnet(x_pred, y, family = "multinomial", type.measure="class", keep=TRUE)
# idmin = match(fit3c$lambda.min, fit3c$lambda)
# confusion.glmnet(fit3c$fit.preval, newy = y, family="multinomial")[[idmin]]


train = sample(seq(length(y)),ceiling(0.7*length(y)),replace=FALSE)
fit3c = cv.glmnet(x_pred, y, family = "multinomial", type.measure="class", keep=TRUE)
idmin = match(fit3c$lambda.min, fit3c$lambda)
cm = confusion.glmnet(fit3c$fit.preval, newy = y, family="multinomial")[[idmin]]
true_values <- colnames(cm)
predicted_values <- rownames(cm)
# Convert to numeric for comparison
predicted_values_num <- as.numeric(predicted_values)

if (all(c(1, 2, 3) %in% predicted_values_num)) {
  print("1, 2, 3 are in predicted values")
  correct = (cm[1,1] + cm[2,2] + cm[3,3]) / sum(cm[,])

} else if (all(c(1, 2) %in% predicted_values_num) && !3 %in% predicted_values_num) {
  print("Only 1 and 2 are in predicted values")
  correct = (cm[1,1] + cm[2,2]) / sum(cm[,])
  
}else if (all(c(1, 3) %in% predicted_values_num) && !2 %in% predicted_values_num) {
  print("Only 1 and 3 are in predicted values")
  correct = (cm[1,1] + cm[2,3]) / sum(cm[,])
 
} else if (all(c(2, 3) %in% predicted_values_num) && !1 %in% predicted_values_num) {
  print("Only 2 and 3 are in predicted values")
  correct = (cm[1,2] + cm[2,3]) / sum(cm[,])
  
} else if (1 %in% predicted_values_num && !2 %in% predicted_values_num && !3 %in% predicted_values_num) {
  print("Only 1 is in predicted values")
  correct = (cm[1,1]) / sum(cm[,])
  
} else if (2 %in% predicted_values_num && !1 %in% predicted_values_num && !3 %in% predicted_values_num) {
  print("Only 2 is in predicted values")
  correct = (cm[1,2]) / sum(cm[,])
  
} else if (3 %in% predicted_values_num && !1 %in% predicted_values_num && !2 %in% predicted_values_num) {
  print("Only 3 is in predicted values")
  correct = (cm[1,3]) / sum(cm[,])
  
} else {
  print("No valid predicted values found")
}


correct
write_clip(correct)

############# For blockwise predictions ##########################

chunk <- 250
n <- nrow(data)
r  <- rep(1:ceiling(n/chunk),each=chunk)[1:n]
sessionwise_data <- split(data,r)

# sessionwise_data = split(data, data$session)
# sessionwise_data = Filter(function(df) nrow(df) >= 10, sessionwise_data)
confusion_matrix = list()
fitted_obj = list()
pct_correct = list()



for (x in 1:length(sessionwise_data)) {
  temp_data = sessionwise_data[x]
  temp_data = na.omit(temp_data[[1]])
  set.seed(4444)  # for reproducibility
  # temp_data = temp_data[sample(nrow(temp_data), 100, replace = TRUE),]
  y = sample(temp_data$thiswell)
  predictor = temp_data$hiswell
  x_pred = cbind(sample(predictor), rep(1, length(y)))
  train = sample(seq(length(y)),ceiling(0.7*length(y)),replace=FALSE)
  fit3c = cv.glmnet(x_pred, y, family = "multinomial", type.measure="class", keep=TRUE)
  fitted_obj[[x]] = fit3c
  idmin = match(fit3c$lambda.min, fit3c$lambda)
  cm = confusion.glmnet(fit3c$fit.preval, newy = y, family="multinomial")[[idmin]]
  true_values <- colnames(cm)
  predicted_values <- rownames(cm)
  # Convert to numeric for comparison
  predicted_values_num <- as.numeric(predicted_values)
  
  if (all(c(1, 2, 3) %in% predicted_values_num)) {
    print("1, 2, 3 are in predicted values")
    correct = (cm[1,1] + cm[2,2] + cm[3,3]) / sum(cm[,])
    pct_correct[[x]] = correct
  } else if (all(c(1, 2) %in% predicted_values_num) && !3 %in% predicted_values_num) {
    print("Only 1 and 2 are in predicted values")
    correct = (cm[1,1] + cm[2,2]) / sum(cm[,])
    pct_correct[[x]] = correct
  }else if (all(c(1, 3) %in% predicted_values_num) && !2 %in% predicted_values_num) {
    print("Only 1 and 3 are in predicted values")
    correct = (cm[1,1] + cm[2,3]) / sum(cm[,])
    pct_correct[[x]] = correct
  } else if (all(c(2, 3) %in% predicted_values_num) && !1 %in% predicted_values_num) {
    print("Only 2 and 3 are in predicted values")
    correct = (cm[1,2] + cm[2,3]) / sum(cm[,])
    pct_correct[[x]] = correct
  } else if (1 %in% predicted_values_num && !2 %in% predicted_values_num && !3 %in% predicted_values_num) {
    print("Only 1 is in predicted values")
    correct = (cm[1,1]) / sum(cm[,])
    pct_correct[[x]] = correct
  } else if (2 %in% predicted_values_num && !1 %in% predicted_values_num && !3 %in% predicted_values_num) {
    print("Only 2 is in predicted values")
    correct = (cm[1,2]) / sum(cm[,])
    pct_correct[[x]] = correct
  } else if (3 %in% predicted_values_num && !1 %in% predicted_values_num && !2 %in% predicted_values_num) {
    print("Only 3 is in predicted values")
    correct = (cm[1,3]) / sum(cm[,])
    pct_correct[[x]] = correct
  } else {
    print("No valid predicted values found")
  }
  
  
 
  
  
  confusion_matrix[[x]] = as.data.frame(cm)
  # confusion_matrix[[x]] = cm
}

pct_correct = t(pct_correct)

library(clipr)
write_clip(pct_correct)

plot(unlist(pct_correct), type = "l", col= "red", ylim = c(0,1))
abline(h = 0.33, col = "grey50")

