library(alluvial)
library(mHMMbayes)
library(readODS)

# data = read_ods("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/cohortAS2_for_mHMM.ods", sheet = 'combined_rats')
data = read_csv("E:/Jadhav lab data/Behavior/combined_rats.csv")
data = na.omit(data)
# dat <- data.frame(id = character(nrow(data)), stringsAsFactors = FALSE)
# 
# dat$id = data$rat
# dat$pMatch = data$match
data$state = data$state +1

data = as.data.frame(data)
# Specify parameters
m <- 3
n_dep <- 1
q_emiss <- c(2)

# Define starting values
start_TM <- diag(.8, m)
start_TM[lower.tri(start_TM) | upper.tri(start_TM)] <- 0.2 / (m - 1)
start_EM <- list(matrix(rep(1 / q_emiss[1], m * q_emiss[1]), 
                        byrow = TRUE, nrow = m, ncol = q_emiss[1]))

# Run mHMM with the sample data
set.seed(14532)  # For reproducibility
out_test <- mHMM(s_data = data, 
                 gen = list(m = m, n_dep = n_dep, q_emiss = q_emiss), 
                 start_val = c(list(start_TM), start_EM), return_path = TRUE, 
                 mcmc = list(J = 5000, burn_in = 500))

# Check the output
print(out_test)


summary(out_test)
gamma_pop <- obtain_gamma(out_test, level = "group", burn_in = 500)
gamma_subj <- obtain_gamma(out_test, level = "subject", burn_in = 500)

emiss_pop <- obtain_emiss(out_test, level = "group", burn_in = 500)
emiss_subj <- obtain_emiss(out_test, level = "subject", burn_in = 500)


library(RColorBrewer)
# Voc_col <- c(brewer.pal(3,"PuBuGn")[c(1,3,2)])
Voc_lab <- c("No Match", "Match")

plot(out_test, component = "emiss", dep = 1,
     dep_lab = c("Rats matching"), cat_lab = Voc_lab)


# Transition probabilities at the group level and for subject number 1, respectively:
plot(gamma_pop, col = rep(rev(brewer.pal(3,"PiYG"))[-2], each = m))
plot(gamma_subj, subj_nr = 1, col = rep(rev(brewer.pal(3,"PiYG"))[-2], each = m))

state_seq <- vit_mHMM(out_test, s_data = data, burn_in = 500, return_state_prob = TRUE)


# Assess model convergence and label-switching
# Run mHMM with the sample data
set.seed(98431)  # For reproducibility
out_test_b <- mHMM(s_data = data, 
                 gen = list(m = m, n_dep = n_dep, q_emiss = q_emiss), 
                 start_val = c(list(start_TM), start_EM),
                 mcmc = list(J = 5000, burn_in = 500))


# Check the output
print(out_test_b)


summary(out_test_b)
gamma_pop <- obtain_gamma(out_test_b, level = "group", burn_in = 500)
gamma_subj <- obtain_gamma(out_test_b, level = "subject", burn_in = 500)


library(RColorBrewer)

plot(out_test_b, component = "emiss", dep = 1,
     dep_lab = c("Rats matching"), cat_lab = Voc_lab)


# Transition probabilities at the group level and for subject number 1, respectively:
plot(gamma_pop, col = rep(rev(brewer.pal(3,"PiYG"))[-2], each = m))
plot(gamma_subj, subj_nr = 5, col = rep(rev(brewer.pal(3,"PiYG"))[-2], each = m))



par(mfrow = c(m,q_emiss[1]))
for(i in 1:m){
  for(q in 1:q_emiss[1]){
    plot(x = 1:5000, y = out_test$emiss_prob_bar[[1]][,(i-1) * q_emiss[1] + q], 
         ylim = c(0,1.4), yaxt = 'n', type = "l", ylab = "Transition probability",
         xlab = "Iteration",  col = "#8da0cb") 
    axis(2, at = seq(0,1, .2), las = 2)
    lines(x = 1:5000, y = out_test_b$emiss_prob_bar[[1]][,(i-1) * q_emiss[1] + q], col = "#e78ac3")
    legend("topright", col = c("#8da0cb", "#e78ac3"), lwd = 2, 
           legend = c("Starting value set 1", "Starting value set 2"), bty = "n")
  }
}

##################################################################################################################
######  Fit multiple HMMs to data (with single initialization) ###########

m = 5
n_dep <- 1
q_emiss <- c(2)

# Function to fit multiple HMMs and extract AIC from printed output
compare_AIC <- function(max_states = 7) {
  results <- data.frame(hidden_states = integer(0), AIC = numeric(0))
  models <- list()  # Initialize a list to store model objects
  
  # Iterate through the range of hidden states (m) to fit multiple models
  for (m in 2:max_states) {
    cat("Fitting HMM with", m, "hidden states...\n")
    
    # Define starting values for the transition matrix (TM) and emission matrix (EM)
    start_TM <- diag(0.8, m)
    start_TM[lower.tri(start_TM) | upper.tri(start_TM)] <- 0.2 / (m - 1)
    
    start_EM <- list(matrix(rep(1 / q_emiss[1], m * q_emiss[1]), 
                            byrow = TRUE, nrow = m, ncol = q_emiss[1]))
    
    # Fit the model and capture the printed output
    tryCatch({
      model <- mHMM(s_data = data, 
                    gen = list(m = m, n_dep = n_dep, q_emiss = q_emiss), 
                    start_val = c(list(start_TM), start_EM), 
                    mcmc = list(J = 2000, burn_in = 200))
      
      # Store the fitted model in the list
      models[[paste0("model_", m, "_states")]] <- model
      
      # Capture printed output as a character string
      output <- capture.output(print(model))
      
      # Extract AIC value using regex to find the relevant line
      aic_line <- grep("Average AIC over all subjects:", output, value = TRUE)
      model_AIC <- as.numeric(sub(".*: ", "", aic_line))
      
      # Store the result
      results <- rbind(results, data.frame(hidden_states = m, AIC = model_AIC))
      
    }, error = function(e) {
      cat("Error fitting model with", m, "states:", e$message, "\n")
    })
  }
  
  return(list(results = results, models = models))
}

# Run the comparison function
output <- compare_AIC(max_states = 7)
results <- output$results
models <- output$models  # Access all stored models

# Print and plot AIC results
print(results)
plot(results$hidden_states, results$AIC, type = "b", 
     xlab = "Number of Hidden States", ylab = "AIC", 
     main = "AIC Comparison Across Models")



library(RColorBrewer)
for (m in 1:6) {
  out_test = models[[m]]
  # Voc_col <- c(brewer.pal(3,"PuBuGn")[c(1,3,2)])
  Voc_lab <- c("No Match", "Match")
  
  plot(out_test, component = "emiss", dep = 1, col = Voc_col, 
       dep_lab = c("Rats matching"), cat_lab = Voc_lab)
  
}



##################################################################################################################
######  Fit multiple HMMs to data (with multiple random initializations) ###########

n_dep <- 1
q_emiss <- c(2)

# Function to fit multiple HMMs with random initialization of start_TM and start_EM
compare_AIC <- function(max_states = 5, num_inits = 5) {
  results <- data.frame(hidden_states = integer(0), AIC = numeric(0))
  best_models <- list()  # Initialize a list to store the best model objects
  
  # Iterate through the range of hidden states (m) to fit multiple models
  for (m in 2:max_states) {
    cat("Fitting HMM with", m, "hidden states...\n")
    
    best_AIC <- Inf
    best_model <- NULL  # Placeholder for the best model with the lowest AIC
    
    # Perform multiple random initializations
    for (init in 1:num_inits) {
      cat("Initialization", init, "for model with", m, "hidden states...\n")
      
      # Randomly initialize the transition matrix (TM) with each row summing to 1
      start_TM <- matrix(runif(m * m), nrow = m, ncol = m)
      start_TM <- t(apply(start_TM, 1, function(row) row / sum(row)))  # Normalize each row
      
      # Randomly initialize the emission matrix (EM) with each row summing to 1
      start_EM <- list(matrix(runif(m * q_emiss[1]), byrow = TRUE, nrow = m, ncol = q_emiss[1]))
      start_EM[[1]] <- t(apply(start_EM[[1]], 1, function(row) row / sum(row)))  # Normalize each row
      
      # Fit the model and capture the printed output
      tryCatch({
        model <- mHMM(s_data = data, 
                      gen = list(m = m, n_dep = n_dep, q_emiss = q_emiss), 
                      start_val = c(list(start_TM), start_EM), return_path = TRUE, 
                      mcmc = list(J = 5000, burn_in = 500))
        # plot(model, component = "emiss", dep = 1,
        #      dep_lab = c("Rats matching"), cat_lab = Voc_lab)
        # Capture printed output as a character string
        output <- capture.output(print(model))
        
        # Extract AIC value using regex to find the relevant line
        aic_line <- grep("Average AIC over all subjects:", output, value = TRUE)
        model_AIC <- as.numeric(sub(".*: ", "", aic_line))
        
        # If this model has a lower AIC, update the best model and best AIC
        if (!is.na(model_AIC) && model_AIC < best_AIC) {
          best_AIC <- model_AIC
          best_model <- model
        }
        
      }, error = function(e) {
        cat("Error fitting model with", m, "states on initialization", init, ":", e$message, "\n")
      })
    }
    
    # Store the best model and AIC for the current number of hidden states
    if (!is.null(best_model)) {
      best_models[[paste0("model_", m, "_states")]] <- best_model
      results <- rbind(results, data.frame(hidden_states = m, AIC = best_AIC))
    }
  }
  
  return(list(results = results, models = best_models))
}

# Run the comparison function
output <- compare_AIC(max_states = 5, num_inits = 10)
results <- output$results
models <- output$models  # Access all stored models

# Print and plot AIC results
print(results)
plot(results$hidden_states, results$AIC, type = "b", 
     xlab = "Number of Hidden States", ylab = "AIC", 
     main = "AIC Comparison Across Models")





# Plot transition and/or emission matrices for different hidden states
library(RColorBrewer)



for (m in 1:length(models)) {
  out_test = models[[m]]
  # Voc_col <- c(brewer.pal(3,"PuBuGn")[c(1,3,2)])
  Voc_lab <- c("No Match", "Match")
  
  plot(out_test, component = "emiss", dep = 1,
       dep_lab = c("Rats matching"), cat_lab = Voc_lab)
  
}

#################### Pick best model based on AIC values ######################
out_test = models[[2]]


plot(out_test, component = "emiss", dep = 1,
     dep_lab = c("Rats matching"), cat_lab = Voc_lab)


# Get the sequence of most likely states
state_seq <- vit_mHMM(out_test, s_data = data, return_state_prob = TRUE)

# Load required packages
library(dplyr)
library(tidyverse)
library(tidyverse)

state_seq <- state_seq %>%
       group_by(subj) %>%
       mutate(time = row_number()) %>%
       ungroup()

# Reshape data to long format
state_seq_long <- state_seq %>%
  pivot_longer(cols = starts_with("pr_state"), 
               names_to = "states", 
               values_to = "probability")

# Plotting probability of hidden state for all individuals
ggplot(state_seq_long, aes(x = time, y = probability, color = states)) +
  geom_line(alpha = 0.4) +
  geom_smooth(se = TRUE, method = "loess", span = 0.05) +  # Smooth line
  labs(x = "Time", y = "State Probability") +
  theme_minimal() +
  facet_wrap(~ subj, scales = "free_x") +
  ggtitle("State Sequences for Each Subject") +
  scale_color_manual(values = c("pr_state_1" = "blue", "pr_state_2" = "red", "pr_state_3" = "green"))


summary(out_test)


############ Get a global transition matrix for all subjects ###################
gamma_pop <- obtain_gamma(out_test, level = "group", burn_in = 500)
gamma_subj <- obtain_gamma(out_test, level = "subject", burn_in = 500)

# Number of subjects
n_subjects <- length(gamma_subj)

# Number of states (assumed to be 3 as per your data)
n_states <- 3

# Initialize a 3D array to store transition matrices for all subjects
global_transition_matrix <- array(NA, dim = c(n_subjects, n_states, n_states))

# Populate the 3D array with the transition matrices from each subject
for (i in 1:n_subjects) {
  # Extract the transition matrix for the current subject
  trans_matrix <- gamma_subj[[paste("Subject", i)]]
  
  # Store it in the 3D array
  global_transition_matrix[i, , ] <- as.matrix(trans_matrix)
}

# Check the dimensions of the resulting array
print(dim(global_transition_matrix))  # Should be (20, #latent states, #latent states)

############ Get a global emission matrix for all subjects ###################

emiss_pop <- obtain_emiss(out_test, level = "group", burn_in = 500)
emiss_subj <- obtain_emiss(out_test, level = "subject", burn_in = 500)

# Number of subjects
n_subjects <- length(gamma_subj)

# Number of states and categories (based on your data)
n_states <- 3
n_categories <- 2

# Initialize a 3D array to store emission matrices for all subjects
global_emission_matrix <- array(NA, dim = c(n_subjects, n_states, n_categories))

# Populate the 3D array with the emission matrices from each subject
for (i in 1:n_subjects) {
  # Construct the subject label
  subject_label <- paste("Subject", i)
  
  # Check if the subject's data exists in the list
  if (is.list(emiss_subj[[1]]) && subject_label %in% names(emiss_subj[[1]])) {
    emiss_matrix <- emiss_subj[[1]][[subject_label]]
    
    # Check if the extracted matrix has the correct dimensions
    if (is.matrix(emiss_matrix) && nrow(emiss_matrix) == n_states && ncol(emiss_matrix) == n_categories) {
      # Store the matrix in the 3D array
      global_emission_matrix[i, , ] <- as.matrix(emiss_matrix)
    } else {
      warning(paste("Mismatch in dimensions for", subject_label))
    }
  } else {
    warning(paste("Data missing for", subject_label))
  }
}

# Check the dimensions of the resulting array
print(dim(global_emission_matrix))  # Should be (20, 3, 2)

# Optional: Display the emission matrix for the 2nd subject
if (all(!is.na(global_emission_matrix[2, , ]))) {
  print(global_emission_matrix[2, , ])
} else {
  cat("Emission data for Subject 2 is missing or incomplete.\n")
}


# Initialize a 3D array to store emission matrices for all subjects
global_cat_emission_matrix <- array(NA, dim = c(n_subjects, 5000, n_states*n_categories))

# Populate the 3D array with the emission matrices from each subject
for (i in 1:n_subjects) {
  # Construct the subject label
  # subject_label <- paste("Subject", i)
  
  # Check if the subject's data exists in the list
 
    cat_emiss <- out_test[[2]][[i]]$cat_emiss
    
    # Check if the extracted matrix has the correct dimensions
    if (is.matrix(cat_emiss) && nrow(cat_emiss) ==  5000 && ncol(cat_emiss) == n_categories*n_states) {
      # Store the matrix in the 3D array
      global_cat_emission_matrix[i, , ] <- as.matrix(cat_emiss)
    } else {
      warning(paste("Mismatch in dimensions for", subject_label))
    }
  
}

# Check the dimensions of the resulting array
print(dim(global_cat_emission_matrix))  # Should be (20, 5000, 6)

#################### Save as .npy files #######################################
library(reticulate)

# Use Python's NumPy via reticulate
np <- import("numpy")

# Save the 3D matrices as .npy files
np$save("global_transition_matrix_2states.npy", global_transition_matrix)
np$save("global_emission_matrix_2states.npy", global_emission_matrix)
np$save("global_cat_emission_matrix_2states.npy", global_cat_emission_matrix)

cat("3D matrices saved as .npy files.\n")
