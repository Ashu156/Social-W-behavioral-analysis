library(VLTimeCausality)
# Generate simulation data
TS <- VLTimeCausality::SimpleSimulationVLtimeseries()

VLTimeCausality::plotTimeSeries(TS$X,TS$Y)


# Run the function
out<-VLTimeCausality::VLGrangerFunc(Y=TS$Y,X=TS$X)

out$BICDiffRatio
out$XgCsY

################################################################################
library(lmtest)
data(ChickEgg)
ChickEgg <- as.data.frame(ChickEgg)

#============ The the egg causes chicken. 
out_test1 <- VLTimeCausality::VLGrangerFunc(X=ChickEgg$egg,Y=ChickEgg$chicken)
out_test1$p.val

out_test1$XgCsY_ftest


#============ The reverse direction has no causal relation
out_test2 <- VLTimeCausality::VLGrangerFunc(Y=ChickEgg$egg,X=ChickEgg$chicken)
out_test2$p.val

out_test2$XgCsY_ftest


#============= For actual dataset ===========================

rat1 <- read.csv("E:/Jadhav lab data/Behavior/CohortAS2/Social W/50%/12-04-2023/log12-04-2023(9-FXM108-FXM109)-Rat1_position_linear.csv")
rat2 <- read.csv("E:/Jadhav lab data/Behavior/CohortAS2/Social W/50%/12-04-2023/log12-04-2023(9-FXM108-FXM109)-Rat2_position_linear.csv")

X =  na_interpolation(rat1$linear_position)
Y =  na_interpolation(rat2$linear_position)

VLTimeCausality::plotTimeSeries(X,Y)


# Define the windowing function
window <- function(vector, window_length, overlap_percentage = 75) {
  # Ensure overlap percentage is between 0 and 100
  if (overlap_percentage < 0 || overlap_percentage > 100) {
    stop("Overlap percentage must be between 0 and 100")
  }
  
  # Calculate the step size based on overlap percentage
  step_size <- window_length * (1 - overlap_percentage / 100)
  
  # Initialize list to store windows
  windows <- list()
  
  # Loop through the vector and extract windows
  start <- 1
  while (start <= length(vector) - window_length + 1) {
    end <- start + window_length - 1
    windows[[length(windows) + 1]] <- vector[start:end]
    start <- start + step_size
  }
  
  # Return the list of windows
  return(windows)
}

Xw <- window(X, window_length = 240, overlap_percentage = 75)
Yw <- window(Y, window_length = 240, overlap_percentage = 75)

# Initialize lists
gc <- list()
bicXY <- list()
bicYX <- list()

# Loop through the data with error handling
for (x in 1:length(Xw)) {
  tryCatch({
    # Compute VL Granger causality
    outXY <- VLTimeCausality::VLGrangerFunc(Y = Yw[[x]], X = Xw[[x]])  # Access the x-th element of Xw and Yw
    outYX <- VLTimeCausality::VLGrangerFunc(Y = Xw[[x]], X = Yw[[x]])  # Access the x-th element
    
    # Use proper logical OR '||' in R
    if (outXY$XgCsY || outYX$XgCsY) {
      gc[[x]] <- 1  # Assign 1 to gc if condition is met
    } else {
      gc[[x]] <- 0  # Assign 0 if the condition is not met
    }
    
    # Assign BIC difference ratios
    bicXY[[x]] <- outXY$BICDiffRatio
    bicYX[[x]] <- outYX$BICDiffRatio
    
  }, error = function(e) {
    # In case of an error, print the error message and continue the loop
    message(sprintf("Error at iteration %d: %s", x, e$message))
    gc[[x]] <- NA  # Assign NA if there's an error
    bicXY[[x]] <- NA
    bicYX[[x]] <- NA
  })
}