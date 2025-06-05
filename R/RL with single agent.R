# define parameters of the 4-armed bandit task
arm_means <- c(10,10,10,13)
arm_sd <- 1.5

# empty plot
plot(NA, 
     ylab = "probability density",
     xlab = "reward",
     type = 'n',
     bty='l',
     xlim = c(4, 20),
     ylim = c(0, 0.3))

# plot each arm's normally distributed reward
x <- seq(-10, 100, 0.1)
polygon(x, dnorm(x, arm_means[1], arm_sd), col = adjustcolor("royalblue", alpha.f = 0.2))
text(10, 0.29, "arms 1-3")
polygon(x, dnorm(x, arm_means[4], arm_sd), col = adjustcolor("orange", alpha.f = 0.2))
text(13, 0.29, "arm 4")

m <- matrix(NA, nrow = 4, ncol = 10,
            dimnames = list(c("arm1","arm2","arm3","arm4"), NULL))

for (i in 1:4) m[i,] <- round(rnorm(10, arm_means[i], arm_sd))

m

# for storing Q values for 4 arms on current trial
Q_values <- rep(0, 4)
names(Q_values) <- paste("arm", 1:4, sep="")

Q_values

beta <- 0

exp(beta * Q_values) / sum(exp(beta * Q_values))

Q_values[1:4] <- c(10,14,9,13)

exp(beta * Q_values) / sum(exp(beta * Q_values))

beta <- 0.3

round(exp(beta * Q_values) / sum(exp(beta * Q_values)), 2)

beta <- 5

round(exp(beta * Q_values) / sum(exp(beta * Q_values)), 2)



Q_values[1:4] <- rep(0,4)
beta <- 0.3

probs <- exp(beta * Q_values) / sum(exp(beta * Q_values))

choice <- sample(1:4, 1, prob = probs)
choice


reward <- round(rnorm(1, mean = arm_means[choice], sd = arm_sd))
reward


alpha <- 0.7

Q_values[choice] <- Q_values[choice] + alpha * (reward - Q_values[choice])

Q_values


#######          ##########

RL_single <- function(alpha, beta) {
  
  # set up arm rewards
  arm_means <- data.frame(p1 = c(10,10,10,13),
                          p2 = c(10,13,10,10),
                          p3 = c(13,10,10,10),
                          p4 = c(10,10,13,10))
  arm_sd <- 1.5
  
  # for storing Q values for 4 arms on current trial, initially all zero
  Q_values <- rep(0, 4)
  
  # for storing Q values, choices, rewards per trial
  output <- as.data.frame(matrix(NA, 100, 6))
  names(output) <- c(paste("arm", 1:4, sep=""), "choice", "reward")
  
  # t-loop
  for (t in 1:100) {
    
    # get softmax probabilities from Q_values and beta
    probs <- exp(beta * Q_values) / sum(exp(beta * Q_values))
    
    # choose an arm based on probs
    choice <- sample(1:4, 1, prob = probs)
    
    # get reward, given current time period
    if (t <= 25) reward <- round(rnorm(1, mean = arm_means$p1[choice], sd = arm_sd))
    if (t > 25 & t <= 50) reward <- round(rnorm(1, mean = arm_means$p2[choice], sd = arm_sd))
    if (t > 50 & t <= 75) reward <- round(rnorm(1, mean = arm_means$p3[choice], sd = arm_sd))
    if (t > 75) reward <- round(rnorm(1, mean = arm_means$p4[choice], sd = arm_sd))
    
    # update Q_values for choice based on reward and alpha
    Q_values[choice] <- Q_values[choice] + alpha * (reward - Q_values[choice])
    
    # store all in output dataframe
    output[t,1:4] <- Q_values
    output$choice[t] <- choice
    output$reward[t] <- reward
    
  }
  
  # record whether correct
  output$correct <- output$choice == c(rep(4,25),rep(2,25),rep(1,25),rep(3,25))
  
  # export output dataframe
  output
  
}

data_model17a <- RL_single(alpha = 0.7, beta = 0.3)

data_model17a[1:25,]

data_model17a[76:100,]