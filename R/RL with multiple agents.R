N <- 10

alpha_mu <- 0.7
alpha_sd <- 0.1

beta_mu <- 0.3
beta_sd <- 0.1

beta <- rnorm(N, beta_mu, beta_sd)  # inverse temperatures
alpha <- rnorm(N, alpha_mu, alpha_sd)  # learning rates
alpha[alpha < 0] <- 0  # ensure all alphas are >0
alpha[alpha > 1] <- 1  # ensure all alphas are <1
beta[beta < 0] <- 0  # ensure all betas are >0

data.frame(agent = 1:N, alpha, beta)


# for storing Q values for 4 arms on current trial, initially all zero
Q_values <- matrix(data = 0,
                   nrow = N, 
                   ncol = 4,
                   dimnames = list(NULL, paste("arm", 1:4, sep="")))

Q_values

probs <- exp(beta * Q_values) / rowSums(exp(beta * Q_values))
probs

choice <- apply(probs, 1, function(probs) sample(1:4, 1, prob = probs))
choice

reward <- round(rnorm(N, mean = arm_means[choice], sd = arm_sd))
reward

for (arm in 1:4) {
  
  chosen <- which(choice==arm) 
  
  Q_values[chosen, arm] <- Q_values[chosen, arm] + 
    alpha[chosen] * (reward[chosen] - Q_values[chosen, arm])
  
}

Q_values

#################                      ##############################

RL_multiple <- function(N, alpha_mu, alpha_sd, beta_mu, beta_sd) {
  
  # set up arm rewards
  arm_means <- data.frame(p1 = c(10,10,10,13),
                          p2 = c(10,13,10,10),
                          p3 = c(13,10,10,10),
                          p4 = c(10,10,13,10))
  arm_sd <- 1.5
  
  # draw agent beta and alpha from overall mean and sd
  beta <- rnorm(N, beta_mu, beta_sd)  # inverse temperatures
  alpha <- rnorm(N, alpha_mu, alpha_sd)  # learning rates
  alpha[alpha < 0] <- 0  # ensure all alphas are >0
  alpha[alpha > 1] <- 1  # ensure all alphas are <1
  beta[beta < 0] <- 0  # ensure all betas are >0
  
  # for storing Q values for k arms on current trial, initially all zero
  Q_values <- matrix(data = 0,
                     nrow = N, 
                     ncol = 4)
  
  # for storing choices and rewards per agent per trial
  output <- data.frame(trial = rep(1:100, each = N),
                       agent = rep(1:N, 100),
                       choice = rep(NA, 100*N),
                       reward = rep(NA, 100*N))
  
  # t-loop
  for (t in 1:100) {
    
    # get softmax probabilities from Q_values and beta
    probs <- exp(beta * Q_values) / rowSums(exp(beta * Q_values))
    
    # choose an arm based on probs
    choice <- apply(probs, 1, function(probs) sample(1:4, 1, prob = probs))
    
    # get reward
    if (t <= 25) reward <- round(rnorm(N, mean = arm_means$p1[choice], sd = arm_sd))
    if (t > 25 & t <= 50) reward <- round(rnorm(N, mean = arm_means$p2[choice], sd = arm_sd))
    if (t > 50 & t <= 75) reward <- round(rnorm(N, mean = arm_means$p3[choice], sd = arm_sd))
    if (t > 75) reward <- round(rnorm(N, mean = arm_means$p4[choice], sd = arm_sd))
    
    # update Q values one arm at a time
    for (arm in 1:4) {
      
      chosen <- which(choice==arm) 
      
      Q_values[chosen,arm] <- Q_values[chosen,arm] + 
        alpha[chosen] * (reward[chosen] - Q_values[chosen,arm])
      
    }
    
    # store choice and reward in output
    output[output$trial == t,]$choice <- choice
    output[output$trial == t,]$reward <- reward
    
  }
  
  # record whether correct
  output$correct <- output$choice == c(rep(4,25*N),rep(2,25*N),rep(1,25*N),rep(3,25*N))
  
  # export output dataframe
  output
  
}

data_model17b <- RL_multiple(N = 200,
                             alpha_mu = 0.7,
                             alpha_sd = 0.1,
                             beta_mu = 0.3,
                             beta_sd = 0.1)

head(data_model17b)


###################                 #############################

plot_correct <- function(output) {
  
  plot_data <- rep(NA, 100)
  for (t in 1:100) plot_data[t] <- mean(output$correct[output$trial == t])
  
  plot(x = 1:100,
       y = plot_data,
       type = 'l',
       ylab = "frequency correct",
       xlab = "timestep",
       ylim = c(0,1),
       lwd = 2)
  
  # dotted vertical lines indicate changes in optimal
  abline(v = c(25,50,75),
         lty = 2)
  
  # dotted horizontal line indicates chance success rate
  abline(h = 0.25,
         lty = 3)
  
}

data_model17b <- RL_multiple(N = 200,
                             alpha_mu = 0.7,
                             alpha_sd = 0,
                             beta_mu = 0.3,
                             beta_sd = 0)

plot_correct(data_model17b)


data_model17b <- RL_multiple(N = 200,
                             alpha_mu = 0.7,
                             alpha_sd = 0.1,
                             beta_mu = 0.3,
                             beta_sd = 0.1)

plot_correct(data_model17b)


data_model17b <- RL_multiple(N = 200,
                             alpha_mu = 0.7,
                             alpha_sd = 1,
                             beta_mu = 0.3,
                             beta_sd = 1)

plot_correct(data_model17b)


data_model17b <- RL_multiple(N = 200,
                             alpha_mu = 0.7,
                             alpha_sd = 0,
                             beta_mu = 5,
                             beta_sd = 0)

plot_correct(data_model17b)

data_model17b <- RL_multiple(N = 200,
                             alpha_mu = 0.7,
                             alpha_sd = 0,
                             beta_mu = 0.1,
                             beta_sd = 0)

plot_correct(data_model17b)

