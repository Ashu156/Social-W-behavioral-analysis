N <- 100  # number of agents
f <- 2 # conformity strength
n <- 1:N  # number of N agents who picked arm 1; arm 2 is then N-n
freq <- n / N  # frequencies of arm 1
prob <- n^f / (n^f + (N-n)^f)  # probabilities according to equation above

plot(freq, 
     prob, 
     type = 'l',
     xlab = "frequency of arm 1",
     ylab = "probability of choosing arm 1")

abline(a = 0, b = 1, lty = 3)  # dotted line for unbiased transmission baseline


########################                     ##############################

RL_social <- function(N, alpha_mu, alpha_sd, beta_mu, beta_sd, s, f) {
  
  # set up arm rewards
  arm_means <- data.frame(p1 = c(10,13,10,10),
                          p2 = c(10,10,10,13),
                          p3 = c(13,10,10,10),
                          p4 = c(10,10,13,10))
  arm_sd <- 1.5
  
  # draw agent beta, alpha, s and f from overall mean and sd
  beta <- rnorm(N, beta_mu, beta_sd)  # inverse temperatures
  alpha <- rnorm(N, alpha_mu, alpha_sd)  # learning rates
  
  # avoid impossible values
  alpha[alpha < 0] <- 0  # ensure all alphas are >0
  alpha[alpha > 1] <- 1  # ensure all alphas are <1
  beta[beta < 0] <- 0  # ensure all betas are >0
  
  # for storing Q values for 4 arms on current trial, initially all zero
  Q_values <- matrix(data = 0,
                     nrow = N, 
                     ncol = 4)
  
  # for storing choices and rewards per agent per trial
  output <- data.frame(trial = rep(1:100, each = N),
                       agent = rep(1:N, 100),
                       choice = rep(NA, 100*N),
                       reward = rep(NA, 100*N))
  
  # vector to hold frequencies of choices for conformity
  n <- rep(NA, 4)
  
  # t-loop
  for (t in 1:100) {
    
    # get asocial softmax probabilities p_RL from Q_values
    p_RL <- exp(beta * Q_values) / rowSums(exp(beta * Q_values))
    
    # get social learning probabilities p_SL from t=2 onwards
    if (t == 1) {
      
      probs <- p_RL
      
    } else {
      
      # get number of agents who chose each option
      for (arm in 1:4) n[arm] <- sum(output[output$trial == (t-1),]$choice == arm)
      
      # conformity according to f
      p_SL <- n^f / sum(n^f)
      
      # convert p_SL to N-row matrix to match p_RL
      p_SL <- matrix(p_SL, nrow = N, ncol = 4, byrow = T)
      
      # update probs by combining p_RL and p_SL according to s
      probs <- (1-s)*p_RL + s*p_SL
      
    }
    
    # choose an arm based on probs
    choice <- apply(probs, 1, function(x) sample(1:4, 1, prob = x))
    
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
  output$correct <- output$choice == c(rep(2,25*N),rep(4,25*N),rep(1,25*N),rep(3,25*N))
  
  # export output dataframe
  output
  
}

data_model17c <- RL_social(N = 200,
                           alpha_mu = 0.7,
                           alpha_sd = 0.1,
                           beta_mu = 0.3,
                           beta_sd = 0.1,
                           s = 0,
                           f = 1)

plot_correct(data_model17c)

data_model17c <- RL_social(N = 200,
                           alpha_mu = 0.7,
                           alpha_sd = 0.1,
                           beta_mu = 0.3,
                           beta_sd = 0.1,
                           s = 0.3,
                           f = 1)

plot_correct(data_model17c)

data_model17c <- RL_social(N = 200,
                           alpha_mu = 0.7,
                           alpha_sd = 0.1,
                           beta_mu = 0.3,
                           beta_sd = 0.1,
                           s = 0.3,
                           f = 2)

plot_correct(data_model17c)

data_model17c <- RL_social(N = 200,
                           alpha_mu = 0.7,
                           alpha_sd = 0.1,
                           beta_mu = 0.3,
                           beta_sd = 0.1,
                           s = 0.8,
                           f = 2)

plot_correct(data_model17c)