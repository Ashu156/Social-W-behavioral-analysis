data = read.csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\CohortER2\\cohortER2_rat_2_50.csv")
data = na.omit(data)
choices = data$match
rewards = data$Reward
# predictor = data$hiswell
# # random_predictor = sample(1:3, length(choices), replace = T)
# random_predictor = sample(predictor)
# 
# 
n = length(choices)

lag_choices1 <- choices[1:(n - 1)]
lag_choices2 <- c(rep(NA, 1), choices[1:(n - 2)])
lag_choices3 <- c(rep(NA, 2), choices[1:(n - 3)])
# 
lag_rewards1 <- rewards[1:(n - 1)]
lag_rewards2 <- c(rep(NA, 1), rewards[1:(n - 2)])
lag_rewards3 <- c(rep(NA, 2), rewards[1:(n - 3)])
# 
lag_RC1 <- choices[1:(n - 1)] + rewards[1:(n - 1)]
lag_rewards2 <- c(rep(NA, 1), rewards[1:(n - 2)])
lag_rewards3 <- c(rep(NA, 2), rewards[1:(n - 3)])

x_choices <- cbind(
  choices[1:(n - 1)],
  c(rep(NA, 1), choices[1:(n - 2)]),
  c(rep(NA, 2), choices[1:(n - 3)])


)

x_rewards <- cbind(
                   rewards[1:(n - 1)],
                   c(rep(NA, 1), rewards[1:(n - 2)]),
                   c(rep(NA, 2), rewards[1:(n - 3)])


)

x_RC <- cbind(
                   choices[1:(n - 1)] + rewards[1:(n - 1)],
                   c(rep(NA, 1), choices[1:(n - 2)] + rewards[1:(n - 2)]),
                   c(rep(NA, 2), choices[1:(n - 3)] + rewards[1:(n - 3)])

)

x_choices = na.omit(x_choices)
x_rewards = na.omit(x_rewards)
x_RC = na.omit(x_RC)


# x_pred = cbind(predictor, rep(1, length(choices)))
# x_pred = cbind(random_predictor, rep(1, length(choices)))

y = choices[4:length(choices)]
# y = choices[1:length(choices)]


library(glmnet)

fit_choices <- glmnet(x_choices, y, family = "binomial")
plot(fit_choices, label = TRUE)
print(fit_choices)

fit_rewards <- glmnet(x_rewards, y, family = "binomial")
plot(fit_rewards, label = TRUE)
print(fit_rewards)

fit_RC <- glmnet(x_RC, y, family = "binomial")
plot(fit_RC, label = TRUE)
print(fit_RC)

# predict(fit, newx = x[1:5,], type = "class", s = c(0.05, 0.01))
cvfit_choices <- cv.glmnet(x_choices, y, family = "binomial", type.measure = "deviance",
                           intercept = TRUE, alpha = 1.0, nfolds = 10)
cvfit_rewards <- cv.glmnet(x_rewards, y, family = "binomial", type.measure = "deviance",
                           intercept = TRUE, alpha = 1.0, nfolds = 10)
cvfit_RC <- cv.glmnet(x_RC, y, family = "binomial", type.measure = "deviance",
                           intercept =TRUE, alpha = 1.0, nfolds = 10)
plot(cvfit_choices)
plot(cvfit_rewards)
plot(cvfit_RC)
# 
# # Get the beta coefficients
# beta_choices = coef(cvfit_choices, s = "lambda.min")
beta_rewards = coef(cvfit_rewards, s = "lambda.min")
# beta_RC = coef(cvfit_RC, s = "lambda.min")
# # beta_coeffs = coef(cvfit)
# devRatio_choices = cvfit_choices$glmnet.fit$dev.ratio[cvfit_choices$index[1]]
devRatio_rewards = cvfit_rewards$glmnet.fit$dev.ratio[cvfit_rewards$index[1]]
# devRatio_RC = cvfit_RC$glmnet.fit$dev.ratio[cvfit_RC$index[1]]
# 
# ratios = (c(devRatio_choices, devRatio_rewards, devRatio_RC))*100
# 
# print(ratios)


# foldid <- sample(1:10, size = length(y), replace = TRUE)
# cv1  <- cv.glmnet(x_choices, y, family = "binomial", type.measure = "deviance",foldid = foldid, alpha = 1)
# cv.5 <- cv.glmnet(x_choices, y, family = "binomial", type.measure = "deviance",foldid = foldid, alpha = 0.5)
# cv0  <- cv.glmnet(x_choices, y, family = "binomial", type.measure = "deviance",foldid = foldid, alpha = 0)




# mites = read.csv("C:/Users/shukl/Downloads/mites.csv", header = TRUE)
df = as.data.frame(cbind(c1 = lag_choices1, 
                         c2 = lag_choices2, 
                         c3 = lag_choices3, 
                         r1 = lag_rewards1, 
                         r2 = lag_rewards2, 
                         r3 = lag_rewards3, 
                         y = y))

logit.reg <- glm(y ~ c1*r1,
                 data = df,
                 family = binomial(link = "logit"))
summary(logit.reg)


# Extracting model coefficients
summary(logit.reg)$coefficients

# odds for matching
exp(logit.reg$coefficient[2:3])

# calculate the pseudo-R2
pseudoR2 <- (logit.reg$null.deviance - logit.reg$deviance) / logit.reg$null.deviance
pseudoR2

library(tidyverse)
ggplot(df, aes(x = r1, y = y)) + geom_point() +
  stat_smooth(method = "glm", method.args = list(family=binomial), se = TRUE) + xlab("last choice") +
  ylab("Probability of matching") +
  ggtitle("Probability of matching against last choice")+theme_classic()
