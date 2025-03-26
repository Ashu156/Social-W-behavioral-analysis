# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:40:02 2025

@author: shukl
"""
#%% Load libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

#%% Load data and fit model

WT = pd.read_csv("C:/Users/shukl/OneDrive/Documents/MATLAB/WT.csv")
FX = pd.read_csv("C:/Users/shukl/OneDrive/Documents/MATLAB/FX.csv")

# Define and fit the GLM
model_formula = 'choice ~ p_Sub + p_Par + p_Both'
glm_model = smf.glm(formula=model_formula,
                    data=WT,
                    family=sm.families.Binomial()).fit()

# Print model summary
print(glm_model.summary())


#%% Plot predictor distribution individually for match and non-match events

# # Define predictors
# predictors = ['p_Sub', 'p_Par', 'p_Both']

# # Set up subplots
# fig, axes = plt.subplots(1, 3) # , figsize=(15, 5)

# # Loop through each predictor and plot the distributions
# for i, predictor in enumerate(predictors):
#     # sns.histplot(data = WT, x = predictor, hue = "choice", kde=True, bins = 30, ax=axes[i], palette={0: "red", 1: "blue"}, alpha=0.4)
#     sns.kdeplot(data = WT, x = predictor, hue = "choice",  ax=axes[i], palette={0: "red", 1: "blue"}, alpha=0.4)
#     axes[i].set_title(f'{predictor}')
#     axes[i].set_xlabel(predictor)
#     axes[i].set_ylabel('Frequency')

# plt.tight_layout()
# plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Define predictors
predictors = ['p_Sub', 'p_Par', 'p_Both']

# Set up subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 

# Initialize a variable to track the global y-axis limits
# y_limits = None

# Loop through each predictor and plot the distributions
for i, predictor in enumerate(predictors):
    # Plot for WT (solid lines)
    sns.kdeplot(data=WT, x=predictor, hue="choice", ax=axes[0, i], palette={0: "red", 1: "blue"})
    
    # Plot for DF2 (dashed lines)
    sns.kdeplot(data = FX, x=predictor, hue="choice", ax=axes[1, i], palette={0: "red", 1: "blue"}, 
                linestyle='--')

    # axes[i].set_title(f'{predictor}')
    # axes[i].set_xlabel(predictor)
    # axes[i].set_ylabel('Frequency')
    
    axes[0, i].set_title(f'{predictor}')
    axes[0, i].set_xlabel(predictor)
    axes[0, i].set_ylabel('Frequency')

    axes[1, i].set_title(f'{predictor}')
    axes[1, i].set_xlabel(predictor)
    axes[1, i].set_ylabel('Frequency')

    # Update the global y-axis limits after the first plot
    # if y_limits is None:
    #     y_limits = axes[i].get_ylim()
    # else:
    #     y_limits = (min(y_limits[0], axes[i].get_ylim()[0]), max(y_limits[1], axes[i].get_ylim()[1]))

# Set the same y-limits for all axes
# for ax in axes:
#     ax.set_ylim(y_limits)

plt.tight_layout()
plt.show()



#%% Plot coefficients individually

sns.lmplot(x = "p_Sub", y = "choice", data = WT, logistic = True, scatter_kws={"s": 5, "alpha": 0.5})
sns.lmplot(x = "p_Par", y = "choice", data = WT, logistic = True, scatter_kws={"s": 5, "alpha": 0.5})
sns.lmplot(x = "p_Both", y = "choice", data = WT, logistic = True, scatter_kws={"s": 5, "alpha": 0.5})

#%% Plot coefficients with 95% CI

# Extract coefficients and confidence intervals from the model
coef_df = pd.DataFrame({
    'Predictor': glm_model.params.index,  # Predictor names
    'Coefficient': glm_model.params.values,  # Coefficient values
    'Lower CI': glm_model.conf_int()[0],  # Lower bound of confidence interval
    'Upper CI': glm_model.conf_int()[1]   # Upper bound of confidence interval
})

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x='Predictor', y='Coefficient', data=coef_df, palette='coolwarm', edgecolor='black')
plt.errorbar(x=coef_df['Predictor'], y=coef_df['Coefficient'], 
             yerr=[coef_df['Coefficient'] - coef_df['Lower CI'], coef_df['Upper CI'] - coef_df['Coefficient']], 
             fmt='none', ecolor='black', capsize=5)

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Predictors')
plt.ylabel('Coefficient Value')
plt.title('GLM Coefficients')
plt.show()

#%%

import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt  # For raincloud plots
import numpy as np
import pandas as pd

WT_match = WT[WT["choice"] == 1]
WT_nonmatch = WT[WT["choice"] == 0]

FX_match = FX[FX["choice"] == 1]
FX_nonmatch = FX[FX["choice"] == 0]


col = "p_Par"

# Clean the datasets (drop NaNs)
WT_match_clean = FX_match[col].dropna()
FX_match_clean = FX_nonmatch[col].dropna()

# Ensure the datasets are not empty
if WT_match_clean.empty or FX_match_clean.empty:
    print("One of the groups has no valid data! T-test cannot be performed.")
else:
    # # Perform Welch’s t-test (equal_var=False handles unequal variances)
    # t_stat, p_value = stats.ttest_ind(WT_match_clean, FX_match_clean, equal_var=False)

    # # Print results
    # print(f"T-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")  # Use scientific notation for small p-values
    # Perform permutation test with 10,000 resamples
    # Perform the permutation test
    result = stats.permutation_test((WT_match_clean, FX_match_clean), 
                                statistic=lambda x, y: np.mean(x) - np.mean(y), 
                                permutation_type='independent', 
                                alternative='two-sided', 
                                n_resamples=10000)

    # Extract test statistic and p-value correctly
    print(f"Permutation Test: Statistic = {result.statistic:.4f}, p-value = {result.pvalue:.4f}")

    # plot
    # Extract the null distribution (permuted differences)
    null_distribution = result.null_distribution
    
    # Extract the observed test statistic
    observed_statistic = result.statistic
    
    # Create a combined DataFrame
    df_raincloud = pd.DataFrame({
        col: np.concatenate([WT_match_clean, FX_match_clean]), 
        "Group": ["WT"] * len(WT_match_clean) + ["FX"] * len(FX_match_clean)
    })
    
    # Plot the null distribution
    fig, axes = plt.subplots(2, 1, figsize=(8, 5)) 

    # Create Raincloud Plot
    pt.RainCloud(x="Group", y=col, data=df_raincloud, 
             palette="Set2", width_viol=1.0, width_box=.2, bw=.2, 
             point_size=3, alpha=0.6, orient="h", move=0.0, pointplot=True, ax=axes[0])
    
    # axes[0].set_xlim((-0.2, 1.2))

    # Histogram of null distribution
    axes[1].hist(null_distribution, bins=100, density=True, histtype='stepfilled', 
             alpha=0.6, color='gray')
    
    
    

    # KDE plot of null distribution
    sns.kdeplot(null_distribution, ax=axes[1])

    # Mark the observed test statistic
    axes[1].axvline(observed_statistic, color='red', linestyle='dashed', linewidth=2)
    
    
    # Compute 95% confidence interval (2.5th and 97.5th percentiles)
    ci_lower = np.percentile(null_distribution, 2.5)
    ci_upper = np.percentile(null_distribution, 97.5)
    
    # Mark the 95% Confidence Interval
    axes[1].axvline(ci_lower, color='blue', linestyle='dashed', linewidth=1)
    axes[1].axvline(ci_upper, color='blue', linestyle='dashed', linewidth=1)

    # Show legend
    # axes[1].legend()

plt.tight_layout()
plt.show()



# Calculate median values
wt_median = WT_match_clean.median()
fx_median = FX_match_clean.median()



# Draw median lines
# plt.axvline(wt_median, color="blue", linestyle="--", linewidth=2, label="WT Median")
# plt.axvline(fx_median, color="red", linestyle="--", linewidth=2, label="FX Median")

# Labels & Title
plt.xlabel("Group")
plt.ylabel(col)
# plt.title("Raincloud Plot of p_Both for WT and FX Groups with Medians")

# Add Legend
plt.legend()

# Show Plot
plt.show()

# Create a combined DataFrame
df_raincloud = pd.DataFrame({
    "score": np.concatenate([WT_match_clean, FX_match_clean]), 
    "group": ["WT"] * len(WT_match_clean) + ["FX"] * len(FX_match_clean)
})

#%% Plot the repeated measures data

# Add "Group" column before reshaping
WT["group"] = "WT"
FX["group"] = "FX"

# Combine WT and FX
df_combined = pd.concat([WT, FX], ignore_index=True)

# Melt the dataframe to reshape it into long format
df_raincloud = df_combined.melt(id_vars=["group"], value_vars=["p_Both", "p_Sub", "p_Par"], 
                            var_name="predictor", value_name="score")
dx = "group"; dy = "score"; dhue = "predictor"; ort = "h"; pal = "Set2"; sigma = .2
f, ax = plt.subplots(figsize=(7, 7))

ax=pt.RainCloud(x = dx, y = dy, hue = dhue, data = df_raincloud, palette = pal, bw = sigma, width_viol = .7,
                ax = ax, orient = ort , alpha = .3, dodge = True, pointplot = True, move = .0)

#%% Using SHAP

import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import time

data = (WT[['p_Sub', 'p_Par', 'p_Both']], WT['choice'].to_numpy())

X_train,X_test,Y_train,Y_test = train_test_split(*data, test_size=0.2, random_state=0)

# rather than use the whole training set to estimate expected values, we could summarize with
# a set of weighted kmeans, each weighted by the number of points they represent. But this dataset
# is so small we don't worry about it
#X_train_summary = shap.kmeans(X_train, 50)

def print_accuracy(f):
    print("Accuracy = {0}%".format(100*np.sum(f(X_test) == Y_test)/len(Y_test)))
    time.sleep(0.5) # to let the print get out before any progress bars
    
    
#%%

model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
model.fit(X_train, Y_train)


explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)
# X_test_array = X_test.toarray() # we need to pass a dense version for the plotting functions

shap.initjs()


shap.summary_plot(shap_values, X_test)


#%% 

ind = 0
shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test[ind,:]
)
