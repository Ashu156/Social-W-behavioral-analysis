# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:25:04 2025

@author: shukl
"""
#%%

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import smoothfit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu

#%%

import os
import pandas as pd

directory = "E:/Jadhav lab data/Behavior" # master directory
# cohorts = ["CohortAS1", "CohortAS2", "CohortER1", "CohortER2", "CohortAS3"] # cohorts
cohorts = ["CohortAS6", "CohortER1"]
folder = "Social W"  # folder name


# files = [
#     "cohortAS1_following_trials_50.csv",
#     "cohortAS2_following_trials_50.csv",
#     "cohortER1_following_trials_50.csv",
#     "cohortER2_following_trials_50.csv",
#     "cohortAS3_following_trials_50.csv"
# ] # files


files = [
    "cohortAS6_following_trials_100.csv",
    # "cohortAS2_following_trials_100.csv",
    "cohortER1_following_trials_100.csv",
    # "cohortER2_following_trials_100.csv",
    # "cohortAS3_following_trials_100.csv"
] # files


# WT = ['XFN2', 'XFN4', 'FXM102', 'FXM103', 'FXM105', 
#       'FXM107', 101, 110, 201, 202]

WT = ['FX116', 'FX117', 'FX120', 'FX123', 'ER4']

# FX = ['XFN1', 'XFN3', 'FXM108', 'FXM109', 'ER1', 
#       'ER2',104, 106, 205, 206]

FX = ['FX114', 'FX115', 'FX121', 'FX122','ER3']

# Dictionaries to store data
wt_data = {}
fx_data = {}

# Loop through files
for cohort, file in zip(cohorts, files):
    path = os.path.join(directory, cohort, folder, file)
    try:
        df = pd.read_csv(path)

        # Group by rats
        for rat_id in df['rat'].unique():
            rat_df = df[df['rat'] == rat_id]
            
            if rat_id in WT:
                if not isinstance(rat_id, str):
                    rat_id = str(rat_id)
                wt_data[rat_id] = rat_df
            elif rat_id in FX:
                if not isinstance(rat_id, str):
                    rat_id = str(rat_id)
                fx_data[rat_id] = rat_df
            else:
                print(f"⚠️ Rat ID '{rat_id}' not found in WT or FX lists.")

        print(f"✅ Processed: {path}")
    except FileNotFoundError:
        print(f"❌ File not found: {path}")

# Optional: view keys
print("\nWT rats loaded:", list(wt_data.keys()))
print("FX rats loaded:", list(fx_data.keys()))

#%% Cramer's V (association between categorical variables)

# Define the Cramer's V function
def CramersV(df):
    contingency_table = pd.crosstab(df['mywell'], df['hiswell'])
    chi2, _, dof, _ = chi2_contingency(contingency_table)
    
    n = contingency_table.sum().sum()
    rows, cols = contingency_table.shape
    min_dim = min(rows - 1, cols - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
    return cramers_v

# Function to compute session-wise Cramer's V for each rat
def compute_sessionwise_cramersv(data_dict):
    result = {}
    for rat_id, df in data_dict.items():
        session_values = []
        for session_id, session_df in df.groupby('session'):
            try:
                v = CramersV(session_df)
                session_values.append(v)
            except Exception as e:
                print(f"Error for rat {rat_id}, session {session_id}: {e}")
                session_values.append(np.nan)
        result[rat_id] = np.array(session_values)
    return result

# Apply to both WT and FX
wt_cramersv = compute_sessionwise_cramersv(wt_data)
fx_cramersv = compute_sessionwise_cramersv(fx_data)
# =============================================================================
# 
# Plot Cramer's V 
# 
# 
# =============================================================================

sess = 34

new_wt_cramersv = []
new_fx_cramersv = []

for key in wt_cramersv.keys():
    non_nan_vals = wt_cramersv[key][~np.isnan(wt_cramersv[key])]
    if len(non_nan_vals) >= sess:
        new_wt_cramersv.append(non_nan_vals[:sess])
    else:
        # Pad with NaNs if fewer than sess values
        padded = np.pad(non_nan_vals, (0, sess - len(non_nan_vals)), constant_values=np.nan)
        new_wt_cramersv.append(padded)

new_wt_cramersv = np.stack(new_wt_cramersv, axis=0)

for key in fx_cramersv.keys():
    non_nan_vals = fx_cramersv[key][~np.isnan(fx_cramersv[key])]
    if len(non_nan_vals) >= sess:
        new_fx_cramersv.append(non_nan_vals[:sess])
    else:
        padded = np.pad(non_nan_vals, (0, sess - len(non_nan_vals)), constant_values=np.nan)
        new_fx_cramersv.append(padded)

new_fx_cramersv = np.stack(new_fx_cramersv, axis=0)

lmbda = 10.0e-1

WTmeanPerf = new_wt_cramersv.mean(axis=0)
WTerrPerf = new_wt_cramersv.std(axis=0) / np.sqrt(new_wt_cramersv.shape[0])

FXmeanPerf = new_fx_cramersv.mean(axis=0)
FXerrPerf = new_fx_cramersv.std(axis=0) / np.sqrt(new_fx_cramersv.shape[0])

# MixedWTmeanPerf = mixedWT.mean(axis=0)
# MixedWTerrPerf = mixedWT.std(axis=0) / np.sqrt(mixedWT.shape[0])

# MixedFXmeanPerf = mixedFX.mean(axis=0)
# MixedFXerrPerf = mixedFX.std(axis=0) / np.sqrt(mixedFX.shape[0])

# Smooth performance for WT group
basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=lmbda)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=lmbda)


sns.set(style='ticks')
sns.set_context('poster')
# plt.figure()


plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = (0,0,0), linestyle = "-")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

# Smooth performance for FX group
basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=lmbda)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=lmbda)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = (1,0,0), linestyle ="-")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                  coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


plt.xlim((-2, 35))


#%% Predictive accuracy of partner's position on current choice




def run_cv_logreg_by_rat(data_dict, label="WT"):
    '''
    Function for running multinomial regression in a cross-validated manner
    '''
    results = {}

    for rat_id, df in data_dict.items():
        # Drop rows with NaNs in 'mywell' or 'hiswell'
        df_clean = df.dropna(subset=['mywell', 'hiswell'])

        # Skip if not enough data
        if df_clean.shape[0] < 5:
            results[rat_id] = np.nan
            continue

        # Extract predictor and response
        X = df_clean[['hiswell']].astype(str).values
        y = df_clean['mywell'].astype(str).values

        # One-hot encode the predictor
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)

        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        try:
            for train_idx, test_idx in skf.split(X_encoded, y):
                X_train, X_test = X_encoded[train_idx], X_encoded[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                accuracies.append(acc)

            # Store average accuracy
            results[rat_id] = np.mean(accuracies)
        
        except Exception as e:
            print(f"Skipping {rat_id} due to error: {e}")
            results[rat_id] = np.nan

    # Return as DataFrame with a single row labeled by group
    return pd.DataFrame(results, index=[label])

# =============================================================================

######################### Implementation ######################################

# =============================================================================

wt_accuracy_df = run_cv_logreg_by_rat(wt_data, label='WT')
fx_accuracy_df = run_cv_logreg_by_rat(fx_data, label='FX')

# Combine both
all_accuracy_df = pd.concat([wt_accuracy_df, fx_accuracy_df])

# =============================================================================
# 
################### Plotting and statistical comparison #######################
# 
# =============================================================================


# Melt data for plotting
accuracy_melted = pd.melt(all_accuracy_df.reset_index(), id_vars='index', var_name='rat_id', value_name='accuracy')
accuracy_melted.rename(columns={'index': 'group'}, inplace=True)

# Drop NaNs
accuracy_melted = accuracy_melted.dropna(subset=['accuracy'])

# Plot
plt.figure()
sns.violinplot(data=accuracy_melted, x='group', y='accuracy', cut = 2)
sns.boxplot(data=accuracy_melted, x='group', y='accuracy', showfliers = False)
sns.swarmplot(data=accuracy_melted, x='group', y='accuracy', color='black', alpha=0.6)
# plt.title('Mean Prediction Accuracy (5-fold CV)')
plt.ylabel('Accuracy')
plt.xlabel('Group')
plt.tight_layout()
plt.show()

# Mann–Whitney U test
wt_vals = accuracy_melted[accuracy_melted['group'] == 'WT']['accuracy']
fx_vals = accuracy_melted[accuracy_melted['group'] == 'FX']['accuracy']

stat, pval = mannwhitneyu(wt_vals, fx_vals, alternative='two-sided')
print(f"Mann–Whitney U test: U={stat:.3f}, p-value={pval:.4f}")

# =============================================================================
# 
# 
# ############### Run permutation test ##########################################
# 
# 
# 
# =============================================================================


def run_permutation_test(data_dict, label, n_iter=1000):
    """
    Run permutation test by shuffling 'mywell' and 'hiswell' independently
    and computing accuracy each time.
    
    Returns:
        DataFrame of shape (n_iter, num_rats)
    """
    all_shuffled_accuracies = []

    for i in range(n_iter):
        shuffled_data = {}

        for rat_id, df in data_dict.items():
            shuffled_df = df.copy()
            shuffled_df['mywell'] = np.random.permutation(shuffled_df['mywell'].values)
            shuffled_df['hiswell'] = np.random.permutation(shuffled_df['hiswell'].values)
            shuffled_data[rat_id] = shuffled_df

        acc_df = run_cv_logreg_by_rat(shuffled_data, label=label)
        all_shuffled_accuracies.append(acc_df.mean())

    # Combine into one DataFrame
    return pd.DataFrame(all_shuffled_accuracies)


n_iter = 1000
wt_null = run_permutation_test(wt_data, label='WT', n_iter=n_iter)
fx_null = run_permutation_test(fx_data, label='FX', n_iter=n_iter)


import matplotlib.pyplot as plt

def plot_null_with_observed(null_df, observed_means, label):
    plt.figure(figsize=(7, 5))
    null_means = null_df.mean(axis=1)
    plt.hist(null_means, bins=40, color='gray', alpha=0.6, label='Null distribution')
    plt.axvline(observed_means.mean(), color='red', linestyle='--', linewidth=2, label='Observed mean accuracy')
    plt.title(f'Null distribution vs Observed Accuracy ({label})')
    plt.xlabel('Mean Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Compute empirical p-value
    p_val = (np.sum(null_means >= observed_means.mean()) + 1) / (len(null_means) + 1)
    print(f"{label} empirical p-value: {p_val:.4f}")


# Compute actual mean accuracies from previous step
wt_obs = wt_accuracy_df.mean()
fx_obs = fx_accuracy_df.mean()

# Plot and compare
plot_null_with_observed(wt_null, wt_obs, label='WT')
plot_null_with_observed(fx_null, fx_obs, label='FX')


# Step 1: Mean across iterations (axis=0), one value per rat
wt_null_accuracy_df = pd.DataFrame({
    'rat_id': wt_null.columns,
    'accuracy': wt_null.mean(axis=0).values,
    'group': 'WT'
})

fx_null_accuracy_df = pd.DataFrame({
    'rat_id': fx_null.columns,
    'accuracy': fx_null.mean(axis=0).values,
    'group': 'FX'
})

# Step 2: Combine both
all_null_accuracy_df = pd.concat([wt_null_accuracy_df, fx_null_accuracy_df], ignore_index=True)

# Plot
plt.figure()
sns.violinplot(data=all_null_accuracy_df, x='group', y='accuracy', cut = 2)
sns.boxplot(data=all_null_accuracy_df, x='group', y='accuracy', showfliers = False)
sns.swarmplot(data=all_null_accuracy_df, x='group', y='accuracy', color='black', alpha=0.6)
# plt.title('Mean Prediction Accuracy (5-fold CV)')
plt.ylabel('Accuracy')
plt.xlabel('Group')
plt.tight_layout()
plt.show()

# Mann–Whitney U test
wt_null_vals = all_null_accuracy_df[all_null_accuracy_df['group'] == 'WT']['accuracy']
fx_null_vals = all_null_accuracy_df[all_null_accuracy_df['group'] == 'FX']['accuracy']

stat, pval = mannwhitneyu(wt_null_vals, fx_null_vals, alternative='two-sided')
print(f"Mann–Whitney U test: U={stat:.3f}, p-value={pval:.4f}")
# =============================================================================
# 
# Plot actual and null accuracies on the same plot
# 
# # Step 1: Add a 'type' column
# =============================================================================
accuracy_melted['type'] = 'actual'
all_null_accuracy_df['type'] = 'null'

# Step 2: Concatenate
combined_df = pd.concat([accuracy_melted, all_null_accuracy_df], ignore_index=True)

# Step 3: Plot
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=combined_df,
    x='group',
    y='accuracy',
    hue='type',
    split=False,  # split violins for null vs actual

)

sns.boxplot(data=combined_df, x='group', y='accuracy', hue = 'type', showfliers = False)
sns.swarmplot(data=combined_df, x='group', y='accuracy', hue = 'type', color='black', alpha=0.6)
# plt.title('Actual vs Null Accuracy for WT and FX')
plt.ylabel('Accuracy')
plt.xlabel('Group')
# plt.legend(title='Accuracy Type')
plt.tight_layout()
plt.show()
