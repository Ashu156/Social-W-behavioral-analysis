# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:46:39 2024

@author: shukl
"""

#%% 1. Load libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import smoothfit
import statsmodels.api as sm
import scipy 

#%% 2. Plot for performance

file_path = "C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_perf_matches_over_transitions.ods"
WT = pd.read_excel(file_path, "WT50", engine="odf")
FX = pd.read_excel(file_path, "FX50", engine="odf")
# mixedWT = pd.read_excel(file_path, "Mixed50_WT", engine="odf")
# mixedFX = pd.read_excel(file_path, "Mixed50_FX", engine="odf")

# WT = WT[['AS1_WT1', 'AS2_WT1', 'AS2_WT2']]
# FX = FX[['AS1_FX1', 'AS2_FX1', 'ER1_FX1']]

# WT = WT[['AS1_WT1', 'AS1_WT2', 'AS2_WT1', 'AS2_WT2', 'AS2_WT3', 'AS2_WT4']]
# FX = FX[['AS1_FX1', 'AS1_FX2', 'AS2_FX1', 'AS2_FX2', 'ER1_FX1', 'ER1_FX2']]

sess = 40

# Create a new DataFrame to store the first 30 non-NaN elements for each column
new_WT = pd.DataFrame()
new_FX = pd.DataFrame()
# new_mixedWT = pd.DataFrame()
# new_mixedFX = pd.DataFrame()

# Iterate through each column in the original DataFrame
for column in WT.columns:
    # Drop NaN values and select the first 30 non-NaN values
    tempData = WT[column].dropna().iloc[:sess]
    # Assign the result to the new DataFrame with the same column name
    new_WT[column] = tempData

# Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
new_WT = new_WT.fillna(method='pad', axis=0)


# Iterate through each column in the original DataFrame
for column in FX.columns:
    # Drop NaN values and select the first 30 non-NaN values
    tempData = FX[column].dropna().iloc[:sess]
    # Assign the result to the new DataFrame with the same column name
    new_FX[column] = tempData

# Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
new_FX = new_FX.fillna(method='pad', axis=0)

# # Iterate through each column in the original DataFrame
# for column in mixedWT.columns:
#     # Drop NaN values and select the first 30 non-NaN values
#     tempData = mixedWT[column].dropna().iloc[:sess]
#     # Assign the result to the new DataFrame with the same column name
#     new_mixedWT[column] = tempData

# # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
# new_mixedWT = new_mixedWT.fillna(method='pad', axis=0)

# # Iterate through each column in the original DataFrame
# for column in mixedFX.columns:
#     # Drop NaN values and select the first 30 non-NaN values
#     tempData = mixedFX[column].dropna().iloc[:sess]
#     # Assign the result to the new DataFrame with the same column name
#     new_mixedFX[column] = tempData

# # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
# new_mixedFX = new_mixedFX.fillna(method='pad', axis=0)
  
    
WT = new_WT.transpose()
FX = new_FX.transpose()
# mixedWT = mixedWT.transpose()
# mixedFX = mixedFX.transpose()



WT = WT.iloc[:, :sess]
FX = FX.iloc[:, :sess]
# mixedWT = mixedWT.iloc[:, :sess]
# mixedFX = mixedFX.iloc[:, :sess]

# WT = WT.iloc[:, :2000]
# FX = FX.iloc[:, :2000]

# WTmeanPerf = WT.apply(lambda x: np.mean(x[x.notnull()]), axis=0)
# FXmeanPerf = FX.apply(lambda x: np.mean(x[x.notnull()]), axis=0)
# WTerrPerf = WT.apply(lambda x: np.std(x[x.notnull()]), axis=0) / np.sqrt(WT.shape[0])
# FXerrPerf = FX.apply(lambda x: np.std(x[x.notnull()]), axis=0) / np.sqrt(FX.shape[0])

# WT = WT.melt()
# FX = FX.melt()

# WTecdf = sm.distributions.ECDF(WT['value'].dropna())
# FXecdf = sm.distributions.ECDF(FX['value'].dropna())

# plt.plot(WTecdf.x, WTecdf.y, color = (0,0,0),   linestyle='-')
# plt.plot(FXecdf.x, FXecdf.y, color = (1,0,0), linestyle='-')

# WT = WT.fillna(0)
# FX = FX.fillna(0)

lmbda = 5.0e-1

WTmeanPerf = WT.mean(axis=0)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])

FXmeanPerf = FX.mean(axis=0)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])

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

# # Smooth performance for mixed WT group
# basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,MixedWTmeanPerf.shape[0], 1), MixedWTmeanPerf, 0, MixedWTmeanPerf.shape[0], 1000, degree=1, lmbda=lmbda)
# basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,MixedWTerrPerf.shape[0], 1), MixedWTerrPerf, 0, MixedWTerrPerf.shape[0], 1000, degree=1, lmbda=lmbda)



# plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'blue', linestyle ="-")

# # Plot the shaded region representing the smoothed SEM
# plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
#                   coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

# # Smooth performance for mixed FX group
# basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,MixedFXmeanPerf.shape[0], 1), MixedFXmeanPerf, 0, MixedFXmeanPerf.shape[0], 1000, degree=1, lmbda=lmbda)
# basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,MixedFXerrPerf.shape[0], 1), MixedFXerrPerf, 0, MixedFXerrPerf.shape[0], 1000, degree=1, lmbda=lmbda)


# plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'pink', linestyle ="-")

# # Plot the shaded region representing the smoothed SEM
# plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
#                     coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

# plt.ylim((10, 85))
# plt.xlim((-2, 35))

#%% Plot performance for shuffled choice data

WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_matches_over_transitions.ods", "WT50_shuffled", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_matches_over_transitions.ods", "FX50_shuffled", engine="odf")


    
WT = WT.transpose()
FX = FX.transpose()

WT = WT*2
FX = FX*2

WT = WT.iloc[:, :40]
FX = FX.iloc[:, :40]

# WT = WT.fillna(0)
# FX = FX.fillna(0)


WTmeanPerf = WT.mean(axis=0)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])

FXmeanPerf = FX.mean(axis=0)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'gray', linestyle = "-")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'pink', linestyle ="-")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


#%% 2. Plot for 100% contingencies

WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_matches_over_transitions.ods", "WT100", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_matches_over_transitions.ods", "FX100", engine="odf")

# WT = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_pMatch_WT_opqCont.csv")
# FX = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_pMatch_FX_opqCont.csv")


# WTdata = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_socialW_perf_psytrack.ods", sheet_name =  "WT50", engine = "odf")
# FXdata =  pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_socialW_perf_psytrack.ods", sheet_name =  "FX50", engine = "odf")
fWT = []
fFX = []

for i in range(WT.shape[0]):
    x = WT.iloc[i, :]
    x = x[x.notna()]
    
    y = FX.iloc[i, :]
    y = y[y.notna()]
    
    fWT.append(x)
    fFX.append(y)
    
    
WT = WT.transpose()
FX = FX.transpose()

WT = WT.iloc[:, :20]
FX = FX.iloc[:, :20]

# WT = WT.fillna(0)
# FX = FX.fillna(0)


WTmeanPerf = WT.mean(axis=0)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])

FXmeanPerf = FX.mean(axis=0)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()


plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="WT pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="FX pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)



# Customize the plot
plt.xlabel('Session')
plt.ylabel('# consecutive losses')
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
# plt.ylim((-0.01, 0.5))
# plt.xlim((-1, 10))


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()
plt.plot(WTmeanPerf.index, WTmeanPerf, color = 'k', linestyle = "-", label="WT pairs")
plt.fill_between(WTmeanPerf.index, WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

plt.plot(FXmeanPerf.index, FXmeanPerf, color = 'r', linestyle = "-", label="FX pairs")
plt.fill_between(FXmeanPerf.index, FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)
# plt.ylim((-0.1, 1.1))
# plt.xlim((-1, 10))



#%% Plot change in leadership status between sessions

# Read the data
data = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_leaderFollower_zeroCrossings.ods", "opqCont", engine="odf")

sns.set(style='ticks')
plt.figure()

# Calculate group means
group_means = data.groupby(["Combination", "Genotype"])["Score"].mean().reset_index()

# Define a custom palette mapping for the Genotype values
palette = {0: "black", 1: "red"}

# Plot the boxplot without filling
sns.boxplot(data=data, x="Combination", y="Score", hue="Genotype", dodge=True, palette=palette)

# Plot the stripplot
sns.stripplot(data=data, x="Combination", y="Score", hue="Genotype", dodge=True, alpha=0.5, legend=False)

# Plot horizontal lines for group means
# for idx, row in group_means.iterrows():
#     color = palette.get(row["Genotype"], "black")
#     plt.axhline(y=row["Score"], color=color, linestyle="-", linewidth=5, xmin=idx/len(group_means), xmax=(idx+1)/len(group_means))

# Set y-axis limits
plt.ylim((-0.1, 1.1))



#%% Plot P(matching) as a performance metric

WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_pMatch.ods", "WT50", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_pMatch.ods", "FX50", engine="odf")


WT = WT.iloc[:34]
FX = FX.iloc[:34]

WTmeanPerf = WT.mean(axis = 1)
WTerrPerf = WT.std(axis = 1) / np.sqrt(WT.shape[1])


FXmeanPerf = FX.mean(axis = 1)
FXerrPerf = FX.std(axis = 1) / np.sqrt(FX.shape[1])

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()


plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'k',linestyle = "-")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)

plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'r',linestyle = "-")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)




# Customize the plot
plt.xlabel('Session#')
plt.ylabel('P(match)')
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((-0.01, 1.05))
plt.xlim((-1, 37))


plt.figure()
plt.plot(WT.index, WTmeanPerf, "-", label="WT pairs")
plt.fill_between(WT.index, WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

plt.plot(FX.index, FXmeanPerf, "-", label="FX pairs")
plt.fill_between(FX.index, FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)

#%% Plot inter-match intervals


WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_IMI.ods", "WT50_2", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_IMI.ods", "FX50_2", engine="odf")


WT = WT.iloc[:, :34]
FX = FX.iloc[:, :34]


# Remove outliers ..............................................

median = WT.median() # median
mad =  (np.abs(WT - median)).median() # median absolute deviation
modified_z_scores = 0.6745 * (WT - median) / mad
outliers = np.abs(modified_z_scores) > 3.5
WT = WT.mask(outliers, np.nan)

median = FX.median() # median
mad =  (np.abs(FX - median)).median() # median absolute deviation
modified_z_scores = 0.6745 * (FX - median) / mad
outliers = np.abs(modified_z_scores) > 3.5
FX = FX.mask(outliers, np.nan)

WTmeanPerf = WT.mean(axis=1)
WTerrPerf = WT.std(axis=1) / np.sqrt(WT.shape[1])

FXmeanPerf = FX.mean(axis=1)
FXerrPerf = FX.std(axis=1) / np.sqrt(FX.shape[1])

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()

plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="WT pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="FX pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)





# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('Inter-match interval (s)', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
# plt.ylim((0, 1.05))
# plt.xlim((-1, 33))


# sns.set(style='ticks')
# plt.figure()

# plt.plot(range(WT.shape[1]), WTmeanPerf)
# plt.fill_between(range(WT.shape[1]), WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

# plt.plot(range(FX.shape[1]), FXmeanPerf)
# plt.fill_between(range(FX.shape[1]), FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)

# # Customize the plot
# plt.xlabel('Session#', fontsize = 14)
# plt.ylabel('Inter-match interval (s)', fontsize = 14)

#%% Plot inter-reward intervals

# window = 5

WT = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/probabilisticW_optimal_triplets_ratio.ods", "WT", engine="odf")
FX = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/probabilisticW_optimal_triplets_ratio.ods", "FX", engine="odf")

WT = WT.transpose()
FX = FX.transpose()

WT = WT.iloc[:, :10]
FX = FX.iloc[:, :10]



WTmeanPerf = WT.mean(axis=0)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])
# WTsmoothErr =  savgol_filter(WTerrPerf, window_size, order, mode = 'nearest')

FXmeanPerf = FX.mean(axis=0)
# FXsmoothPerf = savgol_filter(FXmeanPerf, window_size, order)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])
# FXsmoothErr =  savgol_filter(FXerrPerf, window_size, order, mode = 'nearest')

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
plt.figure()

plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'black', linestyle = "-", label="WT pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'red', linestyle = "-", label="FX pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)






# Customize the plot
# plt.xlabel('Session#', fontsize = 14)
# plt.ylabel('Inter-reward interval (s)', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
# plt.ylim((0, 1.05))
# plt.xlim((-1, 33))


# sns.set(style='ticks')
# plt.figure()

# plt.plot(range(WT.shape[1]), WTmeanPerf)
# plt.fill_between(range(WT.shape[1]), WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

# plt.plot(range(FX.shape[1]), FXmeanPerf)
# plt.fill_between(range(FX.shape[1]), FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)

# # Customize the plot
# plt.xlabel('Session#', fontsize = 14)
# plt.ylabel('Inter-reward interval (s)', fontsize = 14)

#%% Plot lag between matched pokes

WT = pd.read_excel("C:/Users/shukl\OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_lag_between_matched_pokes.ods", "WT100", engine="odf")
FX = pd.read_excel("C:/Users/shukl\OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_lag_between_matched_pokes.ods", "FX100", engine="odf")


WT = WT.iloc[:, :21]
FX = FX.iloc[:, :21]

WTmeanPerf = WT.mean(axis=0)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])


FXmeanPerf = FX.mean(axis=0)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()

plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'k', linestyle = "-")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'r', linestyle = "-")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)




# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('Lag between matched pokes (s)', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
# plt.ylim((0, 1.05))
# plt.xticks(np.arange(0,33,5))
# plt.xlim((-1, 25))

sns.set(style='ticks')
plt.figure()

plt.plot(range(WT.shape[1]), WTmeanPerf)
plt.fill_between(range(WT.shape[1]), WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

plt.plot(range(FX.shape[1]), FXmeanPerf)
plt.fill_between(range(FX.shape[1]), FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)

# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('Lag between matched pokes (s)', fontsize = 14)

#%% Plot P(match|reward)

# window = 5

WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_pMatchAfterReward.ods", "SocialW_50_WT", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_pMatchAfterReward.ods", "SocialW_50_FX", engine="odf")

# def smooth(y, box_pts):
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth

WT = WT.iloc[:, :34]
FX = FX.iloc[:, :34]

WTmeanPerf = WT.mean(axis=0)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])
# WTsmoothErr =  savgol_filter(WTerrPerf, window_size, order, mode = 'nearest')

FXmeanPerf = FX.mean(axis=0)
# FXsmoothPerf = savgol_filter(FXmeanPerf, window_size, order)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])
# FXsmoothErr =  savgol_filter(FXerrPerf, window_size, order, mode = 'nearest')

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()


plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'k', linestyle = '-')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]],color = 'r', linestyle = '-')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)



# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('P(match|reward)', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((-0.01, 1.05))
# plt.xlim((-1, 10))

plt.figure()
plt.plot(WTmeanPerf, "-", label="FX pairs")
plt.fill_between(range(WT.shape[1]), WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

plt.plot(FXmeanPerf, "-", label="FX pairs")
plt.fill_between(range(FX.shape[1]), FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)
plt.ylim((-0.01, 1.05))
# plt.xlim((-1, 10))




#%% Plot P(!match|reward) (50%)

window = 5

WT = 1 - WT
FX = 1 - FX

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



WTmeanPerf = WT.mean(axis=0)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])
# WTsmoothErr =  savgol_filter(WTerrPerf, window_size, order, mode = 'nearest')

FXmeanPerf = FX.mean(axis=0)
# FXsmoothPerf = savgol_filter(FXmeanPerf, window_size, order)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])
# FXsmoothErr =  savgol_filter(FXerrPerf, window_size, order, mode = 'nearest')

sns.set(style='ticks')
plt.figure(figsize=(10, 6), dpi = 300)

plt.plot(WTmeanPerf,'o', color = 'black')
# Plot the smoothed mean
sns.lineplot(data = smooth(WTmeanPerf,window), label = 'WT pairs')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(WTmeanPerf.index, smooth(WTmeanPerf, window) - smooth(WTerrPerf,window),
                 smooth(WTmeanPerf,window) +smooth(WTerrPerf,window), alpha=0.2)


plt.plot(FXmeanPerf,'o', color = 'red')

# Plot the smoothed mean
sns.lineplot(data = smooth(FXmeanPerf, window), label = 'FX pairs')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(FXmeanPerf.index, smooth(FXmeanPerf,window) - smooth(FXerrPerf,window),
                  smooth(FXmeanPerf,window) + smooth(FXerrPerf,window), alpha=0.2)





# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('P(!match|reward)', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((0, 1.05))
plt.xlim((-1, 37))


#%% Plot P(match|!reward)



WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_pMatchAfterNoReward.ods", "SocialW_50_WT", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_pMatchAfterNoReward.ods", "SocialW_50_FX", engine="odf")

WT = WT.iloc[:, :34]
FX = FX.iloc[:, :34]

WTmeanPerf = WT.mean(axis=0)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])
# WTsmoothErr =  savgol_filter(WTerrPerf, window_size, order, mode = 'nearest')

FXmeanPerf = FX.mean(axis=0)
# FXsmoothPerf = savgol_filter(FXmeanPerf, window_size, order)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])
# FXsmoothErr =  savgol_filter(FXerrPerf, window_size, order, mode = 'nearest')



basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'k', linestyle = '-')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]],color = 'r', linestyle = '-')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


# Customize the plot
plt.xlabel('Session#', fontsize = 16)
plt.ylabel('P(match|!reward)', fontsize = 16)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((0, 1.05))
plt.xlim((-1, 37))

#%%  Plot P(!match|!reward)

WT = 1 - WT
FX = 1 - FX

WTmeanPerf = WTmeanPerf
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])
# WTsmoothErr =  savgol_filter(WTerrPerf, window_size, order, mode = 'nearest')

FXmeanPerf = FX.mean(axis=0)
# FXsmoothPerf = savgol_filter(FXmeanPerf, window_size, order)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])
# FXsmoothErr =  savgol_filter(FXerrPerf, window_size, order, mode = 'nearest')

sns.set(style='ticks')
plt.figure(figsize=(10, 6), dpi = 300)

plt.plot(WTmeanPerf,'o', color = 'black')
# Plot the smoothed mean
sns.lineplot(data = smooth(WTmeanPerf,window), label = 'WT pairs')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(WTmeanPerf.index, smooth(WTmeanPerf, window) - smooth(WTerrPerf,window),
                 smooth(WTmeanPerf,window) +smooth(WTerrPerf,window), alpha=0.2)


plt.plot(FXmeanPerf,'o', color = 'red')

# Plot the smoothed mean
sns.lineplot(data = smooth(FXmeanPerf, window), label = 'FX pairs')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(FXmeanPerf.index, smooth(FXmeanPerf,window) - smooth(FXerrPerf,window),
                  smooth(FXmeanPerf,window) + smooth(FXerrPerf,window), alpha=0.2)





# Customize the plot
plt.xlabel('Session#', fontsize = 16)
plt.ylabel('P(!match|!reward)', fontsize = 16)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((0, 1.05))
# plt.xlim((-1, 37))


#%% Plot fraction unmatched trials (50%)



WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_fraction_unrewarded_matches.ods", "SocialW_50_WT", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_fraction_unrewarded_matches.ods", "SocialW_50_FX", engine="odf")

WT = WT.iloc[:,:34]
FX = FX.iloc[:, :34]


WTmeanPerf = WT.mean(axis=0)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])
# WTsmoothErr =  savgol_filter(WTerrPerf, window_size, order, mode = 'nearest')

FXmeanPerf = FX.mean(axis=0)
# FXsmoothPerf = savgol_filter(FXmeanPerf, window_size, order)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])
# FXsmoothErr =  savgol_filter(FXerrPerf, window_size, order, mode = 'nearest')


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()


plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'k', linestyle = '-')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

# plt.plot(FXmeanPerf,'o', color = 'red')
# basis, coeffs = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
# plt.plot(np.arange(0,FXmeanPerf.shape[0]), FXmeanPerf, "-", label="FXmeanPerf")
# plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], "-", label="smooth fit")

# Plot the smoothed mean
# sns.lineplot(data = smooth(FXmeanPerf, 5), label = 'FX pairs')

# Plot the shaded region representing the smoothed SEM
# plt.fill_between(FXmeanPerf.index, smooth(FXmeanPerf, window) - smooth(FXerrPerf,5),
#                  smooth(FXmeanPerf, window) + smooth(FXerrPerf,5), alpha=0.2)

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'r', linestyle = '-')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)



# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('P(match|reward)', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((-0.01, 1.05))
# plt.xlim((-1, 10))

plt.figure()
plt.plot(WTmeanPerf, "-", label="FX pairs")
plt.fill_between(range(WT.shape[1]), WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

plt.plot(FXmeanPerf, "-", label="FX pairs")
plt.fill_between(range(FX.shape[1]), FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)
plt.ylim((-0.01, 1.05))
# plt.xlim((-1, 10))



#%% Plot BIC for different behavioral strategies

WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\bic_all_50.ods", "WT", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\bic_all_50.ods", "FX", engine="odf")

df_WT = pd.melt(WT, var_name='Variable', value_name='Value')
# Convert 'Value' column to numeric
# df_WT['Value'] = pd.to_numeric(df_WT['Value'], errors='coerce')

df_FX = pd.melt(FX, var_name='Variable', value_name='Value')

# WT['Genotype'] = 'WT'
# FX['Genotype'] = 'FX'

# cdf = pd.concat([WT, FX])
# mdf = pd.melt(cdf, var_name='Variable', value_name='Value')



sns.set(style='ticks')
plt.figure()

# Create a boxplot
sns.boxplot(x='Variable', y='Value', data=df_WT, color='0.3',  showfliers=False)
# sns.boxplot(x='Variable', y='Value',  data=mdf, color='0.3',  showfliers=False)
sns.stripplot(x = 'Variable', y = 'Value', data = df_WT, color = 'black', alpha = 0.7)
plt.axhline(0, color='green', linestyle='--', linewidth=2)
plt.ylabel('BIC relative to Q_RPE')
# plt.ylim((-100, 900))


# sns.stripplot(data = df_WT, x='Variable', y='Value', jitter=True, color='black', alpha = 0.7)

# Plot the mean as a horizontal bar
# mean_values = df_WT.mean()
# for idx, value in enumerate(mean_values):
#     plt.axhline(y=value, xmin=idx - 0.2, xmax=idx + 0.2, color='black', linestyle='-', linewidth=2)

# plt.show()

sns.set(style='ticks')
plt.figure()

# Create a boxplot
sns.boxplot(x='Variable', y='Value', data=df_FX, color='red', showfliers=False)
sns.stripplot(x = 'Variable', y = 'Value', data = df_FX, color = 'red', alpha = 0.7)
plt.axhline(0, color='green', linestyle='--', linewidth=2)
plt.ylabel('BIC relative to Q_RPE')
# plt.ylim((-100, 900))

#%% Plot stereotypic matches proportions (100%)

# window = 5

WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_stereotypicMatches.ods", "SocialW_100_WT", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_stereotypicMatches.ods", "SocialW_100_FX", engine="odf")


WT = WT.iloc[:,:26]
FX = FX.iloc[:,:26]


WTmeanPerf = WT.mean(axis=0)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])
# WTsmoothErr =  savgol_filter(WTerrPerf, window_size, order, mode = 'nearest')

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



sns.set(style='ticks')
plt.figure()

# plt.plot(WTmeanPerf,'o', color = 'black')
# Plot the smoothed mean
# sns.lineplot(data = smooth(WTmeanPerf,window), label = 'WT pairs')
plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="WT pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

FXmeanPerf = FX.mean(axis=0)
# FXsmoothPerf = savgol_filter(FXmeanPerf, window_size, order)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])
# FXsmoothErr =  savgol_filter(FXerrPerf, window_size, order, mode = 'nearest')

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



# plt.plot(FXmeanPerf,'o', color = 'red')

# Plot the smoothed mean
# sns.lineplot(data = smooth(FXmeanPerf, window), label = 'FX pairs')

plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="FX pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)






# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('Fraction stereotypic matches', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((-0.01, 1.05))
# plt.xticks(np.arange(0, 37, 5)) 
# plt.xlim((-1, 40))


sns.set(style='ticks')
plt.figure()

plt.plot(range(WT.shape[1]), WTmeanPerf)
plt.fill_between(range(WT.shape[1]), WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)


plt.plot(range(FX.shape[1]), FXmeanPerf)
plt.fill_between(range(FX.shape[1]), FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)



# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('Fraction stereotypic matches', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((-0.01, 1.05))

#%% Plot stereotypic matches proportions (50%)

# window = 5

WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_stereotypicMatches.ods", "SocialW_50_WT", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_stereotypicMatches.ods", "SocialW_50_FX", engine="odf")


WT = WT.iloc[:,:36]
FX = FX.iloc[:,:36]


WTmeanPerf = WT.mean(axis=0)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])
# WTsmoothErr =  savgol_filter(WTerrPerf, window_size, order, mode = 'nearest')

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



sns.set(style='ticks')
plt.figure()

# plt.plot(WTmeanPerf,'o', color = 'black')
# Plot the smoothed mean
# sns.lineplot(data = smooth(WTmeanPerf,window), label = 'WT pairs')
plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="WT pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

FXmeanPerf = FX.mean(axis=0)
# FXsmoothPerf = savgol_filter(FXmeanPerf, window_size, order)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])
# FXsmoothErr =  savgol_filter(FXerrPerf, window_size, order, mode = 'nearest')

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



# plt.plot(FXmeanPerf,'o', color = 'red')

# Plot the smoothed mean
# sns.lineplot(data = smooth(FXmeanPerf, window), label = 'FX pairs')

plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="FX pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)






# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('Fraction stereotypic matches', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((-0.01, 1.05))
# plt.xticks(np.arange(0, 37, 5)) 
# plt.xlim((-1, 40))


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()

plt.plot(range(WT.shape[1]), WTmeanPerf)
plt.fill_between(range(WT.shape[1]), WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)


plt.plot(range(FX.shape[1]), FXmeanPerf)
plt.fill_between(range(FX.shape[1]), FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)



# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('Fraction stereotypic matches', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((-0.01, 1.05))


########## Plot individually for each rat ############

sns.set(style='ticks')
sns.set_context('poster')
plt.figure(figsize=(10, 6), dpi = 300)


# Clip values to be between 0 and 1
FX = np.clip(FX, 0, 1)

# Calculate the total number of rows
num_rows = FX.shape[0]

for i in range(num_rows):
    data = FX.iloc[i]
    
    # Find indices where values are not equal to 0
    non_nan_indices = np.where(~np.isnan(data))[0]
    
    # Use only the non-zero values and their corresponding indices for plotting
    non_nan_data = data.iloc[non_nan_indices]
    non_nan_indices = np.arange(len(non_nan_data))
    
    basis, coeffs = smoothfit.fit1d(non_nan_indices, non_nan_data, 0, len(non_nan_data), 1000, degree=1, lmbda=5.0e-1)
    
    plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], "-", label=f"Row {i + 1}")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Customize the plot
plt.xlabel('Session#', fontsize = 16)
plt.ylabel('Fraction stereotypic matches', fontsize = 16)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((0, 1.0))
# plt.xticks(np.arange(0, 37, 5)) 
# plt.xlim((-1, 40))

    
########### Plot for each pair separately ############



sns.set(style='ticks')
sns.set_context('poster')


# Define the number of rows to be plotted in each figure
rows_per_figure = 2

# Calculate the total number of figures needed
num_figures = int(np.ceil(FX.shape[0] / rows_per_figure))

for figure_index in range(num_figures):
    plt.figure(figsize=(10, 6), dpi=300)

    # Calculate the range of rows to plot for the current figure
    start_row = figure_index * rows_per_figure
    end_row = min((figure_index + 1) * rows_per_figure, WT.shape[0])

    for i in range(start_row, end_row):
        data = FX.iloc[i]
        # Find indices where values are not equal to 0
        non_nan_indices = np.where(~np.isnan(data))[0]
        
        # Use only the non-zero values and their corresponding indices for plotting
        non_nan_data = data.iloc[non_nan_indices]
        non_nan_indices = np.arange(len(non_nan_data))
        
        basis, coeffs = smoothfit.fit1d(non_nan_indices, non_nan_data, 0, len(non_nan_data), 1000, degree=1, lmbda=5.0e-1)
        
        plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], "-", label=f"Row {i + 1}")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Figure {figure_index + 1}")
    plt.show()
    
    # Customize the plot
    plt.xlabel('Session#', fontsize = 16)
    plt.ylabel('Fraction stereotypic matches', fontsize = 16)
    # plt.title('Normalized transitions (all pairs)', fontsize = 16)
    # plt.ylim((0, 1.05))
  

#%% Plot the regressor coefficients of fit logistic GLM

WTc = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_choice_fit_GLM_10steps_WT50.csv")
FXc = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_choice_fit_GLM_10steps_FX50.csv")

WTr = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_reward_fit_GLM_10steps_WT50.csv")
FXr = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_reward_fit_GLM_10steps_FX50.csv")

WTrc = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_choiceXreward_fit_GLM_10steps_WT50.csv")
FXrc = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_choiceXreward_fit_GLM_10steps_FX50.csv")


############ For choice ###########################

plt.figure()
sns.set(style='ticks')

WTcMean = WTc.mean(axis=1)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTcErr = WTc.std(axis=1) / np.sqrt(WTc.shape[1])

plt.plot(WTcMean)
plt.fill_between(WTcMean.index, WTcMean - WTcErr, WTcMean + WTcErr, alpha = 0.2)

FXcMean = FXc.mean(axis=1)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
FXcErr = FXc.std(axis=1) / np.sqrt(FXc.shape[1])

plt.plot(FXcMean)
plt.fill_between(FXcMean.index, FXcMean - FXcErr, FXcMean + FXcErr, alpha = 0.2)

plt.ylabel(r'$\beta$ weights')
plt.xlabel('Lags')
plt.xlim((0, 2))

############# For reward #####################

plt.figure()

sns.set(style='ticks')

WTrMean = WTr.mean(axis=1)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTrErr = WTr.std(axis=1) / np.sqrt(WTr.shape[1])

plt.plot(WTrMean)
plt.fill_between(WTrMean.index, WTrMean - WTrErr, WTrMean + WTrErr, alpha = 0.2)

FXrMean = FXr.mean(axis=1)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
FXrErr = FXr.std(axis=1) / np.sqrt(FXr.shape[1])

plt.plot(FXrMean)
plt.fill_between(FXrMean.index, FXrMean - FXrErr, FXrMean + FXrErr, alpha = 0.2)

plt.ylabel(r'$\beta$ weights')
plt.xlabel('Lags')
plt.xlim((0, 2))


############# For choice X reward #####################

plt.figure()

sns.set(style='ticks')

WTrcMean = WTrc.mean(axis=1)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTrcErr = WTrc.std(axis=1) / np.sqrt(WTrc.shape[1])

plt.plot(WTrcMean)
plt.fill_between(WTrcMean.index, WTrcMean - WTrcErr, WTrcMean + WTrcErr, alpha = 0.2)

FXrcMean = FXrc.mean(axis=1)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
FXrcErr = FXrc.std(axis=1) / np.sqrt(FXrc.shape[1])

plt.plot(FXrcMean)
plt.fill_between(FXrcMean.index, FXrcMean - FXrcErr, FXrcMean + FXrcErr, alpha = 0.2)

plt.ylabel(r'$\beta$ weights')
plt.xlabel('Lags')
plt.xlim((0, 2))

#%% Plot correlation between triplet counts of rats in a pair

WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_corr_tripletCounts.ods", "WT_50",engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_corr_tripletCounts.ods", "FX_50", engine="odf")


WT = WT.iloc[:,:34]
FX = FX.iloc[:,:34]


WTmeanPerf = WT.mean(axis=0)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])
# WTsmoothErr =  savgol_filter(WTerrPerf, window_size, order, mode = 'nearest')

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



sns.set(style='ticks')
plt.figure()

# plt.plot(WTmeanPerf,'o', color = 'black')
# Plot the smoothed mean
# sns.lineplot(data = smooth(WTmeanPerf,window), label = 'WT pairs')
plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="WT pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

FXmeanPerf = FX.mean(axis=0)
# FXsmoothPerf = savgol_filter(FXmeanPerf, window_size, order)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])
# FXsmoothErr =  savgol_filter(FXerrPerf, window_size, order, mode = 'nearest')

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



# plt.plot(FXmeanPerf,'o', color = 'red')

# Plot the smoothed mean
# sns.lineplot(data = smooth(FXmeanPerf, window), label = 'FX pairs')

plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="FX pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)






# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('Correlation', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((-1.05, 1.05))
# plt.xticks(np.arange(0, 37, 5)) 
# plt.xlim((-1, 10))

sns.set(style='ticks')
plt.figure()
plt.plot(WTmeanPerf, "-", label="FX pairs")
plt.fill_between(range(WT.shape[1]), WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

plt.plot(FXmeanPerf, "-", label="FX pairs")
plt.fill_between(range(FX.shape[1]), FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)
plt.ylim((-1.05, 1.05))


# for i in range(0, 4):
#     data = WT.iloc[i]
#     # Find indices where values are not equal to 0
#     non_nan_indices = np.where(~np.isnan(data))[0]
    
#     # Use only the non-zero values and their corresponding indices for plotting
#     non_nan_data = data.iloc[non_nan_indices]
#     non_nan_indices = np.arange(len(non_nan_data))
    
#     basis, coeffs = smoothfit.fit1d(non_nan_indices, non_nan_data, 0, len(non_nan_data), 1000, degree=1, lmbda=5.0e-1)
    
#     plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], "-", label=f"Row {i + 1}")

# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title(f"Figure {figure_index + 1}")
# plt.show()

# # Customize the plot
# plt.xlabel('Session#', fontsize = 16)
# plt.ylabel('Fraction stereotypic matches', fontsize = 16)

#%% Plot PsyTtrack (dynamical GLM) weights

WTdata = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_psytrack_weights.ods", sheet_name = "WT_50_peer_side_weights", engine = "odf")
FXdata = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_psytrack_weights.ods", sheet_name = "WT_50_other_side_weights", engine = "odf")



WTdata = WTdata.iloc[:8, :1000]
FXdata = FXdata.iloc[:8, :1000]



WTmeanPerf = WTdata.mean(axis=0).to_numpy()
FXmeanPerf = FXdata.mean(axis=0).to_numpy()

WTerrPerf = (WTdata.std(axis=0) / np.sqrt(WTdata.shape[0])).to_numpy()
FXerrPerf = (FXdata.std(axis=0) / np.sqrt(FXdata.shape[0])).to_numpy()

# basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
# basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
# plt.figure()


# plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="FX pairs")

# # Plot the shaded region representing the smoothed SEM
# plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
#                  coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

# basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
# basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)





# plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="FX pairs")

# # Plot the shaded region representing the smoothed SEM
# plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
#                  coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)
# plt.figure()
x = np.linspace(0, len(WTmeanPerf), len(WTmeanPerf))
plt.plot(x, WTmeanPerf)
plt.fill_between(x, WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

X = np.linspace(0, len(FXmeanPerf), len(FXmeanPerf))
plt.plot(X, FXmeanPerf)
plt.fill_between(X, FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)

#%% Plot mean cross-validated predictive accuracy (from GLM fit)

s = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_GLM_cv_mean_predicted_accuracy.ods", sheet_name = "FX_50_stim", engine = "odf")
s = pd.melt(s)
s['type'] = 's'


c = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_GLM_cv_mean_predicted_accuracy.ods", sheet_name = "FX_50_cLags", engine = "odf")
c = pd.melt(c)
c['type'] = 'c'

r = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_GLM_cv_mean_predicted_accuracy.ods", sheet_name = "FX_50_rLags", engine = "odf")
r = pd.melt(r)
r['type'] = 'r'

cr = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_GLM_cv_mean_predicted_accuracy.ods", sheet_name = "FX_50_crLags", engine = "odf")
cr = pd.melt(cr)
cr['type'] = 'cr'

df_merged = pd.concat([c, r, cr, s], ignore_index = True)

sns.set(style = 'ticks')
sns.set_context('poster')
plt.figure()
# sns.barplot(data = data, x = 'variable', y = 'value' )
# sns.stripplot(data = data, x = 'variable', y = 'value', jitter = 0)

# sns.lineplot(data = df_merged, x = "variable", y = "value", hue= 'type', marker = 'o',  err_style = 'bars', errorbar = 'se')
# sns.stripplot(data = df_merged, x = 'variable', y = 'value', hue = 'type', jitter = 0)


# plt.figure()
sns.barplot(data = df_merged, x = 'variable', y = 'value', hue = 'type')
sns.stripplot(data = df_merged, x = 'variable', y = 'value', hue = 'type')



plt.ylim((-1.5,1.5))
# plt.axhline(0, color = 'k')
# plt.ylabel('Mean predictive accuracy')
# plt.legend(('choice', 'reward', 'choice X reward'), loc = 'best')
plt.ylabel('Weights')

#%% Plot optimal choice sequences as a function of training

WT = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_optimal_triplets_proportions.ods", sheet_name = "WT50", engine="odf")
FX = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_optimal_triplets_proportions.ods", sheet_name = "FX50", engine="odf")
mixed_WT = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_optimal_triplets_proportions.ods", sheet_name = "Mixed_WT50", engine="odf").T
mixed_FX = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_optimal_triplets_proportions.ods", sheet_name = "Mixed_FX50", engine="odf").T

WT = WT.iloc[:, :34]
FX = FX.iloc[:, :34]
mixed_WT = mixed_WT.iloc[:, :34]
mixed_FX = mixed_FX.iloc[:, :34]

WT = WT.fillna(0)
FX = FX.fillna(0)
mixed_WT = mixed_WT.fillna(0)
mixed_FX = mixed_FX.fillna(0)

WTmeanPerf = WT.mean(axis=0)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])


FXmeanPerf = FX.mean(axis=0)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])

mixed_WTmeanPerf = mixed_WT.mean(axis=0)
mixed_WTerrPerf = mixed_WT.std(axis=0) / np.sqrt(mixed_WT.shape[0])


mixed_FXmeanPerf = mixed_FX.mean(axis=0)
mixed_FXerrPerf = mixed_FX.std(axis=0) / np.sqrt(mixed_FX.shape[0])

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()
plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="WT pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="FX pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,mixed_WTmeanPerf.shape[0], 1), mixed_WTmeanPerf, 0, mixed_WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,mixed_WTerrPerf.shape[0], 1), mixed_WTerrPerf, 0, mixed_WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)

sns.set(style='ticks')
sns.set_context('poster')
plt.figure()

plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="mixed WT pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,mixed_FXmeanPerf.shape[0], 1), mixed_FXmeanPerf, 0, mixed_FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,mixed_FXerrPerf.shape[0], 1), mixed_FXerrPerf, 0, mixed_FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="mixed FX pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

# Customize the plot
plt.xlabel('Session', fontsize = 14)
plt.ylabel('Proportion optimal choice sequences', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((-0.1, 1.1))
# plt.xlim((-1, 10))
plt.legend()


# plt.figure()
# plt.plot(WTmeanPerf.index, WTmeanPerf, "-", label="WT pairs")
# plt.fill_between(WTmeanPerf.index, WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

# plt.plot(FXmeanPerf.index, FXmeanPerf, "-", label="FX pairs")
# plt.fill_between(FXmeanPerf.index, FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)
# plt.ylim((-0.1, 1.1))
# plt.xlim((-1, 10))

#%% 21. Plot predictive accuracy (glmnet)

raw = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_predictive_accuracy.ods", sheet_name = "data50", engine="odf")
# shuffle = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_predictive_accuracy.ods", sheet_name = "shuffle_50", engine="odf")

data = raw[['WT_data', 'FX_data']]
data = pd.melt(data)
data['type'] = 'data'

shuffle = raw[['WT_shuffle', 'FX_shuffle']]
shuffle = pd.melt(shuffle)
shuffle['type'] = 'shuffle'

df_merged = pd.concat([data, shuffle], ignore_index = True)
df_merged['value'] = df_merged['value']*100

sns.set(style='ticks')
sns.set_context('poster')
plt.figure()

sns.boxplot(data = df_merged, x = 'variable', y = 'value', hue = 'type')
sns.stripplot(data = df_merged, x = 'variable', y = 'value', hue = 'type',  s = 10, marker = 'o', dodge = 'True', jitter = 0)

# Customize the plot
plt.ylabel('% Correct predictions')
# plt.ylim((0.0, 100))
plt.axhline(33.33, linestyle = '--', color = 'black')


#%% 22.

WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_predictive_accuracy.ods", sheet_name = "WT_50_data", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_predictive_accuracy.ods", sheet_name = "FX_50_data", engine="odf")


WT = WT.iloc[20:, :5]
FX = FX.iloc[:, :8]

# WT = WT.fillna(0)
# FX = FX.fillna(0)


WTmeanPerf = WT.mean(axis=0)
# WTsmoothPerf = savgol_filter(WTmeanPerf, window_size, order)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])
# WTsmoothErr =  savgol_filter(WTerrPerf, window_size, order, mode = 'nearest')

FXmeanPerf = FX.mean(axis=0)
# FXsmoothPerf = savgol_filter(FXmeanPerf, window_size, order)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])
# FXsmoothErr =  savgol_filter(FXerrPerf, window_size, order, mode = 'nearest')

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()
plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="WT pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)


basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,FXmeanPerf.shape[0], 1), FXmeanPerf, 0, FXmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,FXerrPerf.shape[0], 1), FXerrPerf, 0, FXerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="FX pairs")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                 coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)



# Customize the plot
plt.xlabel('Session', fontsize = 14)
plt.ylabel('% Correct predictions', fontsize = 14)
# plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((-0.1, 1.1))
plt.axhline(0.33, linestyle = '--', color = 'black')
# plt.xlim((-1, 10))


# WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_predictive_accuracy.ods", sheet_name = "WT_50_data", engine="odf")
# FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_predictive_accuracy.ods", sheet_name = "FX_50_data", engine="odf")

# WT = WT.iloc[:, :13]
# FX = FX.iloc[:, :8]


# WTmeanPerf = WT.mean(axis=0)
# WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])

# FXmeanPerf = FX.mean(axis=0)
# FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])


plt.figure()
plt.plot(WTmeanPerf.index, WTmeanPerf, "-", label="WT pairs")
plt.fill_between(WTmeanPerf.index, WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

plt.plot(FXmeanPerf.index, FXmeanPerf, "-", label="FX pairs")
plt.fill_between(FXmeanPerf.index, FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)
plt.ylim((-0.1, 1.1))
plt.axhline(0.33, linestyle = '--', color = 'black')
# plt.xlim((-1, 10))

#%% Plot results from partner-directed attention GLMs

data = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/partner_directed_attention_GLMfits_onlyHighPerformingSessions.ods", "final_data", engine="odf")

data = data.iloc[:, 1:-4]



def filter_df(df, 
                         time_condition=None, 
                         theta_condition=None, 
                         time_col='time', 
                         theta_col='theta'):
    """
    Filters a DataFrame based on conditions for time (string format) and theta columns.
    
    Parameters:
    - df: Input DataFrame
    - time_condition: Lambda function for time comparison (e.g., lambda x: x > 3)
    - theta_condition: Lambda function for theta comparison (e.g., lambda y: y < 90)
    - time_col: Name of the time column (default: 'time')
    - theta_col: Name of the theta column (default: 'theta')
    
    Returns:
    - Filtered DataFrame
    """
    # Create a copy to avoid modifying the original DataFrame
    filtered_df = df.copy()
    
    # Convert time strings to numeric values (e.g., "4s" -> 4)
    # filtered_df['_time_num'] = filtered_df[time_col].str.replace('s', '').astype(int)
    
    # Initialize a boolean mask
    mask = pd.Series([True]*len(filtered_df), index=filtered_df.index)
    
    # Apply time condition if provided
    if time_condition is not None:
        mask &= filtered_df[time_col].apply(time_condition)
    
    # Apply theta condition if provided
    if theta_condition is not None:
        mask &= filtered_df[theta_col].apply(theta_condition)
    
    # Return filtered results and clean up temporary column
    result = filtered_df[mask].drop(columns=['_time_num'], errors='ignore')
    return result


filtered = filter_df(data, 
                    time_condition=lambda x: x == 5, 
                    theta_condition=lambda y: y == 20)

plt.figure(figsize = (3,5))
sns.barplot(filtered.loc[2, ['p_Sub', 'p_Par', 'p_Both']])
plt.ylim((-1.1, 1.6))

plt.figure(figsize = (3,5))
sns.barplot(filtered.loc[3, ['p_Sub', 'p_Par', 'p_Both']])
plt.ylim((-1.1, 1.6))


# Melt the DataFrame
df_melted = filtered.melt(
    id_vars=['group'],  # Keep 'group' as identifier
    value_vars=['p_Sub', 'p_Par', 'p_Both'],  # Columns to melt
    var_name='variable',  # New column for variable names
    value_name='value'  # New column for values
)

plt.figure()

# Create the bar plot
sns.barplot(
    x='variable',  # Variables (p_Sub, p_Par, p_Both)
    y='value',     # Values for each variable
    hue='group',   # Differentiate by genotype (WT/FX)
    data=df_melted
)

# Customize the plot
# plt.title('Comparison of Variables by Genotype')
plt.xlabel('Predictors')
plt.ylabel(r'$\beta$ weights')
plt.legend(title='Genotype', loc='upper right')

plt.show()

#%%

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Define the file path and sheet names
file_path = "C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_perf_matches_over_transitions.ods"  # File path
sheets = ["WT50", "FX50", "Mixed50"]  # Add all sheet names

# Initialize an empty list to store processed data
data_list = []

# Loop through each sheet (category)
for sheet in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet, header=None, engine = "odf")  # No headers in raw data
    
    # Assign unique animal IDs as column indices (starting from 1)
    animal_ids = [f"{sheet}_{i+1}" for i in range(df.shape[1])]

    # Reshape data: Iterate over columns (each column is one animal's performance)
    for animal_id, column in zip(animal_ids, df.columns):
        valid_data = df[column].dropna().reset_index(drop=True)  # Drop NaNs and reset index
        sessions = range(1, len(valid_data) + 1)  # Assign session numbers

        # Create a DataFrame for this animal
        animal_data = pd.DataFrame({
            'animal_id': animal_id,
            'group': sheet,  # Use sheet name as group
            'session': sessions,
            'performance': valid_data.values
        })

        data_list.append(animal_data)

# Combine all processed data into a single DataFrame
final_data = pd.concat(data_list, ignore_index=True)

# Convert types
final_data['animal_id'] = final_data['animal_id'].astype(str)  # Convert to string if needed
final_data['group'] = final_data['group'].astype(str)
final_data['session'] = final_data['session'].astype(int)
final_data['performance'] = final_data['performance'].astype(float)



# Save to CSV
final_data.to_csv("formatted_data.csv", index=False)

import statsmodels.formula.api as smf

# Model formula with interaction
model = smf.mixedlm(
    formula = "performance ~ group * session",  # Fixed effects + interaction
    data = final_data,
    groups = "animal_id"  # Random intercepts for animals
).fit()

# Show results
print(model.summary())

#%%

import smoothfit
import numpy as np
import matplotlib.pyplot as plt

# Generate session values for predictions
sessions = np.arange(final_data["session"].min(), final_data["session"].max() + 1, 1)

# Store smoothed predictions
group_predictions = {}

for group in ["FX100", "Mixed100", "WT100"]:
    # Compute model predictions
    pred = (
        model.params["Intercept"]
        + (model.params.get(f"group[T.{group}]", 0))
        + sessions * (model.params["session"] + model.params.get(f"group[T.{group}]:session", 0))
    )
    
    # Apply smoothfit to smooth the curve
    basis_mean, coeffs_mean = smoothfit.fit1d(
        np.arange(pred.shape[0]), pred, 0, pred.shape[0], 1000, degree=1, lmbda=5.0e-1
    )
    
    # Store smoothed predictions
    group_predictions[group] = (np.linspace(0, pred.shape[0], 1000), basis_mean @ coeffs_mean)

# Plot
plt.figure(figsize=(8, 6))

for group, (x_vals, smoothed_pred) in group_predictions.items():
    plt.plot(x_vals, smoothed_pred, label=group)

plt.xlabel("Session")
plt.ylabel("Predicted Performance (Smoothed)")
plt.title("Smoothed Model Predictions: Performance Over Sessions")
plt.legend(title="Group")
plt.show()

#%% Plot performance on probabilistic foraging task


WT = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/probabilisticW_perf.ods", "WT", engine="odf")
FX = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/probabilisticW_perf.ods", "FX", engine="odf")

    
WT = WT.transpose()
FX = FX.transpose()


WT = WT.iloc[:, :10]
FX = FX.iloc[:, :10]


# WT = WT.iloc[:, :2000]
# FX = FX.iloc[:, :2000]

# WTmeanPerf = WT.apply(lambda x: np.mean(x[x.notnull()]), axis=0)
# FXmeanPerf = FX.apply(lambda x: np.mean(x[x.notnull()]), axis=0)
# WTerrPerf = WT.apply(lambda x: np.std(x[x.notnull()]), axis=0) / np.sqrt(WT.shape[0])
# FXerrPerf = FX.apply(lambda x: np.std(x[x.notnull()]), axis=0) / np.sqrt(FX.shape[0])

# WT = WT.melt()
# FX = FX.melt()

# WTecdf = sm.distributions.ECDF(WT['value'].dropna())
# FXecdf = sm.distributions.ECDF(FX['value'].dropna())

# plt.plot(WTecdf.x, WTecdf.y, color = (0,0,0),   linestyle='-')
# plt.plot(FXecdf.x, FXecdf.y, color = (1,0,0), linestyle='-')

# WT = WT.fillna(0)
# FX = FX.fillna(0)


WTmeanPerf = WT.mean(axis=0)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])

FXmeanPerf = FX.mean(axis=0)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])

lmbda=5.0e-1

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


plt.ylim((30, 65))
plt.axhline(50)
plt.xticks(np.arange(0, 11, step=2))

#%% 2. Linear regression for match rate vs transition rate

from scipy.stats import linregress

sheets = ['WT50', 'FX50']

# Define color mapping
color_map = {"WT50": "black", "FX50": "red"}
text_position_map = {"WT50": (0.05, 0.9), "FX50": (0.7, 0.1)}  # Adjusted positions


plt.figure(figsize = (5, 4))
file_path = "C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_transitions_per_minute.ods"

for sheet in sheets:

    transitions = pd.read_excel(file_path, sheet, engine="odf")
    matches = pd.read_excel(file_path, sheet, engine="odf")
    
    # mixedWT = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_Simpson_diversity_reciprocal.ods", "Mixed50_WT", engine="odf")
    # mixedFX = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_Simpson_diversity_reciprocal.ods", "Mixed50_FX", engine="odf")
    
    sess = 35
    
    # Create a new DataFrame to store the first 30 non-NaN elements for each column
    new_tr = pd.DataFrame()
    new_mtch = pd.DataFrame()
    
    # Iterate through each column in the original DataFrame
    for column in transitions.columns:
        # Drop NaN values and select the first 30 non-NaN values
        tempData = transitions[column].dropna().iloc[:sess]
        # Assign the result to the new DataFrame with the same column name
        new_tr[column] = tempData
    
    # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
    new_tr = new_tr.fillna(method='pad', axis=0)
    
    
    # Iterate through each column in the original DataFrame
    for column in matches.columns:
        # Drop NaN values and select the first 30 non-NaN values
        tempData = matches[column].dropna().iloc[:sess]
        # Assign the result to the new DataFrame with the same column name
        new_mtch[column] = tempData
    
    # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
    new_mtch = new_mtch.fillna(method='pad', axis=0)
    
    
        
    transitions = new_tr.transpose()
    matches = new_mtch.transpose()
    
    
    # Melt both dataframes
    tr_melted = transitions.melt(var_name="Variable", value_name="transition_rate")
    mtch_melted = matches.melt(var_name="Variable", value_name="match_rate")
    
    # Repeat FX rows to match WT
    mtch_melted = mtch_melted.loc[mtch_melted.index.repeat(2)].reset_index(drop=True)
    
    # Concatenate into a single dataframe
    combined_df = pd.concat([tr_melted, mtch_melted["match_rate"]], axis=1)
    
    # Drop NaN values before regression
    combined_df = combined_df.dropna()
    
    # Compute linear regression statistics
    slope, intercept, r_value, p_value, std_err = linregress(combined_df["transition_rate"], combined_df["match_rate"])
    r_squared = r_value**2  # Compute R²
    
    # Plot with sns.regplot()
    
    # Select the correct color for the current sheet
    plot_color = color_map.get(sheet, "black")  # Default to black if sheet name is unexpected
    sns.regplot(data=combined_df, x="transition_rate", y="match_rate", scatter_kws={'s': 30, 'color': plot_color,'alpha': 0.6}, line_kws = {'color': plot_color})
    
    text_x, text_y = text_position_map.get(sheet, (0.05, 0.9))  # Default to (0.05, 0.9)
    # Annotate with R² and p-value at different locations
    plt.text(
        text_x, text_y, f"$R^2$ = {r_squared:.3f}\n$p$-value = {p_value:.3g}",
        transform=plt.gca().transAxes, fontsize=24, bbox=dict(facecolor='white', alpha=0.5)
    )
    
    plt.xlabel("Transition rate")
    plt.ylabel("Match rate")
    plt.xlim((0, 6.5))
    plt.ylim((-0.5, 5.5))
    plt.show()
    
#%% Leadership probabilities after consecutive events

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
import smoothfit



# Settings
WTsheets = ['XFN2', 'XFN4', 'FXM102', 'FXM103', 'FXM105', 'FXM107', 'FXM101', 'FXM110', 'FXM201', 'FXM202']
FXsheets = ['XFN1', 'XFN3', 'FXM108', 'FXM109', 'ER1', 'ER2', 'FXM104', 'FXM106', 'FXM205', 'FXM206']
file_path = "C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_leader_follower_tpm_p1_50.ods"
sess = 34

sns.set(style='ticks')
sns.set_context('poster')

def extract_group_data(sheets):
    group_data = []

    for sheet in sheets:
        df = pd.read_excel(file_path, sheet, engine="odf")
        new_df = pd.DataFrame()

        for column in df.columns:
            tempData = df[column].dropna().iloc[:sess]
            new_df[column] = tempData

        new_df = new_df.fillna(method='pad', axis=0)
        group_data.append(new_df.transpose())

    return group_data

# Get raw data
WT_data = extract_group_data(WTsheets)
FX_data = extract_group_data(FXsheets)
event_labels = WT_data[0].index.tolist()

# Convert to long format for plotting
def make_long_df(data_list, label):
    rows = []
    for rat_idx, df in enumerate(data_list):
        for event in df.index:
            mean_value = df.loc[event].mean()
            rows.append({'Rat': f'{label}_{rat_idx+1}', 'Group': label, 'Event': event, 'Value': mean_value})
    return pd.DataFrame(rows)

WT_long = make_long_df(WT_data, 'WT')
FX_long = make_long_df(FX_data, 'FX')
combined_df = pd.concat([WT_long, FX_long])

# Sort events if needed
combined_df['Event'] = pd.Categorical(combined_df['Event'], categories=event_labels, ordered=True)

colors = {"WT": "gray", "FX": "red"}

# Raincloud plot
plt.figure()
ax = pt.RainCloud(x='Event', y='Value', hue='Group', data=combined_df, palette=colors,
                  bw=.2, width_viol=.6, ax=None, orient='v', move=.2, box_showfliers=False)

plt.ylim((0.0, 1.0))
plt.axhline(0.5, color = 'black', linestyle = '--')
plt.xticks(rotation=45)
plt.ylabel("Probability")
plt.xlabel("Event Pattern")
plt.tight_layout()
plt.show()

###############################################################################
############### Plot mean and sem by sessions #################################
###############################################################################

# Prepare data for plotting
def make_long_df_by_session(data_list, label):
    rows = []
    for rat_idx, df in enumerate(data_list):
        for event in df.index:
            max_sessions = df.shape[1]
            for session_idx in range(min(sess, max_sessions)):
                value = df.iloc[df.index.get_loc(event), session_idx]
                rows.append({
                    'Rat': f'{label}_{rat_idx+1}',
                    'Group': label,
                    'Event': event,
                    'Session': session_idx + 1,
                    'Value': value
                })
    return pd.DataFrame(rows)

WT_long_sess = make_long_df_by_session(WT_data, 'WT')
FX_long_sess = make_long_df_by_session(FX_data, 'FX')
combined_sess_df = pd.concat([WT_long_sess, FX_long_sess])

# Plot

# Get unique events and sessions
unique_events = combined_sess_df['Event'].unique()
sessions = np.sort(combined_sess_df['Session'].unique())
colors = {"black", "red"}

import matplotlib.pyplot as plt

# Create 3-row, 2-column subplot layout
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

# Loop through events and plot in respective subplot
for i, event in enumerate(unique_events):
    ax = axes[i]

    for group in ['WT', 'FX']:
        # Subset data
        subset = combined_sess_df[(combined_sess_df['Group'] == group) & (combined_sess_df['Event'] == event)]

        # Compute mean and sem per session
        means = subset.groupby('Session')['Value'].mean()
        sems = subset.groupby('Session')['Value'].sem()

        # Set smoothing parameters
        lmbda = 5e-1
        x_vals = np.linspace(0, len(sessions)-1, 1000)

        # Fit smooth curves using smoothfit
        basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0, means.shape[0], 1), means,
                                                  0, means.shape[0], 1000, degree=1, lmbda=lmbda)
        basis_err, coeffs_err = smoothfit.fit1d(np.arange(0, sems.shape[0], 1), sems,
                                                0, sems.shape[0], 1000, degree=1, lmbda=lmbda)

        smoothed_mean = coeffs_mean[basis_mean.nodal_dofs[0]]
        smoothed_err = coeffs_err[basis_err.nodal_dofs[0]]
        x_smooth = basis_mean.mesh.p[0]

        # Plot mean and shaded error
        ax.plot(x_smooth, smoothed_mean, label=group)
        ax.fill_between(x_smooth, smoothed_mean - smoothed_err, smoothed_mean + smoothed_err, alpha=0.2)

    ax.set_title(f"Event: {event}")
    ax.axhline(0.5, linestyle='--', color='gray')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2, 35)
    ax.set_xlabel("Session")
    ax.set_ylabel("Probability")
    # ax.legend(title="Group")

# Hide any unused subplots (if less than 6)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#%% Plot violinplots for lineplots

file_path = "C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_entropy.ods"
WT = pd.read_excel(file_path, "WT50", engine="odf")
FX = pd.read_excel(file_path, "FX50", engine="odf")


# WT = WT[['AS1_WT1', 'AS2_WT1', 'AS2_WT2']]
# FX = FX[['AS1_FX1', 'AS2_FX1', 'ER1_FX1']]

# WT = WT[['AS1_WT1', 'AS1_WT2', 'AS2_WT1', 'AS2_WT2', 'AS2_WT3', 'AS2_WT4']]
# FX = FX[['AS1_FX1', 'AS1_FX2', 'AS2_FX1', 'AS2_FX2', 'ER1_FX1', 'ER1_FX2']]

sess = 34

# Create a new DataFrame to store the first 30 non-NaN elements for each column
new_WT = pd.DataFrame()
new_FX = pd.DataFrame()

# Iterate through each column in the original DataFrame
for column in WT.columns:
    # Drop NaN values and select the first 30 non-NaN values
    tempData = WT[column].dropna().iloc[:sess]
    # Assign the result to the new DataFrame with the same column name
    new_WT[column] = tempData

# Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
new_WT = new_WT.fillna(method='pad', axis=0)


# Iterate through each column in the original DataFrame
for column in FX.columns:
    # Drop NaN values and select the first 30 non-NaN values
    tempData = FX[column].dropna().iloc[:sess]
    # Assign the result to the new DataFrame with the same column name
    new_FX[column] = tempData

# Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
new_FX = new_FX.fillna(method='pad', axis=0)




WT = WT.iloc[:sess, :]
FX = FX.iloc[:sess, :]


# WT = new_WT.tail(20)
# FX = new_FX.tail(20)

sns.set(style='ticks')
sns.set_context('poster')

# Get column-wise means
wt_means = WT.mean().values
fx_means = FX.mean().values
pairsWT = WT.columns.tolist()
pairsFX = FX.columns.tolist()

# Create tidy DataFrame
df_plot = pd.DataFrame({
    'Pair': pairsWT + pairsFX,
    'MeanValue': list(wt_means) + list(fx_means),
    'Genotype': ['WT'] * len(pairsWT) + ['FX'] * len(pairsFX)
})

# Plot
plt.figure()
sns.violinplot(data=df_plot, x='Genotype', y='MeanValue', inner='point', cut=2)
# plt.title("Mean Value Distribution per Genotype")
# plt.ylabel("Mean Value")
# plt.xlabel("Genotype")
plt.tight_layout()
# plt.ylim((40, 85))
plt.show()

###############################################################################
######################### Plot first and last n sessions ######################
###############################################################################
n_sess = 9

# Get first and last 20 session means
first20_WT = WT.head(n_sess).mean()
first20_FX = FX.head(n_sess).mean()
last20_WT = WT.tail(n_sess).mean()
last20_FX = FX.tail(n_sess).mean()

# Combine into long-form DataFrame
df_first = pd.DataFrame({
    'Pair': new_WT.columns.tolist() * 2,
    'MeanValue': list(first20_WT) + list(first20_FX),
    'Genotype': ['WT'] * len(first20_WT) + ['FX'] * len(first20_FX),
    'SessionGroup': ['First20'] * (len(first20_WT) + len(first20_FX))
})

df_last = pd.DataFrame({
    'Pair': new_WT.columns.tolist() * 2,
    'MeanValue': list(last20_WT) + list(last20_FX),
    'Genotype': ['WT'] * len(last20_WT) + ['FX'] * len(last20_FX),
    'SessionGroup': ['Last20'] * (len(last20_WT) + len(last20_FX))
})

df_combined = pd.concat([df_first, df_last], ignore_index=True)

# Plot
plt.figure()
ax = sns.violinplot(data=df_combined, x='Genotype', y='MeanValue', hue='SessionGroup',
                    split=False, inner=None, cut=3, linewidth=1)

# Overlay stripplot for individual points
sns.stripplot(data=df_combined, x='Genotype', y='MeanValue', hue='SessionGroup',
              dodge=True, jitter=False, marker='o', alpha=0.7, palette='dark:.3', ax=ax)

# Remove duplicate legends from stripplot
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])

# Compute x positions for connecting lines
genotype_pos = {'WT': -0.2, 'FX': 0.8}  # Slight offsets for line drawing

for pair in new_WT.columns:
    for genotype in ['WT', 'FX']:
        y_first = df_combined[(df_combined['Pair'] == pair) &
                              (df_combined['Genotype'] == genotype) &
                              (df_combined['SessionGroup'] == 'First20')]['MeanValue'].values
        y_last = df_combined[(df_combined['Pair'] == pair) &
                             (df_combined['Genotype'] == genotype) &
                             (df_combined['SessionGroup'] == 'Last20')]['MeanValue'].values

        if len(y_first) and len(y_last):
            x_base = 0 if genotype == 'WT' else 1
            # Get the x positions from dodge offset
            x1 = x_base - 0.15  # First20
            x2 = x_base + 0.15  # Last20
            plt.plot([x1, x2], [y_first[0], y_last[0]], color='gray', alpha=0.6, linewidth=1)

# plt.title("Mean Values Across First and Last 20 Sessions")
plt.tight_layout()
plt.show()

#%% Plot violin plots for performance of individual pairs (compare across groups and conditions)

file_path = "C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_perf_matches_over_transitions.ods"
WT = pd.read_excel(file_path, "WT50", engine="odf")
FX = pd.read_excel(file_path, "FX50", engine="odf")
WTopq = pd.read_excel(file_path, "WTopq", engine="odf")
FXopq = pd.read_excel(file_path, "FXopq", engine="odf")

# WT = WT[['AS1_WT1', 'AS2_WT1', 'AS2_WT2']]
# FX = FX[['AS1_FX1', 'AS2_FX1', 'ER1_FX1']]

# WT = WT[['AS1_WT1', 'AS1_WT2', 'AS2_WT1', 'AS2_WT2', 'AS2_WT3', 'AS2_WT4']]
# FX = FX[['AS1_FX1', 'AS1_FX2', 'AS2_FX1', 'AS2_FX2', 'ER1_FX1', 'ER1_FX2']]

sess = 10
which_sessions = 'last'  # or 'first'

new_WT = pd.DataFrame()
new_FX = pd.DataFrame()

if which_sessions == 'first':
    for column in WT.columns:
        # Drop NaN values and select the first 30 non-NaN values
        tempData = WT[column].dropna().iloc[:sess]
        # Assign the result to the new DataFrame with the same column name
        new_WT[column] = tempData

    # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
    new_WT = new_WT.fillna(method='pad', axis=0)


    # Iterate through each column in the original DataFrame
    for column in FX.columns:
        # Drop NaN values and select the first 30 non-NaN values
        tempData = FX[column].dropna().iloc[:sess]
        # Assign the result to the new DataFrame with the same column name
        new_FX[column] = tempData

    # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
    new_FX = new_FX.fillna(method='pad', axis=0)
    
elif which_sessions == 'last':
    
    for column in WT.columns:
        # Drop NaN values and select the first 30 non-NaN values
        tempData = WT[column].dropna().iloc[-sess:]
        # Assign the result to the new DataFrame with the same column name
        new_WT[column] = tempData.reset_index(drop=True)

    # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
    new_WT = new_WT.fillna(method='pad', axis=0)


    # Iterate through each column in the original DataFrame
    for column in FX.columns:
        # Drop NaN values and select the first 30 non-NaN values
        tempData = FX[column].dropna().iloc[-sess:]
        # Assign the result to the new DataFrame with the same column name
        new_FX[column] = tempData.reset_index(drop=True)

    # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
    new_FX = new_FX.fillna(method='pad', axis=0)


# Transpose for analysis
WT = new_WT.transpose()
FX = new_FX.transpose()

nSess = 10

WT = WT.iloc[:, -nSess:]
FX = FX.iloc[:, -nSess:]

WTmean = WT.mean(axis = 1)
FXmean = FX.mean(axis = 1)

WTmean_df = WTmean.reset_index()
WTmean_df.columns = ['rat', 'value']

FXmean_df = FXmean.reset_index()
FXmean_df.columns = ['rat', 'value']

WTmean_df['condition'] = 'see'
FXmean_df['condition'] = 'see'

WTmean_df['group'] = 'WT'
FXmean_df['group'] = 'FX'

sess = 10
which_sessions = 'first'  # or 'first'

new_WTopq = pd.DataFrame()
new_FXopq = pd.DataFrame()


if which_sessions == 'first':
    for column in WTopq.columns:
        # Drop NaN values and select the first 30 non-NaN values
        tempData = WTopq[column].dropna().iloc[:sess]
        # Assign the result to the new DataFrame with the same column name
        new_WTopq[column] = tempData

    # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
    new_WTopq = new_WTopq.fillna(method='pad', axis=0)


    # Iterate through each column in the original DataFrame
    for column in FXopq.columns:
        # Drop NaN values and select the first 30 non-NaN values
        tempData = FXopq[column].dropna().iloc[:sess]
        # Assign the result to the new DataFrame with the same column name
        new_FXopq[column] = tempData

    # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
    new_FXopq = new_FXopq.fillna(method='pad', axis=0)
    
elif which_sessions == 'last':
    
    for column in WTopq.columns:
        # Drop NaN values and select the first 30 non-NaN values
        tempData = WTopq[column].dropna().iloc[-sess:]
        # Assign the result to the new DataFrame with the same column name
        new_WTopq[column] = tempData.reset_index(drop=True)

    # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
    new_WTopq = new_WTopq.fillna(method='pad', axis=0)


    # Iterate through each column in the original DataFrame
    for column in FXopq.columns:
        # Drop NaN values and select the first 30 non-NaN values
        tempData = FXopq[column].dropna().iloc[-sess:]
        # Assign the result to the new DataFrame with the same column name
        new_FXopq[column] = tempData.reset_index(drop=True)

    # Ensure that the resulting DataFrame has 30 rows by filling NaN if there are fewer than 30 non-NaN values
    new_FXopq = new_FXopq.fillna(method='pad', axis=0)


# Transpose for analysis
WTopq = new_WTopq.transpose()
FXopq = new_FXopq.transpose()

nSess = 8

WTopq = WTopq.iloc[:, -nSess:]
FXopq = FXopq.iloc[:, -nSess:]

WTopqmean = WTopq.mean(axis = 1)
FXopqmean = FXopq.mean(axis = 1)

WTopqmean_df = WTopqmean.reset_index()
WTopqmean_df.columns = ['rat', 'value']

FXopqmean_df = FXopqmean.reset_index()
FXopqmean_df.columns = ['rat', 'value']

WTopqmean_df['condition'] = 'no see'
FXopqmean_df['condition'] = 'no see'

WTopqmean_df['group'] = 'WT'
FXopqmean_df['group'] = 'FX'

combined_df = pd.concat([WTmean_df, FXmean_df, WTopqmean_df, FXopqmean_df], ignore_index = True)

# =============================================================================
# 
########################### Plot and check significance #######################
# 
# 
# =============================================================================

# -- Create combined label for plotting
combined_df['group_condition'] = combined_df['group'] + ' - ' + combined_df['condition']
order = ['WT - see', 'WT - no see', 'FX - see', 'FX - no see']

# -- Plotting
plt.figure()

# Violin plot
sns.violinplot(data=combined_df, x='group_condition', y='value', inner=None, order=order, palette='pastel', cut = 2)

# Optional: Add scatter points
sns.stripplot(data=combined_df, x='group_condition', y='value', hue='rat', dodge=False, size=6, palette='dark:.3', jitter=True, order=order)
# sns.boxplot(data=combined_df, x='group_condition', y='value', hue='rat', dodge=False,  palette='dark:.3',  order=order, showfliers = False)

# Draw lines connecting same rats across conditions in each genotype
for group in combined_df['group'].unique():
    for rat in combined_df[combined_df['group'] == group]['rat'].unique():
        sub = combined_df[(combined_df['rat'] == rat) & (combined_df['group'] == group)]
        if len(sub) == 2:
            x_pos = [order.index(f"{group} - {cond}") for cond in sub['condition']]
            plt.plot(x_pos, sub['value'], color='gray', alpha=0.5, linestyle='--')

# Clean plot
# plt.xlabel('Group and Condition')
# plt.ylabel('Value')
# plt.title('Group-wise Condition Comparison')
# plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()

# -- Mann-Whitney U Tests
for group in combined_df['group'].unique():
    sub = combined_df[combined_df['group'] == group]
    see = sub[sub['condition'] == 'see']['value']
    no_see = sub[sub['condition'] == 'no see']['value']
    # stat, p = scipy.stats.wilcoxon(see, no_see, alternative='two-sided')
    stat, p = scipy.stats.mannwhitneyu(see, no_see, alternative='two-sided')
    print(f"{group}: U={stat:.3f}, p-value={p:.4f}")


#%% Plot violin plots for performance of individual pairs (compare across groups only)

file_path = "C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_perf_matches_over_transitions.ods"

WT = pd.read_excel(file_path, "WT50", engine="odf")
FX = pd.read_excel(file_path, "FX50", engine="odf")


sess = 35
which_sessions = 'first'  # or 'last'

new_WT = pd.DataFrame()

for column in WT.columns:
    tempData = WT[column].dropna()
    tempData = tempData.iloc[:sess] if which_sessions == 'first' else tempData.iloc[-sess:]
    tempData = tempData.reset_index(drop=True)

    # Pad if too short
    if len(tempData) < sess:
        tempData = tempData.reindex(range(sess))

    # Now assign
    new_WT[column] = tempData



for column in FX.columns:
    tempData = FX[column].dropna()
    tempData = tempData.iloc[:sess] if which_sessions == 'first' else tempData.iloc[-sess:]
    tempData = tempData.reset_index(drop=True)

    # Pad if too short
    if len(tempData) < sess:
        tempData = tempData.reindex(range(sess))

    # Now assign
    new_FX[column] = tempData



# Transpose for analysis
WT = new_WT.transpose()
FX = new_FX.transpose()

nSess = 5

WT = WT.iloc[:, -nSess:]
FX = FX.iloc[:, -nSess:]

WTmean = WT.mean(axis = 1)
FXmean = FX.mean(axis = 1)

WTmean_df = WTmean.reset_index()
WTmean_df.columns = ['pair', 'value']

FXmean_df = FXmean.reset_index()
FXmean_df.columns = ['pair', 'value']

WTmean_df['group'] = 'WT'
FXmean_df['group'] = 'FX'

combined_df = pd.concat([WTmean_df, FXmean_df], ignore_index = True)

# -- Plotting
plt.figure()

# Violin plot
sns.violinplot(data=combined_df, x='group', y='value', inner=None,  palette='pastel', cut = 2)

# Optional: Add scatter points
sns.stripplot(data=combined_df, x='group', y='value', dodge=False, size=6, palette='dark:.3', jitter=True)
# sns.boxplot(data=combined_df, x='condition', y='value' dodge=False,  palette='dark:.3',  showfliers = False)

# -- Mann-Whitney U Tests
wt_vals = combined_df[combined_df['group'] == 'WT']['value']
fx_vals = combined_df[combined_df['group'] == 'FX']['value']
# stat, p = scipy.stats.wilcoxon(see, no_see, alternative='two-sided')
stat, p = scipy.stats.mannwhitneyu(wt_vals, fx_vals, alternative='two-sided')
print(f" U={stat:.3f}, p-value={p:.4f}")


#%% Plot trialwsie leader probability from state space model

file_path = "C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_trialwise_leader_probability_ssm.ods"
WT = pd.read_excel(file_path, "WT50", engine="odf")
FX = pd.read_excel(file_path, "FX50", engine="odf")



WTmeanPerf = WT.mean(axis=1)
WTerrPerf = WT.std(axis=1) / np.sqrt(WT.shape[1])

FXmeanPerf = FX.mean(axis=1)
FXerrPerf = FX.std(axis=1) / np.sqrt(FX.shape[1])



sns.set(style='ticks')
sns.set_context('poster')
plt.figure()

plt.plot(range(WT.shape[0]), WTmeanPerf, color = [0, 0, 0])
plt.fill_between(range(WT.shape[0]), WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

plt.plot(range(FX.shape[0]), FXmeanPerf, color = [1, 0, 0])
plt.fill_between(range(FX.shape[0]), FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)
plt.axhline(0.5, color = 'green', linestyle = '--')
plt.ylim((-0.05, 1.05))
plt.show()


# Plot violinplot (averaged across sessions for each rat)
 
from scipy.stats import mannwhitneyu


# Step 1: Compute mean for each rat and create DataFrame
WT_means = pd.DataFrame({
    'Mean Value': WT.mean(axis=0).values,
    'Rat': WT.columns,
    'group': 'WT'
})

FX_means = pd.DataFrame({
    'Mean Value': FX.mean(axis=0).values,
    'Rat': FX.columns,
    'group': 'FX'
})

df_means = pd.concat([WT_means, FX_means], ignore_index=True)

# Step 2: Perform Mann-Whitney U Test
wt_values = df_means[df_means['group'] == 'WT']['Mean Value']
fx_values = df_means[df_means['group'] == 'FX']['Mean Value']

u_stat, p_value = mannwhitneyu(wt_values, fx_values, alternative='two-sided')

print(f"Mann-Whitney U test: U={u_stat:.2f}, p={p_value:.4f}")

# Step 3: Plot
plt.figure(figsize=(4, 5))

# Violin plot
sns.violinplot(y='Mean Value', x='group', data=df_means, inner='box', palette='Set2')
sns.violinplot(y='Mean Value', x='group', data=df_means, inner='quart', palette='Set2')

# Scatter plot of each rat
sns.stripplot(y='Mean Value', x='group', data=df_means, size=6, jitter=False, color='black')

plt.ylim((0.2, 0.8))
plt.tight_layout()
plt.show()

#%% Plot RL params

file_path = "C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_RL2_params.ods"


from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
# Load sheets
df_wt = pd.read_excel(file_path, sheet_name='WT50', engine='odf')
df_fx = pd.read_excel(file_path, sheet_name='FX50', engine='odf')

# Label groups
df_wt['Group'] = 'WT50'
df_fx['Group'] = 'FX50'

# Combine
df_all = pd.concat([df_wt, df_fx], ignore_index=True)

# Melt for plotting
df_melted = df_all.melt(id_vars='Group', var_name='Parameter', value_name='Value')

# List of parameters (assuming exactly 4)
params = df_melted['Parameter'].unique()

# Mann-Whitney U test
results = []
for param in params:
    wt_vals = df_melted[(df_melted['Parameter'] == param) & (df_melted['Group'] == 'WT50')]['Value'].dropna()
    fx_vals = df_melted[(df_melted['Parameter'] == param) & (df_melted['Group'] == 'FX50')]['Value'].dropna()
    # stat, p = mannwhitneyu(wt_vals, fx_vals, alternative='two-sided')
    stat, p = ttest_ind(wt_vals, fx_vals, equal_var=False)
    results.append({'Parameter': param, 'p_value': p})

results_df = pd.DataFrame(results).set_index('Parameter')

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, param in enumerate(params):
    ax = axes[i]
    subset = df_melted[df_melted['Parameter'] == param]
    sns.violinplot(data=subset, x='Group', y='Value', ax=ax)
    sns.boxplot(data=subset, x='Group', y='Value', ax=ax)
    sns.stripplot(data=subset, x='Group', y='Value', size=6, jitter=True, color='black', ax=ax)
    
    # Add p-value annotation
    p = results_df.loc[param, 'p_value']
    y_max = subset['Value'].max()
    ax.set_title(f"{param} (p = {p:.3e})", fontsize=10)

# Adjust layout
plt.tight_layout()
plt.show()

