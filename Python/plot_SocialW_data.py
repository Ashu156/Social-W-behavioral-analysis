# -*- coding: utf-8 -*-
"""op
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

#%% 2. Plot for performance

WT = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_pokeTime.ods", "WT50", engine="odf")
FX = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_pokeTime.ods", "FX50", engine="odf")
mixedWT = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_pokeTime.ods", "Mixed50_WT", engine="odf")
mixedFX = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/allCohorts_pokeTime.ods", "Mixed50_FX", engine="odf")


    
WT = WT.transpose()
FX = FX.transpose()
mixedWT = mixedWT.transpose()
mixedFX = mixedFX.transpose()

sess = 34

WT = WT.iloc[:, :sess]
FX = FX.iloc[:, :sess]
mixedWT = mixedWT.iloc[:, :sess]
mixedFX = mixedFX.iloc[:, :sess]

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

MixedWTmeanPerf = mixedWT.mean(axis=0)
MixedWTerrPerf = mixedWT.std(axis=0) / np.sqrt(mixedWT.shape[0])

MixedFXmeanPerf = mixedFX.mean(axis=0)
MixedFXerrPerf = mixedFX.std(axis=0) / np.sqrt(mixedFX.shape[0])

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

# Smooth performance for mixed WT group
basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,MixedWTmeanPerf.shape[0], 1), MixedWTmeanPerf, 0, MixedWTmeanPerf.shape[0], 1000, degree=1, lmbda=lmbda)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,MixedWTerrPerf.shape[0], 1), MixedWTerrPerf, 0, MixedWTerrPerf.shape[0], 1000, degree=1, lmbda=lmbda)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'blue', linestyle ="-")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                  coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

# Smooth performance for mixed FX group
basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,MixedFXmeanPerf.shape[0], 1), MixedFXmeanPerf, 0, MixedFXmeanPerf.shape[0], 1000, degree=1, lmbda=lmbda)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,MixedFXerrPerf.shape[0], 1), MixedFXerrPerf, 0, MixedFXerrPerf.shape[0], 1000, degree=1, lmbda=lmbda)



plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], color = 'pink', linestyle ="-")

# Plot the shaded region representing the smoothed SEM
plt.fill_between(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]] - coeffs_err[basis_err.nodal_dofs[0]],
                    coeffs_mean[basis_mean.nodal_dofs[0]] + coeffs_err[basis_err.nodal_dofs[0]], alpha=0.2)

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


WT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_IMI.ods", "WTopq", engine="odf")
FX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_IMI.ods", "FXopq", engine="odf")


WT = WT.iloc[:, :8]
FX = FX.iloc[:, :8]

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

WTmeanPerf = WT.mean(axis=0)
WTerrPerf = WT.std(axis=0) / np.sqrt(WT.shape[0])

FXmeanPerf = FX.mean(axis=0)
FXerrPerf = FX.std(axis=0) / np.sqrt(FX.shape[0])

basis_mean, coeffs_mean = smoothfit.fit1d(np.arange(0,WTmeanPerf.shape[0], 1), WTmeanPerf, 0, WTmeanPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)
basis_err, coeffs_err = smoothfit.fit1d(np.arange(0,WTerrPerf.shape[0], 1), WTerrPerf, 0, WTerrPerf.shape[0], 1000, degree=1, lmbda=5.0e-1)


sns.set(style='ticks')
sns.set_context('poster')
plt.figure()

plt.plot(basis_mean.mesh.p[0], coeffs_mean[basis_mean.nodal_dofs[0]], "-", label="FX pairs")

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


sns.set(style='ticks')
plt.figure()

plt.plot(range(WT.shape[1]), WTmeanPerf)
plt.fill_between(range(WT.shape[1]), WTmeanPerf - WTerrPerf, WTmeanPerf + WTerrPerf, alpha = 0.2)

plt.plot(range(FX.shape[1]), FXmeanPerf)
plt.fill_between(range(FX.shape[1]), FXmeanPerf - FXerrPerf, FXmeanPerf + FXerrPerf, alpha = 0.2)

# Customize the plot
plt.xlabel('Session#', fontsize = 14)
plt.ylabel('Inter-match interval (s)', fontsize = 14)

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
