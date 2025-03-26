# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:08:23 2024

@author: shukl
"""

#%% 1. Load libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import smoothfit

#%% 2. Load data and plot state probabilities for all rats

match_data = pd.read_excel("C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/Analyzed/cohortAS2_for_mHMM.ods", 'combined_rats', engine = 'odf')
data = pd.read_csv("C:/Users/shukl/OneDrive/Documents/state_seq_3states.csv")

WT = data.iloc[:23526, :]
FX = data.iloc[23527:, :]


n_subj = 20

# p1 = []
# p2 = []
# idx1 = []
# idx2 = []
# idx3 = []
p3 = []


# Initialize a 4x4 grid of subplots
fig, axes = plt.subplots(4, 5, figsize=(12, 12), sharex=False, sharey=True)
axes = axes.flatten()  # Flatten the 2D array to 1D for easy indexing in the loop

for subj_id in range(1, n_subj + 1):  # Adjust range to include all 20 subjects
    lmbda = 5.0e2
    subj_data = data.loc[data['subj'] == subj_id, ['pr_state_1', 'pr_state_2', 'pr_state_3']]
    
    # Smooth fit for each state
    basis_pr_state_1, coeffs_pr_state_1 = smoothfit.fit1d(
        np.arange(0, len(subj_data), 1), subj_data['pr_state_1'].to_numpy(),
        0, len(subj_data), 1000, degree=1, lmbda=lmbda
    )
    basis_pr_state_2, coeffs_pr_state_2 = smoothfit.fit1d(
        np.arange(0, len(subj_data), 1), subj_data['pr_state_2'].to_numpy(),
        0, len(subj_data), 1000, degree=1, lmbda=lmbda
    )
    basis_pr_state_3, coeffs_pr_state_3 = smoothfit.fit1d(
        np.arange(0, len(subj_data), 1), subj_data['pr_state_3'].to_numpy(),
        0, len(subj_data), 1000, degree=1, lmbda=lmbda
    )
    
    # Plot on the corresponding subplot
    ax = axes[subj_id - 1]  # Adjust index for 0-based indexing
    ax.plot(range(0, len(subj_data)), match_data.loc[match_data['rat'] == subj_id, 'state'], color=(0,0,0), label='match')
    ax.plot(basis_pr_state_1.mesh.p[0], coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]], color=(0,0,1), linestyle="-", label='pr_state_1')
    ax.plot(basis_pr_state_2.mesh.p[0], coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]], color=(1,0,0), linestyle="-", label='pr_state_2')
    ax.plot(basis_pr_state_3.mesh.p[0], coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]], color=(0,1,0), linestyle="-", label='pr_state_3')
    
    # ax.plot(range(0, len(subj_data)), match_data.loc[match_data['rat'] == subj_id, 'state'], color=(0,0,0), label='match')
    # ax.plot(range(0, len(subj_data)), subj_data['pr_state_1'].to_numpy(), color=(0,0,1), linestyle="-", label='pr_state_1')
    # ax.plot(range(0, len(subj_data)), subj_data['pr_state_2'].to_numpy(), color=(1,0,0), linestyle="-", label='pr_state_2')
    # ax.plot(range(0, len(subj_data)), subj_data['pr_state_3'].to_numpy(), color=(0,1,0), linestyle="-", label='pr_state_3')
    ax.set_title(f'Rat {subj_id}')
    
    # Show legend only for the first subplot to avoid clutter
    # if subj_id == 1:
        # ax.legend(['match', 'pr_state_1', 'pr_state_2', 'pr_state_3'])
        # ax.legend(['pr_state_1', 'pr_state_2', 'pr_state_3'])
        
    # p1.append(coeffs_pr_state_1)
    # p2.append(coeffs_pr_state_2)
    # idx3.append(basis_pr_state_3.mesh.p[0])
    p3.append(subj_data['pr_state_3'].to_numpy())

# Adjust layout and show the full plot
plt.tight_layout()
plt.show()


#%% State probability comparisons (for first n trials)

p1 = []
p2 = []
p3 = []

nTrials = 2000
for subj_id in range(1, n_subj + 1):  # Adjust range to include all 20 subjects
    lmbda = 5.0e2
    p1.append(data.loc[data['subj'] == subj_id, 'pr_state_1'].to_numpy())
    p2.append(data.loc[data['subj'] == subj_id, 'pr_state_2'].to_numpy())
    p3.append(data.loc[data['subj'] == subj_id, 'pr_state_3'].to_numpy())
    
for i in range(20):
    if len(p1[i]) > nTrials:
        p1[i] = p1[i][:nTrials]
        p2[i] = p2[i][:nTrials]
        p3[i] = p3[i][:nTrials]
    else:
        p1[i] = np.pad(p1[i], (0, nTrials - len(p1[i])), constant_values = np.nan)
        p2[i] = np.pad(p2[i], (0, nTrials - len(p2[i])), constant_values = np.nan)
        p3[i] = np.pad(p3[i], (0, nTrials - len(p3[i])), constant_values = np.nan)
    

p1_WT = p1[:10]
p2_WT = p2[:10]
p3_WT = p3[:10]

p1_FX = p1[10:]
p2_FX = p2[10:]
p3_FX = p3[10:]



p1_WTmean = np.nanmean(np.array(p1_WT), axis = 0)
p2_WTmean = np.nanmean(np.array(p2_WT), axis = 0)
p3_WTmean = np.nanmean(np.array(p3_WT), axis = 0)

p1_FXmean = np.nanmean(np.array(p1_FX), axis = 0)
p2_FXmean = np.nanmean(np.array(p2_FX), axis = 0)
p3_FXmean = np.nanmean(np.array(p3_FX), axis = 0)


p1_WTerr = np.nanstd(np.array(p1_WT), axis = 0) / np.sqrt(10)
p2_WTerr = np.nanstd(np.array(p2_WT), axis = 0) / np.sqrt(10)
p3_WTerr = np.nanstd(np.array(p3_WT), axis = 0) / np.sqrt(10)

p1_FXerr = np.nanstd(np.array(p1_FX), axis = 0) / np.sqrt(10)
p2_FXerr = np.nanstd(np.array(p2_FX), axis = 0) / np.sqrt(10)
p3_FXerr = np.nanstd(np.array(p3_FX), axis = 0) / np.sqrt(10)

basis_pr_state_1, coeffs_pr_state_1 = smoothfit.fit1d(
    np.arange(0, len(p1_WTmean), 1), p1_WTmean,
    0, len(p1_WTmean), nTrials+500, degree=1, lmbda=lmbda
)

basis_pr_state_2, coeffs_pr_state_2 = smoothfit.fit1d(
    np.arange(0, len(p2_WTmean), 1), p2_WTmean,
    0, len(p2_WTmean), nTrials+500, degree=1, lmbda=lmbda
)

basis_pr_state_3, coeffs_pr_state_3 = smoothfit.fit1d(
    np.arange(0, len(p3_WTmean), 1), p3_WTmean,
    0, len(p3_WTmean), nTrials+500, degree=1, lmbda=lmbda
)

# For error smoothing
basis_pr_state_1err, coeffs_pr_state_1err = smoothfit.fit1d(
    np.arange(0, len(p1_WTerr), 1), p1_WTerr,
    0, len(p1_WTerr), nTrials+500, degree=1, lmbda=lmbda
)

basis_pr_state_2err, coeffs_pr_state_2err = smoothfit.fit1d(
    np.arange(0, len(p2_WTerr), 1), p2_WTerr,
    0, len(p2_WTerr), nTrials+500, degree=1, lmbda=lmbda
)


basis_pr_state_3err, coeffs_pr_state_3err = smoothfit.fit1d(
    np.arange(0, len(p3_WTerr), 1), p3_WTerr,
    0, len(p3_WTerr), nTrials+500, degree=1, lmbda=lmbda
)

plt.figure()
plt.plot(basis_pr_state_1.mesh.p[0], coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]], color = 'b')

plt.fill_between(basis_pr_state_1.mesh.p[0], coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]] - coeffs_pr_state_1err[basis_pr_state_1err.nodal_dofs[0]],
                 coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]] + coeffs_pr_state_1err[basis_pr_state_1err.nodal_dofs[0]], alpha=0.2)

plt.plot(basis_pr_state_2.mesh.p[0], coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]], color = 'r')

plt.fill_between(basis_pr_state_2.mesh.p[0], coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]] - coeffs_pr_state_2err[basis_pr_state_2err.nodal_dofs[0]],
                 coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]] + coeffs_pr_state_2err[basis_pr_state_2err.nodal_dofs[0]], alpha=0.2)

plt.plot(basis_pr_state_3.mesh.p[0], coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]], color = 'g')

plt.fill_between(basis_pr_state_3.mesh.p[0], coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]] - coeffs_pr_state_3err[basis_pr_state_3err.nodal_dofs[0]],
                 coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]] + coeffs_pr_state_3err[basis_pr_state_3err.nodal_dofs[0]], alpha=0.2)


plt.axhline(1/3, linestyle = '--', color = 'k')
plt.ylim((0, 0.8))
plt.xlabel(('Trials'))
plt.ylabel(('Probability'))
# plt.legend(['State 1', 'State 2', 'State 3', 'Random'], loc = 'best')
plt.title(f'WT (first {nTrials} trials)')
plt.show()
plt.tight_layout()


#################### For FX rats ##############################
basis_pr_state_1, coeffs_pr_state_1 = smoothfit.fit1d(
    np.arange(0, len(p1_FXmean), 1), p1_FXmean,
    0, len(p1_FXmean), nTrials+500, degree=1, lmbda=lmbda
)

basis_pr_state_2, coeffs_pr_state_2 = smoothfit.fit1d(
    np.arange(0, len(p2_FXmean), 1), p2_FXmean,
    0, len(p2_FXmean), nTrials+500, degree=1, lmbda=lmbda
)

basis_pr_state_3, coeffs_pr_state_3 = smoothfit.fit1d(
    np.arange(0, len(p3_FXmean), 1), p3_FXmean,
    0, len(p3_FXmean), nTrials+500, degree=1, lmbda=lmbda
)

# For error smoothing
basis_pr_state_1err, coeffs_pr_state_1err = smoothfit.fit1d(
    np.arange(0, len(p1_FXerr), 1), p1_FXerr,
    0, len(p1_FXerr), nTrials+500, degree=1, lmbda=lmbda
)

basis_pr_state_2err, coeffs_pr_state_2err = smoothfit.fit1d(
    np.arange(0, len(p2_FXerr), 1), p2_FXerr,
    0, len(p2_FXerr), nTrials+500, degree=1, lmbda=lmbda
)


basis_pr_state_3err, coeffs_pr_state_3err = smoothfit.fit1d(
    np.arange(0, len(p3_FXerr), 1), p3_FXerr,
    0, len(p3_FXerr), nTrials+500, degree=1, lmbda=lmbda
)

plt.figure()
plt.plot(basis_pr_state_1.mesh.p[0], coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]], color = 'b')

plt.fill_between(basis_pr_state_1.mesh.p[0], coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]] - coeffs_pr_state_1err[basis_pr_state_1err.nodal_dofs[0]],
                 coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]] + coeffs_pr_state_1err[basis_pr_state_1err.nodal_dofs[0]], alpha=0.2)

plt.plot(basis_pr_state_2.mesh.p[0], coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]], color = 'r')

plt.fill_between(basis_pr_state_2.mesh.p[0], coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]] - coeffs_pr_state_2err[basis_pr_state_2err.nodal_dofs[0]],
                 coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]] + coeffs_pr_state_2err[basis_pr_state_2err.nodal_dofs[0]], alpha=0.2)

plt.plot(basis_pr_state_3.mesh.p[0], coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]], color = 'g')

plt.fill_between(basis_pr_state_3.mesh.p[0], coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]] - coeffs_pr_state_3err[basis_pr_state_3err.nodal_dofs[0]],
                 coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]] + coeffs_pr_state_3err[basis_pr_state_3err.nodal_dofs[0]], alpha=0.2)


plt.axhline(1/3, linestyle = '--', color = 'k')
plt.ylim((0, 0.8))
plt.xlim((0, nTrials))
plt.xlabel(('Trials'))
plt.ylabel(('Probability'))
# plt.legend(['State 1', 'State 2', 'State 3', 'Random'], loc = 'best')
plt.title(f'FX (first {nTrials} trials)')

plt.show()
plt.tight_layout()

#%% State probability comparisons (for last n trials)

p1 = []
p2 = []
p3 = []

nTrials = 1500
for subj_id in range(1, n_subj + 1):  # Adjust range to include all 20 subjects
    lmbda = 5.0e2
    p1.append(data.loc[data['subj'] == subj_id, 'pr_state_1'].to_numpy())
    p2.append(data.loc[data['subj'] == subj_id, 'pr_state_2'].to_numpy())
    p3.append(data.loc[data['subj'] == subj_id, 'pr_state_3'].to_numpy())
    
for i in range(20):
    if len(p1[i]) > nTrials:
        p1[i] = p1[i][:nTrials]
        p2[i] = p2[i][:nTrials]
        p3[i] = p3[i][:nTrials]
    else:
        p1[i] = np.pad(p1[i], (0, nTrials - len(p1[i])), constant_values = np.nan)
        p2[i] = np.pad(p2[i], (0, nTrials - len(p2[i])), constant_values = np.nan)
        p3[i] = np.pad(p3[i], (0, nTrials - len(p3[i])), constant_values = np.nan)
    

p1_WT = p1[:10]
p2_WT = p2[:10]
p3_WT = p3[:10]

p1_FX = p1[10:]
p2_FX = p2[10:]
p3_FX = p3[10:]

p1_WTmean = np.nanmean(np.array(p1_WT), axis = 0)
p2_WTmean = np.nanmean(np.array(p2_WT), axis = 0)
p3_WTmean = np.nanmean(np.array(p3_WT), axis = 0)

p1_FXmean = np.nanmean(np.array(p1_FX), axis = 0)
p2_FXmean = np.nanmean(np.array(p2_FX), axis = 0)
p3_FXmean = np.nanmean(np.array(p3_FX), axis = 0)


p1_WTerr = np.nanstd(np.array(p1_WT), axis = 0) / np.sqrt(10)
p2_WTerr = np.nanstd(np.array(p2_WT), axis = 0) / np.sqrt(10)
p3_WTerr = np.nanstd(np.array(p3_WT), axis = 0) / np.sqrt(10)

p1_FXerr = np.nanstd(np.array(p1_FX), axis = 0) / np.sqrt(10)
p2_FXerr = np.nanstd(np.array(p2_FX), axis = 0) / np.sqrt(10)
p3_FXerr = np.nanstd(np.array(p3_FX), axis = 0) / np.sqrt(10)

basis_pr_state_1, coeffs_pr_state_1 = smoothfit.fit1d(
    np.arange(0, len(p1_WTmean), 1), p1_WTmean,
    0, len(p1_WTmean), nTrials, degree=1, lmbda=lmbda
)

basis_pr_state_2, coeffs_pr_state_2 = smoothfit.fit1d(
    np.arange(0, len(p2_WTmean), 1), p2_WTmean,
    0, len(p2_WTmean), nTrials, degree=1, lmbda=lmbda
)

basis_pr_state_3, coeffs_pr_state_3 = smoothfit.fit1d(
    np.arange(0, len(p3_WTmean), 1), p3_WTmean,
    0, len(p3_WTmean), nTrials, degree=1, lmbda=lmbda
)

# For error smoothing
basis_pr_state_1err, coeffs_pr_state_1err = smoothfit.fit1d(
    np.arange(0, len(p1_WTerr), 1), p1_WTerr,
    0, len(p1_WTerr), nTrials, degree=1, lmbda=lmbda
)

basis_pr_state_2err, coeffs_pr_state_2err = smoothfit.fit1d(
    np.arange(0, len(p2_WTerr), 1), p2_WTerr,
    0, len(p2_WTerr), nTrials, degree=1, lmbda=lmbda
)


basis_pr_state_3err, coeffs_pr_state_3err = smoothfit.fit1d(
    np.arange(0, len(p3_WTerr), 1), p3_WTerr,
    0, len(p3_WTerr), nTrials, degree=1, lmbda=lmbda
)

plt.figure()
plt.plot(basis_pr_state_1.mesh.p[0], coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]], color = 'b')

plt.fill_between(basis_pr_state_1.mesh.p[0], coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]] - coeffs_pr_state_1err[basis_pr_state_1err.nodal_dofs[0]],
                 coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]] + coeffs_pr_state_1err[basis_pr_state_1err.nodal_dofs[0]], alpha=0.2)

plt.plot(basis_pr_state_2.mesh.p[0], coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]], color = 'r')

plt.fill_between(basis_pr_state_2.mesh.p[0], coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]] - coeffs_pr_state_2err[basis_pr_state_2err.nodal_dofs[0]],
                 coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]] + coeffs_pr_state_2err[basis_pr_state_2err.nodal_dofs[0]], alpha=0.2)

plt.plot(basis_pr_state_3.mesh.p[0], coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]], color = 'g')

plt.fill_between(basis_pr_state_3.mesh.p[0], coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]] - coeffs_pr_state_3err[basis_pr_state_3err.nodal_dofs[0]],
                 coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]] + coeffs_pr_state_3err[basis_pr_state_3err.nodal_dofs[0]], alpha=0.2)


plt.axhline(1/3, linestyle = '--', color = 'k')
plt.ylim((0, 0.8))
plt.xlabel(('Trials'))
plt.ylabel(('Probability'))
# plt.legend(['State 1', 'State 2', 'State 3', 'Random'], loc = 'best')
plt.title((f'WT (last {nTrials} trials)'))
plt.show()
plt.tight_layout()


########################## For FX rats ####################################
basis_pr_state_1, coeffs_pr_state_1 = smoothfit.fit1d(
    np.arange(0, len(p1_FXmean), 1), p1_FXmean,
    0, len(p1_FXmean), nTrials, degree=1, lmbda=lmbda
)

basis_pr_state_2, coeffs_pr_state_2 = smoothfit.fit1d(
    np.arange(0, len(p2_FXmean), 1), p2_FXmean,
    0, len(p2_FXmean), nTrials, degree=1, lmbda=lmbda
)

basis_pr_state_3, coeffs_pr_state_3 = smoothfit.fit1d(
    np.arange(0, len(p3_FXmean), 1), p3_FXmean,
    0, len(p3_FXmean), nTrials, degree=1, lmbda=lmbda
)

# For error smoothing
basis_pr_state_1err, coeffs_pr_state_1err = smoothfit.fit1d(
    np.arange(0, len(p1_WTerr), 1), p1_WTerr,
    0, len(p1_WTerr), nTrials, degree=1, lmbda=lmbda
)

basis_pr_state_2err, coeffs_pr_state_2err = smoothfit.fit1d(
    np.arange(0, len(p2_WTerr), 1), p2_WTerr,
    0, len(p2_WTerr), nTrials, degree=1, lmbda=lmbda
)


basis_pr_state_3err, coeffs_pr_state_3err = smoothfit.fit1d(
    np.arange(0, len(p3_WTerr), 1), p3_WTerr,
    0, len(p3_WTerr), nTrials, degree=1, lmbda=lmbda
)

plt.figure()
plt.plot(basis_pr_state_1.mesh.p[0], coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]], color = 'b')

plt.fill_between(basis_pr_state_1.mesh.p[0], coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]] - coeffs_pr_state_1err[basis_pr_state_1err.nodal_dofs[0]],
                 coeffs_pr_state_1[basis_pr_state_1.nodal_dofs[0]] + coeffs_pr_state_1err[basis_pr_state_1err.nodal_dofs[0]], alpha=0.2)

plt.plot(basis_pr_state_2.mesh.p[0], coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]], color = 'r')

plt.fill_between(basis_pr_state_2.mesh.p[0], coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]] - coeffs_pr_state_2err[basis_pr_state_2err.nodal_dofs[0]],
                 coeffs_pr_state_2[basis_pr_state_2.nodal_dofs[0]] + coeffs_pr_state_2err[basis_pr_state_2err.nodal_dofs[0]], alpha=0.2)

plt.plot(basis_pr_state_3.mesh.p[0], coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]], color = 'g')

plt.fill_between(basis_pr_state_3.mesh.p[0], coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]] - coeffs_pr_state_3err[basis_pr_state_3err.nodal_dofs[0]],
                 coeffs_pr_state_3[basis_pr_state_3.nodal_dofs[0]] + coeffs_pr_state_3err[basis_pr_state_3err.nodal_dofs[0]], alpha=0.2)


plt.axhline(1/3, linestyle = '--', color = 'k')
plt.ylim((0, 0.8))
plt.xlabel(('Trials'))
plt.ylabel(('Probability'))
# plt.legend(['State 1', 'State 2', 'State 3', 'Random'], loc = 'best')
plt.title((f'FX (last {nTrials} trials)'))
plt.show()
plt.tight_layout()

#%% 3. Calculate fraction occupancy of each state and plot 

state1_counts = []
state2_counts = []
state3_counts = []

nTrials = 2000

for subj_id in range(1, n_subj + 1):
    subj_data = data.loc[data['subj'] == subj_id, 'state'].to_numpy()
    
    # state1 = len(np.where(subj_data == 1)[0]) / len(subj_data)
    # state2 = len(np.where(subj_data == 2)[0]) / len(subj_data)
    # state3 = len(np.where(subj_data == 3)[0]) / len(subj_data)
    
    subj_data = subj_data[-nTrials:]
    
    state1 = len(np.where(subj_data == 1)[0]) / nTrials
    state2 = len(np.where(subj_data == 2)[0]) / nTrials
    state3 = len(np.where(subj_data == 3)[0]) / nTrials
    
    state1_counts.append(state1)
    state2_counts.append(state2)
    state3_counts.append(state3)





# For WT_counts DataFrame
WT_counts = pd.DataFrame({
    'state': ['state 1'] * len(state1_counts[:10]) + 
             ['state 2'] * len(state2_counts[:10]) + 
             ['state 3'] * len(state3_counts[:10]),
    'counts': state1_counts[:10] + state2_counts[:10] + state3_counts[:10]
})

# For FX_counts DataFrame
FX_counts = pd.DataFrame({
    'state': ['state 1'] * len(state1_counts[10:]) + 
             ['state 2'] * len(state2_counts[10:]) + 
             ['state 3'] * len(state3_counts[10:]),
    'counts': state1_counts[10:] + state2_counts[10:] + state3_counts[10:]
})

WT_counts['group'] = 'WT'
FX_counts['group'] = 'FX'

count_data = pd.concat([WT_counts, FX_counts])

# Plot fraction occupancy for each genotype
ax = sns.barplot(x = 'state', y = 'counts', data = count_data, hue = 'group', errorbar="se", alpha = 0.6)
ax.set_ylabel('fraction occupancy')
sns.stripplot(x='state', y='counts', data = count_data, hue = 'group',  jitter = True, dodge = True, alpha=0.9, size=5, ax = ax)


##############################################################################
from scipy.stats import ttest_ind

column1 = WT_counts.loc[WT_counts['state'] == 'state 3', 'counts'].to_numpy()
column2 = FX_counts.loc[FX_counts['state'] == 'state 3', 'counts'].to_numpy()

# Perform an unpaired t-test
t_stat, p_value = ttest_ind(column1, column2)

# Display the results
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis (significant difference between means)")
else:
    print("Fail to reject the null hypothesis (no significant difference between means)")


#%% Chi square test of independence for checking associations of match events with states

# import statsmodels.api as sm
import scipy.stats as stats
coeffs = []
props = []

for subj_id in range(1, n_subj + 1):
    subj_data = data.loc[data['subj'] == subj_id]
    # subj_match_data = match_data.loc[match_data['rat'] == subj_id, 'state']
    
    df = pd.DataFrame({'state': subj_data['state'], 'match': match_data.loc[match_data['rat'] == subj_id, 'state']})
    # Define the full set of states (e.g., 1, 2, 3)
    all_states = [1, 2, 3]
    
    # Calculate proportions while ensuring all states are present
    proportions = (
        df.groupby('state')['match']
        .mean()
        .reindex(all_states, fill_value=0)
        .reset_index()
    )
    
    # Rename columns for clarity
    proportions.columns = ['state', 'proportion']



    # Create a contingency table
    # Initialize a 3x2 contingency table with zeros
    contingency_table = pd.DataFrame(np.zeros((3, 2)), 
                                     index=[1, 2, 3], 
                                     columns=[0, 1])

    # Populate the contingency table
    populated_table = pd.crosstab(df['state'], df['match'])
    
    # Update the initialized table with the populated values
    contingency_table.update(populated_table)

    # Perform the chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(populated_table)
    
    # y = subj_match_data
    
    # # X = subj_data[['pr_state_1', 'pr_state_2', 'pr_state_3']].to_numpy()
    # X = subj_data['state'].to_numpy()
    # # X = sm.add_constant(X)
    # model = sm.GLM(y, X, family=sm.families.Binomial())
    # result = model.fit(maxiter = 10000, tol = 1e-4)
    # coeffs.append(result.params)
    coeffs.append(p)
    props.append(proportions)
    
coeffs = np.array(coeffs)
WT_coeffs = coeffs[:10] # first 10 entries are from WTs
FX_coeffs = coeffs[10:] # last 10 entries are from KOs

# Plot proprtion of match events in each state (for all rats)

from scipy.stats import sem

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(props, ignore_index=True)

# Calculate mean and SEM for each state across all entries
summary_df = combined_df.groupby('state')['proportion'].agg(
    mean=np.mean, sem=sem
).reset_index()

# Plotting
plt.figure(figsize = (8, 5))
ax = sns.barplot(x = 'state', y = 'mean', data=summary_df, ci=None, palette='muted')

# Get the x positions of the bars
x_positions = range(len(summary_df))

# Adding error bars for SEM
ax.errorbar(x=x_positions, y=summary_df['mean'], 
            yerr=summary_df['sem'], fmt='none', capsize=5, color='black')

# Adding labels and title
ax.set_xlabel('State')
ax.set_ylabel('Mean Proportion ± SEM')
ax.set_title('Mean Proportions by State with SEM')
plt.xticks(ticks=x_positions, labels=summary_df['state'])  # Ensuring correct labels for x-axis
plt.tight_layout()
plt.show()


# Split the list into WT and FX
WT_proportions = pd.concat(props[:10], ignore_index=True)
FX_proportions = pd.concat(props[10:], ignore_index=True)

WT_proportions['group'] = 'WT'
FX_proportions['group'] = 'FX'

# Calculate the mean and SEM for both WT and FX for each state
WT_summary = WT_proportions.groupby('state')['proportion'].agg(mean=np.mean, sem=sem).reset_index()
FX_summary = FX_proportions.groupby('state')['proportion'].agg(mean=np.mean, sem=sem).reset_index()

# Add a column to differentiate between WT and FX
WT_summary['group'] = 'WT'
FX_summary['group'] = 'FX'

# Combine both summaries into one DataFrame
combined_summary = pd.concat([WT_summary, FX_summary])

# Create a bar plot
plt.figure(figsize=(10, 6))

# Plotting side by side with bar width adjustment
sns.barplot(x='state', y = 'mean', hue = 'group', data = combined_summary, ci = None, palette = 'muted')

# Adding error bars for SEM
x_positions = range(len(WT_summary))
for i, state in enumerate(WT_summary['state']):
    plt.errorbar(x=i - 0.2, y = WT_summary.loc[WT_summary['state'] == state, 'mean'].values, 
                 yerr = WT_summary.loc[WT_summary['state'] == state, 'sem'].values, fmt = 'none', capsize = 5, color = 'black')
    plt.errorbar(x=i + 0.2, y = FX_summary.loc[FX_summary['state'] == state, 'mean'].values, 
                 yerr = FX_summary.loc[FX_summary['state'] == state, 'sem'].values, fmt = 'none', capsize = 5, color = 'black')

# Add individual data points
# WT individual points, manually adjust x-position for better alignment
sns.stripplot(x='state', y='proportion', data = pd.concat([WT_proportions, FX_proportions]), hue = 'group', jitter = True, dodge = True, alpha=0.6, size=5)


# Adding labels and title
plt.xlabel('State')
plt.ylabel('P(correct)')
plt.title('Proportions of correct choices by state')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.legend(title ='Group', loc = 'upper left')


#%% Load global transition and emission matrices

import numpy as np

# Load the matrices
transition_matrix = np.load("C:/Users/shukl/OneDrive/Documents/global_transition_matrix_3states.npy") # (n_subj, n_states, n_states)
emission_matrix = np.load("C:/Users/shukl/OneDrive/Documents/global_emission_matrix_3states.npy") # (n_subj, n_states, n_category_observables)
cat_emission_matrix = np.load("C:/Users/shukl/OneDrive/Documents/global_cat_emission_matrix_3states.npy") # (n_subj, n_states, n_category_observables)

# Check the shapes
print("Transition Matrix Shape:", transition_matrix.shape)
print("Emission Matrix Shape:", emission_matrix.shape)
print("Categorical Emission Matrix Shape:", cat_emission_matrix.shape)

# cat_emission = pd.DataFrame(cat_emission_matrix)
# Plot the kernel density of emission prob for all subjects

burn_in = 500
WT_cat_emiss = cat_emission_matrix[:9, :, :]
FX_cat_emiss = cat_emission_matrix[10:, :, :]

# Plot for WT rats
fig, axes = plt.subplots(1, 3, figsize = (5, 5), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array to 1D for easy indexing in the loop

for state in range(3):
    for i in range(len(WT_cat_emiss)):
        ax = axes[state]
        sns.kdeplot(WT_cat_emiss[i, burn_in:5000, 2*state], color = 'r',  alpha = 0.7, linewidth = 1.0, linestyle = '--', ax = ax)
        sns.kdeplot(WT_cat_emiss[i, burn_in:5000, 2*state+1], color = 'g',  alpha = 0.7, linewidth = 1.0, linestyle = '--',ax = ax)
        ax.set_xlabel("Conditional probability")
        
    sns.kdeplot(np.mean(WT_cat_emiss[:, burn_in:5000, 2*state], axis = 0), color = 'r',  alpha = 1.0,  linewidth = 2.0, ax = ax)
    sns.kdeplot(np.mean(WT_cat_emiss[:, burn_in:5000, 2*state+1], axis = 0), color = 'g',  alpha = 1.0,  linewidth = 2.0, ax = ax)

# Plot for FX 
fig, axes = plt.subplots(1, 3, figsize = (5, 5), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array to 1D for easy indexing in the loop       
for state in range(3):
    for i in range(len(FX_cat_emiss)):
        ax = axes[state]
        sns.kdeplot(FX_cat_emiss[i, burn_in:5000, 2*state], color = 'r', linestyle = '--', alpha = 0.7, linewidth = 0.5*2, ax = ax)
        sns.kdeplot(FX_cat_emiss[i, burn_in:5000, 2*state+1], color = 'g', linestyle = '--', alpha = 0.7, linewidth = 0.5*2, ax = ax)
        ax.set_xlabel("Conditional probability")
    sns.kdeplot(np.mean(FX_cat_emiss[:, burn_in:5000, 2*state], axis = 0), color = 'r',  alpha = 1.0,  linewidth = 3.0, ax = ax)
    sns.kdeplot(np.mean(FX_cat_emiss[:, burn_in:5000, 2*state+1], axis = 0), color = 'g',  alpha = 1.0,  linewidth = 3.0, ax = ax)
        
# Plot for all rats
fig, axes = plt.subplots(1, 3, figsize = (5, 5), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array to 1D for easy indexing in the loop

for state in range(3):
    for i in range(len(cat_emission_matrix)):
        ax = axes[state]
        sns.kdeplot(cat_emission_matrix[i, burn_in:5000, 2*state], color = 'r',  linestyle = '--', alpha = 0.7,  linewidth = 0.5*2, ax = ax)
        sns.kdeplot(cat_emission_matrix[i, burn_in:5000, 2*state+1], color = 'g', linestyle = '--', alpha = 0.7, linewidth = 0.5*2, ax = ax)
        ax.set_xlabel("Conditional probability")     

    sns.kdeplot(np.mean(cat_emission_matrix[:, burn_in:5000, 2*state], axis = 0), color = 'r',  alpha = 1.0,  linewidth = 3.0, ax = ax)
    sns.kdeplot(np.mean(cat_emission_matrix[:, burn_in:5000, 2*state+1], axis = 0), color = 'g',  alpha = 1.0,  linewidth = 3.0, ax = ax)
    
# Plot WT and FX mean on the same plot
fig, axes = plt.subplots(1, 3, figsize = (5, 5), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array to 1D for easy indexing in the loop

for state in range(3):
    ax = axes[state]
    
    sns.kdeplot(np.mean(WT_cat_emiss[:, burn_in:5000, 2*state], axis = 0), color = 'r',  alpha = 1.0, ax = ax)
    sns.kdeplot(np.mean(WT_cat_emiss[:, burn_in:5000, 2*state+1], axis = 0), color = 'g',  alpha = 1.0, ax = ax)
    
    sns.kdeplot(np.mean(FX_cat_emiss[:, burn_in:5000, 2*state], axis = 0), color = 'r',  linestyle = '--', alpha = 1.0, ax = ax)
    sns.kdeplot(np.mean(FX_cat_emiss[:, burn_in:5000, 2*state+1], axis = 0), color = 'g', linestyle = '--',  alpha = 1.0, ax = ax)
    ax.set_xlabel("Conditional probability") 
    
#%% Compare transition matrices

import numpy as np
import pandas as pd


WT = transition_matrix[:10,:,:]
FX = transition_matrix[10:,:,:]

# Assuming FX is your 10x3x3 transition matrix
num_individuals = FX.shape[0]
from_states, to_states = FX.shape[1], FX.shape[2]

# Initialize lists to store data
data = {
    'individual': [],
    'from_state': [],
    'to_state': [],
    'prob': []
}

# Populate the data dictionary
for individual in range(num_individuals):
    for from_state in range(from_states):
        for to_state in range(to_states):
            data['individual'].append(individual)
            data['from_state'].append(from_state)
            data['to_state'].append(to_state)
            data['prob'].append(FX[individual, from_state, to_state])

# Convert the dictionary to a pandas DataFrame
df1 = pd.DataFrame(data)

# Assuming FX is your 10x3x3 transition matrix
num_individuals = WT.shape[0]
from_states, to_states = WT.shape[1], WT.shape[2]

# Initialize lists to store data
data = {
    'individual': [],
    'from_state': [],
    'to_state': [],
    'prob': []
}

# Populate the data dictionary
for individual in range(num_individuals):
    for from_state in range(from_states):
        for to_state in range(to_states):
            data['individual'].append(individual)
            data['from_state'].append(from_state)
            data['to_state'].append(to_state)
            data['prob'].append(FX[individual, from_state, to_state])

# Convert the dictionary to a pandas DataFrame
df2 = pd.DataFrame(data)


df1['group'] = 'FX'
df2['group'] = 'WT'

# Combine both DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Create a 'transition' column that concatenates 'from_state' and 'to_state'
combined_df['transition'] = combined_df['from_state'].astype(str) + ' -> ' + combined_df['to_state'].astype(str)

# Select relevant columns ('transition', 'prob', 'group')
final_df = combined_df[['transition', 'prob', 'group']]


###################### Perform pairwise comparisons ###########################

from scipy import stats
import pandas as pd
import statsmodels.stats.multitest as smm

# Step 1: Get unique transitions
unique_transitions = final_df['transition'].unique()

# Step 2: Initialize lists to store results
transition_list = []
group1_mean = []
group2_mean = []
t_stat_list = []
p_val_list = []

# Step 3: Perform t-test for each transition type
for transition in unique_transitions:
    # Filter data for the current transition
    data_group1 = final_df[(final_df['transition'] == transition) & (final_df['group'] == 'FX')]['prob']
    data_group2 = final_df[(final_df['transition'] == transition) & (final_df['group'] == 'WT')]['prob']
    
    # Perform independent t-test
    t_stat, p_val = stats.ttest_ind(data_group1, data_group2, equal_var=False) # Welch's t-test
    
    # Store results
    transition_list.append(transition)
    group1_mean.append(data_group1.mean())
    group2_mean.append(data_group2.mean())
    t_stat_list.append(t_stat)
    p_val_list.append(p_val)

# Step 4: Multiple comparison correction (Benjamini-Hochberg)
adjusted_p_vals = smm.multipletests(p_val_list, method='fdr_bh')[1]

# Step 5: Create a DataFrame with the results
results_df = pd.DataFrame({
    'transition': transition_list,
    'group1_mean': group1_mean,
    'group2_mean': group2_mean,
    't_stat': t_stat_list,
    'p_val': p_val_list,
    'adjusted_p_val': adjusted_p_vals
})

################## Plotting the pairwise comparisons ##########################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Calculate mean and standard error for each transition and group
summary_df = final_df.groupby(['transition', 'group']).agg(
    mean_prob=('prob', 'mean'),
    sem_prob=('prob', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
).reset_index()

# Step 2: Plotting with seaborn
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=summary_df,
    x='transition',
    y='mean_prob',
    hue='group',
    errorbar=None,  # Disable seaborn's default confidence interval calculation
    palette='Set2'
)

# Step 3: Add error bars using the correct positions
for i, row in summary_df.iterrows():
    # Get the x position of the current bar (group-specific position)
    x_pos = ax.patches[i].get_x() + ax.patches[i].get_width() / 2
    
    # Get the y-position of the current bar (height of the bar)
    y_pos = ax.patches[i].get_height()

    # Add the error bars at the correct position
    ax.errorbar(
        x=x_pos,  # Position of the bar
        y=y_pos,  # The height of the bar (mean probability)
        yerr=row['sem_prob'],
        fmt='none',  # No marker, just the error bars
        ecolor='black',
        capsize=5,
        capthick=2,
        elinewidth=1.5
    )

# Step 4: Customize the plot
plt.xlabel('Transition Type')
plt.ylabel('Probability')
plt.title('Pairwise Comparisons of Transition Probabilities Across Groups')
plt.xticks(rotation=45)
plt.legend(title='Group')
plt.tight_layout()
plt.show()

