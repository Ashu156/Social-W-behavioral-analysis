# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:05:19 2024

@author: shukl
"""

#%% Load libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import pandas as pd
import smoothfit
import statsmodels.api as sm
from statsmodels.formula.api import ols

#%% Load the variables of interest

# fWT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_peer_directed_attention_mean_freq_looking_at_partner_following_events.ods", "WT_50", engine="odf")
# fFX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_peer_directed_attention_mean_freq_looking_at_partner_following_events.ods", "WT_50", engine="odf")

fWT = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_predictive_accuracy.ods", sheet_name = "WT_50_data", engine="odf")
fFX = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_predictive_accuracy.ods", sheet_name = "FX_50_data", engine="odf")

fWT = fWT.iloc[20:, :36]
fFX = fFX.iloc[:, :36]

fWT = fWT.transpose()
fFX = fFX.transpose()


# fWT = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_matches_over_transitions_WT_opaque.csv")
# fFX = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_perf_matches_over_transitions_FX_opaque.csv")

# fWT = fWT.iloc[:10]
# fFX = fFX.iloc[:10]






WT = fWT.stack()
WT = WT.to_frame()
WT.reset_index(level=1, drop=True, inplace=True)
WT['Genotype'] = 'WT'

FX = fFX.stack()
FX = FX.to_frame()
FX.reset_index(level=1, drop=True, inplace=True)
FX['Genotype'] = 'FX'

concatenated_df = pd.concat([WT, FX])

concatenated_df.reset_index(inplace=True)
concatenated_df.rename(columns={'index': 'session'}, inplace=True)
concatenated_df.rename(columns={0: 'perf'}, inplace=True)


perf_lm = ols('perf ~ C(session)*C(Genotype)', data=concatenated_df).fit()

table = sm.stats.anova_lm(perf_lm, typ=2) # Type 2 ANOVA DataFrame

print(table)


#%%

import pandas as pd

data = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_predictive_accuracy.ods", sheet_name = "data_50", engine="odf")
shuffle = pd.read_excel("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allCohorts_predictive_accuracy.ods", sheet_name = "shuffle_50", engine="odf")

fWT = shuffle['WT']
fWT = fWT.to_frame(name = 'pct_correct')
fWT['Subject'] = ['SU' + str(i + 1) for i in range(len(fWT))]
fWT['type'] = ['data' for i in range(len(fWT))]
fWT['genotype'] = ['WT' for i in range(len(fWT))]

fFX = shuffle['FX']
fFX = fFX.to_frame(name = 'pct_correct')
fFX['Subject'] = ['SU' + str(i + 1 + len(fWT)) for i in range(len(fFX))]
fFX['type'] = ['data' for i in range(len(fFX))]
fFX['genotype'] = ['FX' for i in range(len(fWT))]

fWTs = shuffle['WT']
fWTs = fWTs.to_frame(name = 'pct_correct')
fWTs['Subject'] = ['SU' + str(i + 1) for i in range(len(fWTs))]
fWTs['type'] = ['shuffle' for i in range(len(fWTs))]
fWTs['genotype'] = ['WT' for i in range(len(fWTs))]

fFXs = shuffle['FX']
fFXs = fFXs.to_frame(name = 'pct_correct')
fFXs['Subject'] = ['SU' + str(i + 1 + len(fWTs)) for i in range(len(fFXs))]
fFXs['type'] = ['shuffle' for i in range(len(fFXs))]
fFXs['genotype'] = ['FX' for i in range(len(fFXs))]


finalWT = pd.concat([fWT, fWTs], ignore_index = True)
finalFX = pd.concat([fFX, fFXs], ignore_index = True)


final_df = pd.concat([finalWT, finalFX], ignore_index = True)
# # Sort the DataFrame by the 'Subject' column
# df_sorted = final_df.sort_values(
#     by='Subject',
#     key=lambda x: x.str.extract(r'([A-Za-z]+)(\d+)').apply(
#         lambda y: y[0].zfill(2) + y[1].zfill(3),
#         axis=1
#     )
# )

#Run stats 

# import pingouin as pg

# # Repeated measures two-way ANOVA
# aov = pg.rm_anova(dv = 'pct_correct', within = ['type', 'genotype'], subject='Subject', data = final_df, detailed = True)
# print(aov)

# # Post-hoc pairwise comparisons
# posthoc = pg.pairwise_ttests(dv = 'pct_correct', within = ['type', 'genotype'], subject = 'Subject', data = final_df, padjust = 'bonf')
# print(posthoc)

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

final_df['genotype'] = final_df['genotype'].astype('category')
final_df['type'] = final_df['type'].astype('category')
# Model specification
model = mixedlm("pct_correct ~ type", final_df, groups=final_df["genotype"], re_formula="~type")

# Fit the model
result = model.fit()

# Print the results
print(result.summary())
