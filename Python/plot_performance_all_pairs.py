# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:40:15 2024

@author: shukl
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import savgol_filter
import seaborn as sns
import pandas as pd



# Load .mat files
socialW_perfWT = loadmat('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/WT_pair_normalized_transitions_by_session.mat')
matrix_WT = socialW_perfWT['WT_100']


socialW_perfFX = loadmat('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/FX_pair_normalized_transitions_by_session.mat')
matrix_FX = socialW_perfFX['FX_100']

# socialW_perfMixed = loadmat('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/mixed_pair_perf_rewardContWise.mat')
# matrix_mixed = socialW_perfMixed['Mixed_100']



#%% Plot pefromance on 100% reward contingency
window_size = 5
order = 3

# Replace zeros with NaNs
matrix_WT[matrix_WT == 0] = np.nan
matrix_FX[matrix_FX == 0] = np.nan
# matrix_mixed[matrix_mixed == 0] = np.nan



rows1, cols1 = matrix_WT.shape
rows2, cols2 = matrix_FX.shape
# rows3, cols3 = matrix_mixed.shape



# plt.figure()

# Plot columns and mean for the first matrix
for col in range(cols1):
    # smoothed_row = savgol_filter(matrix_WT[:, col], window_size, order, mode='constant', cval=np.nan)
    # plt.plot(range(1, rows1 + 1), smoothed_row, color='lightgray', linewidth=1, label=f'matrix_WT - Column {col + 1}')

    mean_values1 = np.nanmean(matrix_WT, axis=1)
    mean_values1 = mean_values1[~np.isnan(mean_values1)]
    sem_values1 = np.nanstd(matrix_WT, axis=1) / np.sqrt(np.sum(~np.isnan(matrix_WT), axis=1))
    sem_values1 = sem_values1[~np.isnan(sem_values1)]
    smoothed_mean1 =  savgol_filter( mean_values1, window_size, order)
    smoothed_sem1 = savgol_filter(sem_values1, window_size, order)
    # plt.plot(range(1, rows1 + 1), smoothed_mean1, 'k', linewidth=2, label='matrix_WT Mean')

# Plot columns and mean for the second matrix
for col in range(cols2):
    # smoothed_row = savgol_filter(matrix_FX[:, col], window_size, order, mode='constant', cval=np.nan)
    # plt.plot(range(1, rows2 + 1),smoothed_row, color='pink', linewidth=1, label=f'matrix_FX - Column {col + 1}')

    mean_values2 = np.nanmean(matrix_FX, axis=1)
    mean_values2 = mean_values2[~np.isnan(mean_values2)]
    sem_values2 = np.nanstd(matrix_FX, axis=1) / np.sqrt(np.sum(~np.isnan(matrix_FX), axis=1))
    sem_values2 = sem_values2[~np.isnan(sem_values2)]
    smoothed_mean2 =  savgol_filter( mean_values2, window_size, order)
    smoothed_sem2 = savgol_filter(sem_values2, window_size, order)
    # plt.plot(range(1, rows2 + 1), smoothed_mean2, 'red', linewidth=2, label='matrix_FX Mean')
    
# for col in range(cols3):
#     # smoothed_row = savgol_filter(matrix_mixed[:, col], window_size, order, mode='constant', cval=np.nan)
#     # plt.plot(range(1, rows3 + 1),smoothed_row, color='pink', linewidth=1, label=f'matrix_mixed - Column {col + 1}')

#     mean_values3 = np.nanmean(matrix_mixed, axis=1)
#     mean_values3 = mean_values3[~np.isnan(mean_values3)]
#     sem_values3 = np.nanstd(matrix_mixed, axis=1) / np.sqrt(np.sum(~np.isnan(matrix_mixed), axis=1))
#     sem_values1 = sem_values1[~np.isnan(sem_values1)]
#     smoothed_mean3 =  savgol_filter( mean_values3, window_size, order)
#     smoothed_sem3 = savgol_filter(sem_values3, window_size, order)
#     # plt.plot(range(1, rows3 + 1), smoothed_mean3, 'red', linewidth=2, label='matrix_mixed Mean')



    # plt.xlabel('Day #')
    # plt.ylabel('% Performance')
    # plt.title('Social W performance for all WT and FX pairs')
    # plt.ylim((0, 100))
    # plt.axhline((0, '--'))
    # plt.legend()
    # plt.show()



# Create a DataFrame for plotting
df_WT = pd.DataFrame({'Smoothed Mean1': smoothed_mean1, 'Smoothed SEM1': smoothed_sem1})
df_FX = pd.DataFrame({'Smoothed Mean2': smoothed_mean2, 'Smoothed SEM2': smoothed_sem2})
# df_mixed = pd.DataFrame({'Smoothed Mean3': smoothed_mean3, 'Smoothed SEM3': smoothed_sem3})



# Clip all dataframes to the same length (e.g., 30)
# clip_length = 30
# df_WT = df_WT.iloc[:clip_length, :]
# df_FX = df_FX.iloc[:clip_length, :]
# df_mixed = df_mixed.iloc[:clip_length, :]




# # Create a dataframe with NaN values for the gap between indices 30 and 35
# gap_data = pd.DataFrame({'Column1': [np.nan] * 5, 'Column2': [np.nan] * 5})

# # Concatenate existing clipped dataframes with the gap data
# df_WT = pd.concat([df_WT, gap_data, dfopq_WT], ignore_index=True)
# df_FX = pd.concat([df_FX, gap_data, dfopq_FX], ignore_index=True)
# df_mixed = pd.concat([df_mixed, gap_data, dfopq_mixed], ignore_index=True)

sns.set(style='whitegrid')
plt.figure(figsize=(10, 6), dpi = 300)

# Plot the smoothed mean
sns.lineplot(data=df_WT['Smoothed Mean1'], label='WT pairs')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(df_WT.index, df_WT['Smoothed Mean1'] - df_WT['Smoothed SEM1'],
                 df_WT['Smoothed Mean1'] + df_WT['Smoothed SEM1'], alpha=0.2)


# Plot the smoothed mean
sns.lineplot(data=df_FX['Smoothed Mean2'], label='FX pairs')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(df_FX.index, df_FX['Smoothed Mean2'] - df_FX['Smoothed SEM2'],
                 df_FX['Smoothed Mean2'] + df_FX['Smoothed SEM2'], alpha=0.2)
# # Plot the smoothed mean
# sns.lineplot(data=df_mixed['Smoothed Mean3'], label='Mixed pair')

# # Plot the shaded region representing the smoothed SEM
# plt.fill_between(df_mixed.index, df_mixed['Smoothed Mean3'] - df_mixed['Smoothed SEM3'],
#                  df_mixed['Smoothed Mean3'] + df_mixed['Smoothed SEM3'], alpha=0.2)

# Customize the plot
plt.xlabel('Session#', fontsize = 16)
plt.ylabel('Transitions /min', fontsize = 16)
plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((0, 5))
# plt.xlim((0, clip_length))
plt.legend()
plt.show()
# plt.savefig('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/socialW100_perf_by_session.svg', format='svg', transparent = True)


#%% Plot pefromqnce on 50% reward contingency

matrix_WT = socialW_perfWT['WT_50']
matrix_FX = socialW_perfFX['FX_50']
# matrix_mixed = socialW_perfMixed['Mixed_50']

window_size = 5
order = 3

# Replace zeros with NaNs
matrix_WT[matrix_WT == 0] = np.nan
matrix_FX[matrix_FX == 0] = np.nan
# matrix_mixed[matrix_mixed == 0] = np.nan



rows1, cols1 = matrix_WT.shape
rows2, cols2 = matrix_FX.shape
# rows3, cols3 = matrix_mixed.shape



# plt.figure()

# Plot columns and mean for the first matrix
for col in range(cols1):
    # smoothed_row = savgol_filter(matrix_WT[:, col], window_size, order, mode='constant', cval=np.nan)
    # plt.plot(range(1, rows1 + 1), smoothed_row, color='lightgray', linewidth=1, label=f'matrix_WT - Column {col + 1}')

    mean_values1 = np.nanmean(matrix_WT, axis=1)
    mean_values1 = mean_values1[~np.isnan(mean_values1)]
    sem_values1 = np.nanstd(matrix_WT, axis=1) / np.sqrt(np.sum(~np.isnan(matrix_WT), axis=1))
    sem_values1 = sem_values1[~np.isnan(sem_values1)]
    smoothed_mean1 =  savgol_filter( mean_values1, window_size, order)
    smoothed_sem1 = savgol_filter(sem_values1, window_size, order)
    # plt.plot(range(1, rows1 + 1), smoothed_mean1, 'k', linewidth=2, label='matrix_WT Mean')

# Plot columns and mean for the second matrix
for col in range(cols2):
    # smoothed_row = savgol_filter(matrix_FX[:, col], window_size, order, mode='constant', cval=np.nan)
    # plt.plot(range(1, rows2 + 1),smoothed_row, color='pink', linewidth=1, label=f'matrix_FX - Column {col + 1}')

    mean_values2 = np.nanmean(matrix_FX, axis=1)
    mean_values2 = mean_values2[~np.isnan(mean_values2)]
    sem_values2 = np.nanstd(matrix_FX, axis=1) / np.sqrt(np.sum(~np.isnan(matrix_FX), axis=1))
    sem_values2 = sem_values2[~np.isnan(sem_values2)]
    smoothed_mean2 =  savgol_filter( mean_values2, window_size, order)
    smoothed_sem2 = savgol_filter(sem_values2, window_size, order)
    # plt.plot(range(1, rows2 + 1), smoothed_mean2, 'red', linewidth=2, label='matrix_FX Mean')
    
# for col in range(cols3):
#     # smoothed_row = savgol_filter(matrix_mixed[:, col], window_size, order, mode='constant', cval=np.nan)
#     # plt.plot(range(1, rows3 + 1),smoothed_row, color='pink', linewidth=1, label=f'matrix_mixed - Column {col + 1}')

#     mean_values3 = np.nanmean(matrix_mixed, axis=1)
#     mean_values3 = mean_values3[~np.isnan(mean_values3)]
#     sem_values3 = np.nanstd(matrix_mixed, axis=1) / np.sqrt(np.sum(~np.isnan(matrix_mixed), axis=1))
#     sem_values3 = sem_values3[~np.isnan(sem_values3)]
#     smoothed_mean3 =  savgol_filter( mean_values3, window_size, order)
#     smoothed_sem3 = savgol_filter(sem_values3, window_size, order)
#     # plt.plot(range(1, rows3 + 1), smoothed_mean3, 'red', linewidth=2, label='matrix_mixed Mean')



    # plt.xlabel('Day #')
    # plt.ylabel('% Performance')
    # plt.title('Social W performance for all WT and FX pairs')
    # plt.ylim((0, 100))
    # plt.axhline((0, '--'))
    # plt.legend()
    # plt.show()



# Create a DataFrame for plotting
df_WT = pd.DataFrame({'Smoothed Mean1': smoothed_mean1, 'Smoothed SEM1': smoothed_sem1})
df_FX = pd.DataFrame({'Smoothed Mean2': smoothed_mean2, 'Smoothed SEM2': smoothed_sem2})
# df_mixed = pd.DataFrame({'Smoothed Mean3': smoothed_mean3, 'Smoothed SEM3': smoothed_sem3})



# Clip all dataframes to the same length (e.g., 30)
# clip_length = 30
# df_WT = df_WT.iloc[:clip_length, :]
# df_FX = df_FX.iloc[:clip_length, :]
# df_mixed = df_mixed.iloc[:clip_length, :]




# # Create a dataframe with NaN values for the gap between indices 30 and 35
# gap_data = pd.DataFrame({'Column1': [np.nan] * 5, 'Column2': [np.nan] * 5})

# # Concatenate existing clipped dataframes with the gap data
# df_WT = pd.concat([df_WT, gap_data, dfopq_WT], ignore_index=True)
# df_FX = pd.concat([df_FX, gap_data, dfopq_FX], ignore_index=True)
# df_mixed = pd.concat([df_mixed, gap_data, dfopq_mixed], ignore_index=True)

sns.set(style='whitegrid')
plt.figure(figsize=(10, 6), dpi = 300)

# Plot the smoothed mean
sns.lineplot(data=df_WT['Smoothed Mean1'], label='WT pairs')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(df_WT.index, df_WT['Smoothed Mean1'] - df_WT['Smoothed SEM1'],
                 df_WT['Smoothed Mean1'] + df_WT['Smoothed SEM1'], alpha=0.2)


# Plot the smoothed mean
sns.lineplot(data=df_FX['Smoothed Mean2'], label='FX pairs')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(df_FX.index, df_FX['Smoothed Mean2'] - df_FX['Smoothed SEM2'],
                 df_FX['Smoothed Mean2'] + df_FX['Smoothed SEM2'], alpha=0.2)
# # Plot the smoothed mean
# sns.lineplot(data=df_mixed['Smoothed Mean3'], label='Mixed pair')

# # Plot the shaded region representing the smoothed SEM
# plt.fill_between(df_mixed.index, df_mixed['Smoothed Mean3'] - df_mixed['Smoothed SEM3'],
#                  df_mixed['Smoothed Mean3'] + df_mixed['Smoothed SEM3'], alpha=0.2)

# Customize the plot
plt.xlabel('Session#', fontsize = 16)
plt.ylabel('Transitions /min', fontsize = 16)
plt.title('Normalized transitions (all pairs)', fontsize = 16)
plt.ylim((0, 5))
# plt.xlim((0, clip_length))
plt.legend()
plt.show()
# plt.savefig('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/social50_perf_by_session.png', format='png', transparent = True)


#%% Plot performance for opaque control experiments

socialWopq_perfWT = loadmat('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/socialWopq_perf_by_session_WT.mat')
matrixopq_WT = socialWopq_perfWT['socialW_perf_WT']


socialWopq_perfFX = loadmat('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/socialWopq_perf_by_session_FX.mat')
matrixopq_FX = socialWopq_perfFX['socialW_perf_FX']

# socialWopq_perfMixed = loadmat('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/socialWopq_perf_by_session_mixed.mat')
# matrixopq_mixed = socialWopq_perfMixed['socialW_perf_mixed']


matrixopq_WT[matrixopq_WT == 0] = np.nan
matrixopq_FX[matrixopq_FX == 0] = np.nan
# matrixopq_mixed[matrixopq_mixed == 0] = np.nan

rows4, cols4 = matrixopq_WT.shape
rows5, cols5 = matrixopq_FX.shape
# rows6, cols6 = matrixopq_mixed.shape


    
for col in range(cols4):
    # smoothed_row = savgol_filter(matrix_WT[:, col], window_size, order, mode='constant', cval=np.nan)
    # plt.plot(range(1, rows1 + 1), smoothed_row, color='lightgray', linewidth=1, label=f'matrix_WT - Column {col + 1}')

    mean_values4 = np.nanmean(matrixopq_WT, axis=1)
    sem_values4 = np.nanstd(matrixopq_WT, axis=1) / np.sqrt(np.sum(~np.isnan(matrixopq_WT), axis=1))
    smoothed_mean4 =  savgol_filter( mean_values4, window_size, order)
    smoothed_sem4 = savgol_filter(sem_values4, window_size, order)
    # plt.plot(range(1, rows1 + 1), smoothed_mean1, 'k', linewidth=2, label='matrix_WT Mean')

# Plot columns and mean for the second matrix
for col in range(cols5):
    # smoothed_row = savgol_filter(matrix_FX[:, col], window_size, order, mode='constant', cval=np.nan)
    # plt.plot(range(1, rows2 + 1),smoothed_row, color='pink', linewidth=1, label=f'matrix_FX - Column {col + 1}')

    mean_values5 = np.nanmean(matrixopq_FX, axis=1)
    sem_values5 = np.nanstd(matrixopq_FX, axis=1) / np.sqrt(np.sum(~np.isnan(matrixopq_FX), axis=1))
    smoothed_mean5 =  savgol_filter( mean_values5, window_size, order)
    smoothed_sem5 = savgol_filter(sem_values5, window_size, order)
    # plt.plot(range(1, rows2 + 1), smoothed_mean2, 'red', linewidth=2, label='matrix_FX Mean')
    
# for col in range(cols6):
#     # smoothed_row = savgol_filter(matrix_mixed[:, col], window_size, order, mode='constant', cval=np.nan)
#     # plt.plot(range(1, rows3 + 1),smoothed_row, color='pink', linewidth=1, label=f'matrix_mixed - Column {col + 1}')

#     mean_values6 = np.nanmean(matrixopq_mixed, axis=1)
#     sem_values6 = np.nanstd(matrixopq_mixed, axis=1) / np.sqrt(np.sum(~np.isnan(matrixopq_mixed), axis=1))
#     smoothed_mean6 =  savgol_filter( mean_values6, window_size, order)
#     smoothed_sem6 = savgol_filter(sem_values6, window_size, order)
#     # plt.plot(range(1, rows3 + 1), smoothed_mean3, 'red', linewidth=2, label='matrix_mixed Mean')




dfopq_WT = pd.DataFrame({'Smoothed Mean1': smoothed_mean4, 'Smoothed SEM1': smoothed_sem4})
dfopq_FX = pd.DataFrame({'Smoothed Mean2': smoothed_mean5, 'Smoothed SEM2': smoothed_sem5})
# dfopq_mixed = pd.DataFrame({'Smoothed Mean3': smoothed_mean6, 'Smoothed SEM3': smoothed_sem6})



# Plot
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6), dpi = 300)

# Plot the smoothed mean
sns.lineplot(data=dfopq_WT['Smoothed Mean1'], label='WT pairs')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(dfopq_WT.index, dfopq_WT['Smoothed Mean1'] - dfopq_WT['Smoothed SEM1'],
                 dfopq_WT['Smoothed Mean1'] + dfopq_WT['Smoothed SEM1'], alpha=0.2)


# Plot the smoothed mean
sns.lineplot(data=dfopq_FX['Smoothed Mean2'], label='FX pairs')

# Plot the shaded region representing the smoothed SEM
plt.fill_between(dfopq_FX.index, dfopq_FX['Smoothed Mean2'] - dfopq_FX['Smoothed SEM2'],
                 dfopq_FX['Smoothed Mean2'] + dfopq_FX['Smoothed SEM2'], alpha=0.2)
# # Plot the smoothed mean
# sns.lineplot(data=dfopq_mixed['Smoothed Mean3'], label='Mixed pair')

# # Plot the shaded region representing the smoothed SEM
# plt.fill_between(dfopq_mixed.index, dfopq_mixed['Smoothed Mean3'] - dfopq_mixed['Smoothed SEM3'],
#                  dfopq_mixed['Smoothed Mean3'] + dfopq_mixed['Smoothed SEM3'], alpha=0.2)

# Customize the plot
plt.xlabel('Day#', fontsize = 16)
plt.ylabel('% Performance', fontsize = 16)
plt.title('Social W Opaque Control performance for all pairs', fontsize = 16)
plt.ylim((0, 100))
# plt.xlim((-1, 9))
plt.legend()
plt.show()
# plt.savefig('C:/Users/shukl/OneDrive/Documents/MATLAB/Jadhav lab/socialWopq_perf_by_session.svg', format='svg', transparent = True)
