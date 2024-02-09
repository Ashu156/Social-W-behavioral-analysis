# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:43:22 2023

@author: ashutoshshukla

Requires the track_linearization library to be installed prior to running the 
script.Available at LorenFranklab Github
 
https://github.com/LorenFrankLab/track_linearization/tree/master

conda install -c franklab track_linearization

"""
#%% Import required libraries

import os
import numpy as np
import pandas as pd
from track_linearization import make_track_graph, plot_track_graph
from track_linearization import get_linearized_position
from track_linearization.utils import plot_graph_as_1D
import matplotlib.pyplot as plt
import cv2
import glob

from sklearn.impute import KNNImputer


#%%



# # Initialize the KNN imputer
# knn_imputer = KNNImputer(missing_values = np.nan, n_neighbors = 10, weights = 'distance', 
#                           metric='nan_euclidean') #, copy = True, add_indicator = True)
# position = position.to_numpy()

# # Perform imputation
# position = knn_imputer.fit_transform(position)

# plt.scatter(position[::, 0], position[::, 1], s = 10, zorder = 11)

# position = pd.DataFrame(position)



#%%

# cohort='Cohort-6'
dataFolder = "E:/Jadhav lab data/Behavior/Cohort 1/Social W/09-22-2023"
os.chdir(dataFolder)
os.listdir()

files = [os.path.join(dataFolder, file) for file in glob.glob('*.csv') if 'DLC' not in file]

directory = dataFolder

for csv_file in files:
    
    
    # Get the base name without extension
    
    if'-Rat1_corrected-SnoutTracking.csv'in csv_file and 'Rat1_position_linear' not in csv_file:
        base_name1 = os.path.splitext(os.path.basename(csv_file))[0][:-24] # remove the '_corrected-SnoutTracking'
        rat1_csv = pd.read_csv(csv_file)
        rat1_nodes_path = os.path.join(directory, f'{base_name1}.npy')
        rat1_nodes = np.load(rat1_nodes_path)
        print(csv_file)
        print(rat1_nodes_path)
        
        #%% Provide the edges

        edges = [
                (0, 1),  # connects nodes 0 and 1
                (2, 3),  # connects nodes 2 and 3
                (4, 5),  # connects nodes 4 and 5
                (1, 3),  # connects nodes 2 and 4
                (3, 5),  # connects nodes 4 and 5
                ]
        
        position1 = rat1_csv
        node_positions1 = rat1_nodes

        track_graph1 = make_track_graph(node_positions1, edges) 

        # Visulaize the nodes and edges of the W-track
        fig, ax = plt.subplots()
        plot_track_graph(track_graph1, ax=ax, draw_edge_labels=True)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xlabel("x-position")
        ax.set_ylabel("y-position")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        #%% Get the linearized position and save it as a .csv file

        position_linear = get_linearized_position(position = position1.values, track_graph = track_graph1)
        destination_path1 = os.path.join(directory, f'{base_name1}_position_linear.csv')
        position_linear.to_csv(destination_path1,index = False)
        print(destination_path1)


        position1 = position1.to_numpy()


        #%% Plot the scatter overlayed on the W-track skeleton

        fps = 29.015;

        start_min = 0;
        start_sec = 1;
        start_frame = int(np.rint((start_min*60 + start_sec)*fps))

        end_min = 10;
        end_sec = 50;
        end_frame = int(np.rint((end_min*60 + end_sec)*fps))

        fig, ax = plt.subplots()
        plot_track_graph(track_graph1, ax=ax, draw_edge_labels=True)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xlabel("x-position")
        ax.set_ylabel("y-position")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


        plt.scatter(position1[start_frame:end_frame, 0], position1[start_frame:end_frame, 1], s=10, zorder=11)



#%%

        edge_order = [(2, 3), (3, 5), (5, 4), (3, 1), (1, 0)] # for cohortAS1
        edge_spacing = [460, 0, 460, 0]
        
        # edge_order = [(2, 3), (3, 1), (1, 0), (3, 5), (5, 4)] # for cohortAS2
        # edge_spacing = [460, 0, 460, 0]
        
        fig, ax = plt.subplots(figsize=(7, 1))
        plot_graph_as_1D(track_graph1, edge_spacing=edge_spacing, edge_order=edge_order, ax=ax)


        #%%

        position_df = get_linearized_position(
            position = position1,
            track_graph = track_graph1,
            edge_spacing = edge_spacing,
            edge_order = edge_order,
        )
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(
            position_df.index[::],
            position_df.linear_position[::],
            s=10,
            zorder=2,
            clip_on=False,
        )
        
        
        ax.plot(
            position_df.index,
            position_df.linear_position,
            color="lightgrey",
            zorder=1,
            clip_on=False,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim((0, position_df.shape[0]))
        ax.set_ylabel("Position")
        ax.set_xlabel("Time")
        plot_graph_as_1D(
            track_graph1,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            ax=ax,
            axis="y",
            other_axis_start=position_df.index.max() + 1,
        )
        
        start_node_linear_position = 0.0
        ticks = []
        
        for ind, edge in enumerate(edge_order):
            end_node_linear_position = (
                start_node_linear_position + track_graph1.edges[edge]["distance"]
            )
            ax.axhline(start_node_linear_position, color="lightgrey", linestyle="--")
            ax.axhline(end_node_linear_position, color="lightgrey", linestyle="--")
            ticks.append(start_node_linear_position)
            ticks.append(end_node_linear_position)
            try:
                start_node_linear_position += (
                    track_graph1.edges[edge]["distance"] + edge_spacing[ind]
                )
            except IndexError:
                pass
        ax.set_yticks(ticks)
        
        
#%%      ######################### FOR Rat 2 #################################

for csv_file in files:        
      if '-Rat2_corrected-SnoutTracking.csv' in csv_file and 'position_linear' not in csv_file:
        base_name2 = os.path.splitext(os.path.basename(csv_file))[0][:-24] # remove the '_corrected-SnoutTracking'
        rat2_csv = pd.read_csv(csv_file)
        rat2_nodes_path = os.path.join(directory, f'{base_name2}.npy')
        rat2_nodes = np.load(rat2_nodes_path)
        # print(csv_file)
        # print(rat2_nodes_path)
    
   

        
   
        
        #%% Provide the edges

        edges = [
                (0, 1),  # connects nodes 0 and 1
                (2, 3),  # connects nodes 2 and 3
                (4, 5),  # connects nodes 4 and 5
                (1, 3),  # connects nodes 2 and 4
                (3, 5),  # connects nodes 4 and 5
                ]
        
        
        position2 = rat2_csv
        node_positions2 = rat2_nodes


        track_graph2 = make_track_graph(node_positions2, edges) 

        # Visulaize the nodes and edges of the W-track
        fig, ax = plt.subplots()
        plot_track_graph(track_graph2, ax=ax, draw_edge_labels=True)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xlabel("x-position")
        ax.set_ylabel("y-position")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        #%% Get the linearized position and save it as a .csv file

        position_linear = get_linearized_position(position = position2.values, track_graph = track_graph2)
        destination_path2 = os.path.join(directory, f'{base_name2}_position_linear.csv')
        position_linear.to_csv(destination_path2,index = False)
        print(destination_path2)

        position2 = position2.to_numpy()


        #%% Plot the scatter overlayed on the W-track skeleton

        fps = 29.015;

        start_min = 0;
        start_sec = 1;
        start_frame = int(np.rint((start_min*60 + start_sec)*fps))

        end_min = 10;
        end_sec = 50;
        end_frame = int(np.rint((end_min*60 + end_sec)*fps))

        fig, ax = plt.subplots()
        plot_track_graph(track_graph2, ax=ax, draw_edge_labels=True)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xlabel("x-position")
        ax.set_ylabel("y-position")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


        plt.scatter(position2[start_frame:end_frame, 0], position2[start_frame:end_frame, 1], s=10, zorder=11)



#%%

        edge_order = [(2, 3), (3, 5), (5, 4), (3, 1), (1, 0)] # for cohortAS1
        edge_spacing = [460, 0, 460, 0]
        
        # edge_order = [(2, 3), (3, 1), (1, 0), (3, 5), (5, 4)] # for cohortAS2
        # edge_spacing = [460, 0, 460, 0]
        
        fig, ax = plt.subplots(figsize=(7, 1))
        plot_graph_as_1D(track_graph2, edge_spacing=edge_spacing, edge_order=edge_order, ax=ax)


        #%%

        position_df = get_linearized_position(
            position = position2,
            track_graph = track_graph2,
            edge_spacing = edge_spacing,
            edge_order = edge_order,
        )
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(
            position_df.index[::],
            position_df.linear_position[::],
            s=10,
            zorder=2,
            clip_on=False,
        )
        
        
        ax.plot(
            position_df.index,
            position_df.linear_position,
            color="lightgrey",
            zorder=1,
            clip_on=False,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim((0, position_df.shape[0]))
        ax.set_ylabel("Position")
        ax.set_xlabel("Time")
        plot_graph_as_1D(
            track_graph2,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            ax=ax,
            axis="y",
            other_axis_start=position_df.index.max() + 1,
        )
        
        start_node_linear_position = 0.0
        ticks = []
        
        for ind, edge in enumerate(edge_order):
            end_node_linear_position = (
                start_node_linear_position + track_graph2.edges[edge]["distance"]
            )
            ax.axhline(start_node_linear_position, color="lightgrey", linestyle="--")
            ax.axhline(end_node_linear_position, color="lightgrey", linestyle="--")
            ticks.append(start_node_linear_position)
            ticks.append(end_node_linear_position)
            try:
                start_node_linear_position += (
                    track_graph2.edges[edge]["distance"] + edge_spacing[ind]
                )
            except IndexError:
                pass
        ax.set_yticks(ticks)


#%%

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal


# plt.ion()

# fig, ax = plt.subplots()
# ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
# ax.set_xlabel("x-position")
# ax.set_ylabel("y-position")
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# # x = np.linspace(0, 30)

# # position = np.concatenate(
# #     (
# #         np.stack((np.zeros_like(x), x[::-1]), axis=1),
# #         np.stack((x, np.zeros_like(x)), axis=1),
# #         np.stack((np.ones_like(x) * 30, x), axis=1)
# #     )
# # )
# # position += multivariate_normal(mean=0, cov=.05).rvs(position.shape)

# plt.scatter(position[::, 0], position[::, 1], s=10, zorder=11)
# # plt.scatter(position[50:100, 0], position[50:100, 1], s=10, zorder=11)
# # plt.scatter(position[100:, 0], position[100:, 1], s=10, zorder=11)

# x = plt.ginput()
# print(x)
# plt.show()