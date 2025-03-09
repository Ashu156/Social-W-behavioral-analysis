# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:51:09 2025

@author: shukl
"""
#%% Import libraries

import pandas as pd
import re
import os
import glob
from datetime import datetime
import numpy as np
import smoothfit
import matplotlib.pyplot as plt
import seaborn as sns
 
#%% Function for parsing behavioral events from .stateScriptLog files

def parse_social_events(file_path):
    
    """
    Function for parsing behavioral events
    """
    
  
    # Read logfile 
    with open(file_path, 'r') as f:
        log_lines = f.readlines()
        
    rat1_data = []
    rat2_data = []
    
    current_pokes = {}   # Temporary storage for ongoing pokes
    goalwell = None      # Store nextWell value for reward conditions
    
    for i, line in enumerate(log_lines):
        line = line.strip()
        
        """
        Detect poke events
        
        """
        poke_match = re.match(r"(\d+)\s+(Poke) in well([A-Z\d]+)", line)
        unpoke_match = re.match(r"(\d+)\s+(UnPoke) in well([A-Z\d]+)", line)
        
                    
        if poke_match:
            timestamp, well = float(poke_match.group(1))/1000.0, poke_match.group(3)
            # if well.isalpha():
            #     well = ord(well) - 64 # convert alphabetical well to numeric
            current_pokes[well] = timestamp  # Store start time
    
        elif unpoke_match:
            timestamp, well = float(unpoke_match.group(1))/1000.0, unpoke_match.group(3)
            # if well.isalpha():
            #     well = ord(well) - 64 # convert alphabetical well to numeric
            if well in current_pokes:
                start_time = current_pokes.pop(well)
                if well.isdigit():
                    rat1_data.append([start_time, timestamp, int(well), None, 0, 0, None])
                else:
    
                    rat2_data.append([start_time, timestamp, ord(well) - 64, None, 0, 0, None])
    
    """
    Parse match, reward and goal events
    
    """
    
    for i, line in enumerate(log_lines):
        line = line.strip()
        
        match_event = re.match(r"(\d+)\s+trycount = (\d+)", line)
        
        if match_event:
            # match_ts, rat2_well, rat1_well = float(match_event.group(1))/1000.0, ord(match_event.group(2)[0].upper()) - 64, int(match_event.group(2)[1])
            match_ts = float(match_event.group(1)) / 1000.0
            rat1_nearest_idx = min(range(len(rat1_data)), key=lambda i: abs(rat1_data[i][0] - match_ts))
            rat1_nearest_value = rat1_data[rat1_nearest_idx][0]
            
            rat2_nearest_idx = min(range(len(rat2_data)), key=lambda i: abs(rat2_data[i][0] - match_ts))
            rat2_nearest_value = rat2_data[rat2_nearest_idx][0]
            
            rat1_data[rat1_nearest_idx][4] = 1
            rat2_data[rat2_nearest_idx][4] = 1
            
        """
        Parse reward events
        
        """    
        reward_match = re.match(r"(\d+)\s+count = (\d+)", line)
            
        if reward_match:
            reward_ts = float(reward_match.group(1))/1000.0
            
            rat1_nearest_idx = min(range(len(rat1_data)), key=lambda i: abs(rat1_data[i][0] - reward_ts))
            # rat1_nearest_value = rat1_data[rat1_nearest_idx][0]
            
            rat2_nearest_idx = min(range(len(rat2_data)), key=lambda i: abs(rat2_data[i][0] - reward_ts))
            # rat2_nearest_value = rat2_data[rat2_nearest_idx][0]
            
            rat1_data[rat1_nearest_idx][5] = 1
            rat2_data[rat2_nearest_idx][5] = 1
                
        """
        Parse next target well
        
        """          
        nextwell_match = re.match(r"(\d+)\s+nextWell = (\d+)", line)
        if nextwell_match:
            nextwell_ts = float(nextwell_match.group(1))/1000.0
            goalwell = int(nextwell_match.group(2))
        
            rat1_nearest_idx = min(range(len(rat1_data)), key=lambda i: abs(rat1_data[i][0] - nextwell_ts))
            # rat1_nearest_value = rat1_data[rat1_nearest_idx][0]
        
            rat2_nearest_idx = min(range(len(rat2_data)), key=lambda i: abs(rat2_data[i][0] - nextwell_ts))
            # rat2_nearest_value = rat2_data[rat2_nearest_idx][0]
            
            rat1_data[rat1_nearest_idx][6] = goalwell
            rat2_data[rat2_nearest_idx][6] = goalwell
    
    """
    Convert to DataFrames 
    
    """
    columns = ['start', 'end', 'well', 'last well', 'match', 'reward', 'goal well']
    df_rat1 = pd.DataFrame(rat1_data, columns=columns)
    df_rat2 = pd.DataFrame(rat2_data, columns=columns)
    
    df_rat1['last well'] = df_rat1['well'].shift(1)
    df_rat2['last well'] = df_rat2['well'].shift(1)

    return df_rat1, df_rat2


#%% Function for parsing behavioral events from .stateScriptLog files (single rat experiments)

def parse_individual_events(file_path):
    
    """
    Function for parsing behavioral events
    """
    
  
    # Read logfile 
    with open(file_path, 'r') as f:
        log_lines = f.readlines()
        
    rat_data = []
    
    current_pokes = {}   # Temporary storage for ongoing pokes
    goalwell = None      # Store nextWell value for reward conditions
    
    for i, line in enumerate(log_lines):
        line = line.strip()
        
        """
        Detect poke events
        
        """
        poke_match = re.match(r"(\d+)\s+(Poke) in well([A-Z\d]+)", line)
        unpoke_match = re.match(r"(\d+)\s+(UnPoke) in well([A-Z\d]+)", line)
        
                    
        if poke_match:
            timestamp, well = float(poke_match.group(1))/1000.0, poke_match.group(3)
            # if well.isalpha():
            #     well = ord(well) - 64 # convert alphabetical well to numeric
            current_pokes[well] = timestamp  # Store start time
    
        elif unpoke_match:
            timestamp, well = float(unpoke_match.group(1))/1000.0, unpoke_match.group(3)
            # if well.isalpha():
            #     well = ord(well) - 64 # convert alphabetical well to numeric
            if well in current_pokes:
                start_time = current_pokes.pop(well)
                if well.isdigit():
                    rat_data.append([start_time, timestamp, int(well), None, 0, 0, None])
                else:
    
                    rat_data.append([start_time, timestamp, ord(well) - 64, None, 0, 0, None])
    
    """
    Parse match, reward and goal events
    
    """
    
    for i, line in enumerate(log_lines):
        line = line.strip()
        
        match_event = re.match(r"(\d+)\s+trycount = (\d+)", line)
        
        if match_event:
            # match_ts, rat2_well, rat1_well = float(match_event.group(1))/1000.0, ord(match_event.group(2)[0].upper()) - 64, int(match_event.group(2)[1])
            match_ts = float(match_event.group(1)) / 1000.0
            rat_nearest_idx = min(range(len(rat_data)), key=lambda i: abs(rat_data[i][0] - match_ts))
            rat_nearest_value = rat_data[rat_nearest_idx][0]
            
            
            rat_data[rat_nearest_idx][4] = 1
            
            
        """
        Parse reward events
        
        """    
        reward_match = re.match(r"(\d+)\s+count = (\d+)", line)
            
        if reward_match:
            reward_ts = float(reward_match.group(1))/1000.0
            
            rat_nearest_idx = min(range(len(rat_data)), key=lambda i: abs(rat_data[i][0] - reward_ts))
            
            

            
            rat_data[rat_nearest_idx][5] = 1
            
                
        """
        Parse next target well
        
        """          
        nextwell_match = re.match(r"(\d+)\s+nextWell = (\d+)", line)
        if nextwell_match:
            nextwell_ts = float(nextwell_match.group(1))/1000.0
            goalwell = int(nextwell_match.group(2))
        
            rat_nearest_idx = min(range(len(rat_data)), key=lambda i: abs(rat_data[i][0] - nextwell_ts))
            
            
            rat_data[rat_nearest_idx][6] = goalwell
            
    
    """
    Convert to DataFrames 
    
    """
    columns = ['start', 'end', 'well', 'last well', 'match', 'reward', 'goal well']
    df_rat = pd.DataFrame(rat_data, columns=columns)
    
    df_rat['last well'] = df_rat['well'].shift(1)


    return df_rat

#%% Test functions on a single file

# file_path = "E:/Jadhav lab data/Behavior/CohortAS2/Social W/50%/01-03-2024/log01-03-2024(11-FXM107-FXM105).stateScriptLog"
file_path = "E:/Jadhav lab data/Behavior/CohortAS6/Social W/50%/01-15-2025/log01-15-2025(3-FX115-FX117).stateScriptLog"
df_rat1, df_rat2 = parse_social_events(file_path)
# file_path = "E:/Jadhav lab data/Behavior/CohortAS5/ProbabilisticW/10-29-2024/log10-29-2024(1-FX68).stateScriptLog"
# file_path = "E:/Jadhav lab data/Behavior/CohortAS4/05-15-2024/log05-15-2024(3-FXM305).stateScriptLog"
# df_rat = parse_individual_events(file_path)

#%% Function for extracting transition sequences 

def combine_consecutive_wells(df):
    """
    Function to extract the transition sequence of well
    visits from rat poking data. It also caluclates duration
    a rat remains / dwells at a well.
    
    """
    # Calculate dwell time if not already present
    if 'poke_time' not in df.columns:
        df['poke_time'] = df['end'] - df['start']
    
    # Identify groups of consecutive rows with the same `thiswell`
    df['group'] = (df['well'] != df['well'].shift()).cumsum()
    
    # Group by `group` and aggregate
    result = (
        df.groupby('group')
        .agg(
            start=('start', 'first'),                # First start time
            end=('end', 'last'),                    # Last end time
            well=('well', 'first'),         # First thiswell (same for the group)
            dwell_time=('poke_time', 'sum'),       # Sum of dwell times in the group
        
        )
        .reset_index(drop=True)
    )
    
    return result

#%% Function for counting triplet occurrences from transition data

def count_valid_triplets(rat_df, column_name='well'):
    """
    Counts the number of unique and non-overlapping triplets from a given sequence in a DataFrame column.

    Parameters:
        rat_df (pd.DataFrame): The DataFrame containing the sequence data.
        column_name (str): The column name in the DataFrame containing the sequence.

    Returns:
        dict: A dictionary with triplet counts.
    """
    keys = ['121', '123', '131', '132', '212', '232', '313', '323']
    valid_triplets = {key: 0 for key in keys}

    # Define equivalent triplets mapping
    triplet_map = {
        '213': '123',
        '312': '123',
        '231': '132',
        '321': '132'
    }

    seq = rat_df[column_name]

    base = 3
    q, r = divmod(len(seq), base)

    for i in range(0, base * q, base):
        triplet = seq[i:i+3].astype(int).astype(str).str.cat()

        # Normalize equivalent triplets
        triplet = triplet_map.get(triplet, triplet)  # Convert to base representation if needed

        # Check if the triplet exists in the dictionary
        if triplet in valid_triplets:
            valid_triplets[triplet] += 1  # Increment count

    return valid_triplets


#%% Function for filtering following trials

def filter_following_trials(df_rat1, df_rat2):
    """
    Filters trials where both rats were at different wells,
    and the other rat has completed its well entry before the subject rat starts.
    """
    
    # Find transition sequence for each rat
    rat1_df = combine_consecutive_wells(df_rat1)
    rat2_df = combine_consecutive_wells(df_rat2)
    
    rat1_df["poke time"] = rat1_df["dwell_time"]
    rat2_df["poke time"] = rat2_df["dwell_time"]
    
    rat1_df["dwell_time"] = rat1_df["end"] - rat1_df["start"]
    rat2_df["dwell_time"] = rat2_df["end"] - rat2_df["start"]
    
    # Add 'last well' column
    rat1_df["last well"] = rat1_df['well'].shift(1)
    rat2_df["last well"] = rat2_df['well'].shift(1)
    
    # Track peer's last well at all times
    rat1_df["hiswell"] = np.nan
    rat2_df["hiswell"] = np.nan
    
    # Assign last poke-in event of partner
    
    # For rat1
    for tr in range(len(rat1_df)):
        his_last_poke = rat2_df[rat2_df["start"] < rat1_df["start"].iloc[tr]].index.max()
        if his_last_poke is not None and not np.isnan(his_last_poke):
            rat1_df.at[tr, "hiswell"] = rat2_df.at[his_last_poke, "well"]
    
    # For rat2
    for tr in range(len(rat2_df)):
        his_last_poke = rat1_df[rat1_df["start"] < rat2_df["start"].iloc[tr]].index.max()
        if his_last_poke is not None and not np.isnan(his_last_poke):
            rat2_df.at[tr, "hiswell"] = rat1_df.at[his_last_poke, "well"]
    
    # Add last well information
    rat1_df["last well"] = rat1_df["well"].shift(1)
    rat2_df["last well"] = rat2_df["well"].shift(1)
        
    # Identifying candidates separately for rat1 and rat2
    rat1_df["following trial"] = (rat1_df["well"] != rat1_df["last well"]) & (rat1_df["hiswell"] != rat1_df["last well"]) & (rat1_df["hiswell"] != rat1_df["hiswell"].shift(1)) & (pd.notna(rat1_df["hiswell"]))
    rat2_df["following trial"] = (rat2_df["well"] != rat2_df["last well"]) & (rat2_df["hiswell"] != rat2_df["last well"]) & (rat2_df["hiswell"] != rat2_df["hiswell"].shift(1)) & (pd.notna(rat2_df["hiswell"]))
    
    # Calculate transit time between arms
    rat1_df["transit_time"] = rat1_df["start"].iloc[1:].reset_index(drop=True) - rat1_df["end"].iloc[:-1].reset_index(drop=True)
    rat2_df["transit_time"] = rat2_df["start"].iloc[1:].reset_index(drop=True) - rat2_df["end"].iloc[:-1].reset_index(drop=True)
    
    # Check whether following trial is a match and rewarded event

    # For rat1 
    for tr in range(len(rat1_df)):
        if rat1_df["following trial"].iloc[tr]:
            index = df_rat1.index[df_rat1["start"] == rat1_df["start"].iloc[tr]].max()
            if index is not None and not np.isnan(index):
                rat1_df.at[tr, "match"] = df_rat1.at[index, "match"]
                rat1_df.at[tr, "reward"] = df_rat1.at[index, "reward"]
    # For rat2 
    for tr in range(len(rat2_df)):
        if rat2_df["following trial"].iloc[tr]:
            index = df_rat2.index[df_rat2["start"] == rat2_df["start"].iloc[tr]].max()
            if index is not None and not np.isnan(index):
                rat2_df.at[tr, "match"] = df_rat2.at[index, "match"]
                rat2_df.at[tr, "reward"] = df_rat2.at[index, "reward"]

    
    return rat1_df, rat2_df

#%% Batch process data from social W cohorts
 
# Define the dtype for the structured array
dtype = np.dtype([
    ('name', 'U100'),      # File name
    ('folder', 'U255'),    # Folder path
    ('date', 'U50'),       # Extracted date
    ('cohortnum', 'U50'),  # Cohort number
    ('runum', 'U50'),      # Run number
    ('ratnums', 'U50'),    # Rat numbers as string
    ('ratnames', 'O'),     # List of rat names (object)
    ('ratsamples', 'O'),   # Tuple of two DataFrames (object)
    ('match', 'O'),        # matches for rats
    ('reward', 'O'),       # rewards for rats
    ('nTransitions', 'O'), # transitions for rats
    ('perf', 'U50'),       # performance metric for pair
    ('duration', 'U50'),   # session duration
    ('transition sequence', 'O'), # transition sequence
    ('triplet counts', 'O') # triplet counts
])


# Function to extract data from filename
def parse_filename(filename):
    """
    Function to extract data from filename
    """
    match = re.search(r"log(\d{2}-\d{2}-\d{4})\((\d+)-([A-Z]+\d+)-([A-Z]+\d+)\)\.stateScriptLog", filename)
    if match:
        date, runum, rat1, rat2 = match.groups()
        return date, runum, f"{rat1},{rat2}", [rat1, rat2]
    return None, None, None, None

# Function to load .stateScriptLog files
def process_social_stateScriptLogs(base_dir):
    
    """
    Function to extract data from filename
    """
    
    struct_data = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".stateScriptLog"):
                full_path = os.path.join(root, file)
                date, runum, ratnums, ratnames = parse_filename(file)

                # Dummy DataFrames for example
                df_rat1, df_rat2 = parse_social_events(full_path)
                
                match1 = df_rat1['match'].sum()
                match2 = df_rat2['match'].sum()
                
                reward1 = df_rat1['reward'].sum()
                reward2 = df_rat2['reward'].sum()
                
                session_length = (max(df_rat1['end'].max(), df_rat2['end'].max()) 
                                  - min(df_rat1['start'].min(), df_rat2['start'].min())) / 60
                
                rat1_df = combine_consecutive_wells(df_rat1)
                rat2_df = combine_consecutive_wells(df_rat2)
                
                count1 = count_valid_triplets(rat1_df, column_name='well')
                count2 = count_valid_triplets(rat2_df, column_name='well')
                count = [count1, count2]
                
                seq1 = rat1_df['well'].astype(int).astype(str).str.cat()
                seq2 = rat2_df['well'].astype(int).astype(str).str.cat()
                seq = [seq1, seq2]
                
                transitions1 = len(rat1_df)
                transitions2 = len(rat2_df)
                nTransitions = [transitions1, transitions2]
                
                if np.any((transitions1 > 150) | (transitions2 > 150)):
                    transitions1 = np.nan
                    transitions2 = np.nan
                    nTransitions = [transitions1, transitions2]
                    perf = np.nan
                else:
                    perf = 200*(match1 / np.sum(nTransitions))


                struct_data.append((
                    file,  # name
                    root,  # folder
                    date,  # date
                    root.split("/")[-2],  # cohortnum (assuming one level up)
                    runum,  # runum
                    ratnums,  # ratnums
                    ratnames,  # ratnames
                    (df_rat1, df_rat2),  # ratsamples (tuple of DataFrames)
                    (match1, match2),    # matches for rats
                    (reward1, reward2),  # rewards for rats
                    nTransitions,        # transitions for rats
                    perf,                # pair performance
                    session_length,      # session duration
                    seq,                 # transition sequences
                    count                # triplet counts
                ))
                
                
                # Initialize an empty list to store the data
                structured_data = []

                for entry in struct_data:
                    data_entry = {
                        'name': entry[0],
                        'folder': entry[1],
                        'date': entry[2],
                        'cohortnum': entry[3],
                        'runum': entry[4],
                        'ratnum': entry[5],
                        'ratnames': entry[6],
                        'ratsamples': entry[7],  # DataFrame
                        'match': entry[8],
                        'reward': entry[9],
                        'nTransitions': entry[10],
                        'perf': entry[11],
                        'duration': entry[12],
                        'transition_sequence': entry[13],  # List
                        'triplet counts': entry[14]  # Dict
                    }
                    structured_data.append(data_entry)
    
    return structured_data

# Define base directory
base_dir = "E:/Jadhav lab data/Behavior/CohortAS6/Social W/OpaqueControl"

# Load the structured array
data = process_social_stateScriptLogs(base_dir)

# Find indices with date as None
index = []
for i in range(len(data)):  
   if data[i]['date'] is None:
       index.append(i)

# Exclude those entries 
new_data = [ele for idx, ele in enumerate(data) if idx not in index]

data = new_data

# Sort by converted date and integer run number
sorted_data = sorted(data, key=lambda x: (datetime.strptime(x['date'], "%m-%d-%Y"), int(x['runum'])))

# Dictionary to store performance values for each rat
pair_performance = {}

for i in range(len(sorted_data)):
    pair = tuple(sorted(sorted_data[i]['ratnames']))  # Extract rat name
    perf_value = sorted_data[i]['perf']  # Extract performance value

    if pair not in pair_performance:
        pair_performance[pair] = []
    
    pair_performance[pair].append(perf_value)  # Append performance value

# Convert lists to NumPy arrays
for pair in pair_performance:
    pair_performance[pair] = np.array(pair_performance[pair])


#%% Parse data separately for each pair and extract performance, matches, etc.

from collections import defaultdict
import numpy as np

# Dictionary to store data for each unique rat pair
pairwise_data = defaultdict(list)

# Iterate through sorted_data and group by unique rat pair
for entry in sorted_data:
    rat_pair = tuple(sorted(entry['ratnames']))  # Ensure order consistency by sorting the pair
    pairwise_data[rat_pair].append(entry)  # Store the entry in the corresponding pair's list

# Convert lists to NumPy arrays
pairwise_arrays = {pair: np.array(entries, dtype=object) for pair, entries in pairwise_data.items()}
    
    
# Dictionary to store performance data for each unique rat pair
pairwise_perf = {}
pairwise_match_rate = {}
pairwise_reward_rate = {}


for entry in sorted_data:
    ratnames = tuple(sorted(entry['ratnames']))  # entry[6] contains ['Rat1', 'Rat2']
    perf = entry['perf']  # Assuming perf is the second last column
    duration = float(entry['duration'])
    match_rate = entry['match'][0] / duration
    reward_rate = entry['reward'][0] / duration

    if ratnames not in pairwise_perf:
        pairwise_perf[ratnames] = []
        pairwise_match_rate[ratnames] = []
        pairwise_reward_rate[ratnames] = []

    pairwise_perf[ratnames].append(perf)
    pairwise_match_rate[ratnames].append(match_rate)
    pairwise_reward_rate[ratnames].append(reward_rate)

# Convert to NumPy arrays
pairwise_perf = {pair: np.array(perfs, dtype=float) for pair, perfs in pairwise_perf.items()}
pairwise_match_rate = {pair: np.array(perfs, dtype=float) for pair, perfs in pairwise_match_rate.items()}
pairwise_reward_rate = {pair: np.array(perfs, dtype=float) for pair, perfs in pairwise_reward_rate.items()}

#%% Parse performance, match rate, etc. for each rat separately

# Dictionary to store performance data for each individual rat
ratwise_perf = {}
ratwise_match_rate = {}
ratwise_reward_rate = {}
ratwise_transition_rate = {}
ratwise_triplet_count = {}

for entry in sorted_data:
    rat1, rat2 = entry['ratnames']  # entry[6] contains ['Rat1', 'Rat2']
    perf = entry['perf']  # Assuming perf is the second last column
    duration = float(entry['duration'])

    # Extract values separately for each rat
    match_rates = entry['match'] # Assuming entry['match'] is a list/array [rat1_match, rat2_match]
    reward_rates = entry['reward']
    transition_rates = entry['nTransitions']
    triplet_counts = entry['triplet counts']

    for i, rat in enumerate([rat1, rat2]):  # Iterate over both rats with index
        if rat not in ratwise_perf:
            ratwise_perf[rat] = []
            ratwise_match_rate[rat] = []
            ratwise_reward_rate[rat] = []
            ratwise_transition_rate[rat] = []
            ratwise_triplet_count[rat] = []

        ratwise_perf[rat].append(perf)  # Assuming performance is the same for both rats
        ratwise_match_rate[rat].append(match_rates[i] / duration)  # Extract individual match rate
        ratwise_reward_rate[rat].append(reward_rates[i] / duration)  # Extract individual reward rate
        ratwise_transition_rate[rat].append(transition_rates[i] / duration)  # Extract individual transition rate
        ratwise_triplet_count[rat].append(triplet_counts[i])
        
# Convert to NumPy arrays
ratwise_perf = {rat: np.array(perfs, dtype=float) for rat, perfs in ratwise_perf.items()}
ratwise_match_rate = {rat: np.array(rates, dtype=float) for rat, rates in ratwise_match_rate.items()}
ratwise_reward_rate = {rat: np.array(rates, dtype=float) for rat, rates in ratwise_reward_rate.items()}
ratwise_transition_rate = {rat: np.array(rates, dtype=float) for rat, rates in ratwise_transition_rate.items()}


# Convert specific parameters to numpy arrays 
data_dict = ratwise_match_rate
# Sort dictionary keys in ascending order
sorted_keys = sorted(data_dict.keys())

# Find the maximum length among all arrays
max_length = max(len(arr) for arr in data_dict.values())

# Pad arrays with NaN to match the max length
padded_dict = {key: np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for key, arr in data_dict.items()}

# Convert to DataFrame
df = pd.DataFrame(padded_dict).to_numpy()


#%% Get optimal triplet count as plot heatmap for rats separately

import seaborn as sns
import matplotlib.pyplot as plt

ratwise_optimal_triplet_ratio = {}

# Convert list of dictionaries to DataFrame, filling missing values with 0
for key in ratwise_triplet_count.keys():
    
    df = pd.DataFrame(ratwise_triplet_count[key]).T
    ratwise_optimal_triplet_ratio[key]= []
    

    ratio = ((df.loc['123', :] + df.loc['132', :]) / df.sum()).to_numpy()
    ratwise_optimal_triplet_ratio[key].append(ratio)
    
    # Plot heatmap
    plt.figure()
    sns.heatmap(df, annot = False, cmap = "coolwarm", cbar = True, vmin = 0.0, vmax = 40.0)
    
    # Labels and title
    plt.xlabel("Session")
    plt.ylabel("Triplet")
    plt.title(key)
    
    plt.show()

#%% Plot optimal_triplet ratio for each rat

for key in ratwise_optimal_triplet_ratio.keys():  
    tempData = ratwise_optimal_triplet_ratio[key][0]
    # Smooth performance for WT group
    basis, coeffs = smoothfit.fit1d(np.arange(0,len(tempData), 1), tempData, 0, len(tempData), 1000, degree=1, lmbda=5.0e-0)
    # plt.plot(tempData)
    plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]],  linestyle = "-", label = key)
    plt.plot()
    plt.legend()
    
#%% Mutual information between choice sequences


from collections import defaultdict
import numpy as np
from sklearn.feature_selection import mutual_info_regression

# Dictionary to store data for each unique rat pair
pairwise_data = defaultdict(list)

# Iterate through sorted_data and group by unique rat pair
for entry in sorted_data:
    rat_pair = tuple(sorted(entry['ratnames']))  # Ensure order consistency by sorting the pair
    pairwise_data[rat_pair].append(entry)  # Store the entry in the corresponding pair's list

# Convert lists to NumPy arrays
pairwise_arrays = {pair: np.array(entries, dtype=object) for pair, entries in pairwise_data.items()}
    
    
# Dictionary to store performance data for each unique rat pair
pairwise_mi = {}



for entry in sorted_data:
    ratnames = tuple(sorted(entry['ratnames']))  # entry[6] contains ['Rat1', 'Rat2']
    rat1 = entry['ratsamples'][0]  # Assuming perf is the second last column
    rat2 = entry['ratsamples'][1]  # Assuming perf is the second last column
    
    rat1_df = combine_consecutive_wells(rat1)
    rat2_df = combine_consecutive_wells(rat2)
    
    # For rat1
    for tr in range(len(rat1_df)):
        his_last_poke = rat2_df[rat2_df["start"] < rat1_df["start"].iloc[tr]].index.max()
        if his_last_poke is not None and not np.isnan(his_last_poke):
            rat1_df.at[tr, "hiswell"] = rat2_df.at[his_last_poke, "well"]
            
    if rat1_df.shape[0] > 5:
                # Ensure 'thiswell' and 'hiswell' columns are numeric
                
                try:
                    # Drop rows where either 'well' or 'hiswell' has NaN
                    filtered_df = rat1_df.dropna(subset=['well', 'hiswell'])

                    # Convert to NumPy arrays
                    a = filtered_df['well'].to_numpy().reshape(-1, 1).astype(float)
                    target = filtered_df['hiswell'].to_numpy().astype(float)
                    
                    
                    # Calculate mutual information using mutual_info_regression
                    mi = mutual_info_regression(a, target, random_state = 42)
                    
                except ValueError as e:
                    print(f"Error processing file {entry['name']}: {e}")
                    mi = [np.nan]
    else:
        mi = [np.nan]      
    

    if ratnames not in pairwise_mi:
        pairwise_mi[ratnames] = []
        

    pairwise_mi[ratnames].append(mi)
    

# Convert to NumPy arrays
pairwise_mi = {pair: np.array(mis, dtype=float) for pair, mis in pairwise_mi.items()}


#%% Leader-follower relation

# Initialize dictionaries
ratwise_leader = {}
ratwise_pSwitch = {}
ratwise_lag = {}  # New dictionary to store the pairwise lag between rat1 and rat2 for each match

# Iterate through each entry in sorted_data
for entry in sorted_data:
    rat1_df, rat2_df = entry['ratsamples']
    rat1, rat2 = entry['ratnames']  # Assuming 'rats' key stores their names

    if rat1_df['match'].sum() == rat2_df['match'].sum():
        idx1 = rat1_df.index[rat1_df['match'] == 1]
        idx2 = rat2_df.index[rat2_df['match'] == 1]
        
        r1 = np.zeros(len(idx1))
        r2 = np.zeros(len(idx2))
        pairwise_lags = []  # List to store pairwise lag for each match
        
        def last_arrival_time(df, row_index):
            """
            Function for checking when a rat arrived at a well 
            """
            current_well = df.loc[row_index, 'well']
    
            # Find previous occurrences of the same well
            previous_rows = df.loc[:row_index - 1]  # Select rows before current index
            last_occurrence = previous_rows[previous_rows['well'] == current_well]
    
            if not last_occurrence.empty:
                return last_occurrence.iloc[-1]['start']  # Return the last arrival time
            else:
                return None  # No previous visit to the well
        

        def pSwitch_leader(binary_vector):
            """
            Computes the probability that a 1 is followed by a 0 in a binary vector.
        
            Parameters:
            binary_vector (array-like): A 1D binary array (list or NumPy array).
        
            Returns:
            float: Probability that 1 is followed by 0.
            """
            binary_vector = np.asarray(binary_vector)  # Ensure input is a NumPy array
            
            if len(binary_vector) < 2:
                return 0  # If there are less than 2 elements, return 0 probability
            
            transitions = (binary_vector[:-1] == 1) & (binary_vector[1:] == 0)
            count_1s = np.sum(binary_vector[:-1] == 1)  # Count total 1s excluding last position
            
            return np.sum(transitions) / count_1s if count_1s > 0 else 0


        for match in range(len(idx1)): 
            r1_arrival = last_arrival_time(rat1_df, idx1[match])
            r2_arrival = last_arrival_time(rat2_df, idx2[match])
            
            # r1_arrival = rat1_df.iloc[idx1[match]]['start']
            # r2_arrival = rat2_df.iloc[idx2[match]]['start']
            
            if r1_arrival is None:
                r1_arrival = rat1_df.loc[idx1[match], 'start']
                
            if r2_arrival is None:
                r2_arrival = rat2_df.loc[idx2[match], 'start']
            
            # Calculate the absolute lag (time difference) between the two rats' arrivals
            lag = abs(r1_arrival - r2_arrival)
            pairwise_lags.append(lag)  # Store only the lag value
            
            # Determine which rat is the leader based on arrival times
            if r1_arrival < r2_arrival:
                r1[match] = 1
            elif r1_arrival > r2_arrival:
                r2[match] = 1
            
        # Convert binary leader vectors into proportions
        r1_Lead = np.sum(r1) / len(r1)
        r2_Lead = np.sum(r2) / len(r2)
        
        # Calculate probability of leadership switch between consecutive matches
        r1_pSwitch = pSwitch_leader(r1)
        r2_pSwitch = pSwitch_leader(r2)
        
        # Compute the mean arrival lag for the pair
        mean_arrival_lag = np.mean(np.array(pairwise_lags))
        
        # Assign the arrays to the correct rat IDs in a single loop
        for rat, r_lead, r_pSwitch in zip(entry['ratnames'], [r1_Lead, r2_Lead], [r1_pSwitch, r2_pSwitch]):
            ratwise_leader.setdefault(rat, []).append(r_lead)
            ratwise_pSwitch.setdefault(rat, []).append(r_pSwitch)
            ratwise_lag.setdefault(rat, []).append(mean_arrival_lag)  # Store the same mean arrival lag for both rats

# Convert lists to numpy arrays
ratwise_leader = {rat: np.array(values) for rat, values in ratwise_leader.items()}
ratwise_pSwitch = {rat: np.array(values) for rat, values in ratwise_pSwitch.items()}
ratwise_lag = {rat: np.array(values) for rat, values in ratwise_lag.items()}

#%% Compute departure latency

# Initialize dictionary
ratwise_departure = {}

# Iterate through each entry in sorted_data
for entry in sorted_data:
    rat1_df, rat2_df = entry['ratsamples']
    rat1, rat2 = entry['ratnames']  # Assuming 'ratnames' key stores their names

    def compute_latency_to_departure(df):
        latencies = []
        
        for idx, row in df[df['match'] == 1].iterrows():
            start_time = row['start']
            well = row['well']
            
            # Find the next occurrence where the rat transitions to a different well
            transition_row = df[(df.index > idx) & (df['well'] != well) & (df['start'] > start_time)].head(1)
            
            if not transition_row.empty:
                last_same_well_idx = transition_row.index[0] - 1  # Get the previous index
                if last_same_well_idx in df.index:
                    unpoke_row = df.loc[last_same_well_idx]  # Get the last row where the rat was in the same well
                    
                    end_time = unpoke_row['end']
                    latency = end_time - start_time

                latencies.append(latency)  # Store just latency
        
        return np.array(latencies) if latencies else np.array([])

    # Compute and store latencies as NumPy arrays for each rat
    for rat, df in zip([rat1, rat2], [rat1_df, rat2_df]):
        if rat not in ratwise_departure:
            ratwise_departure[rat] = []  # Initialize empty list for this rat
        
        latency_array = compute_latency_to_departure(df)
        mean_latency = np.mean(latency_array)
        
        if latency_array.size > 0:  # Only store non-empty arrays
            ratwise_departure[rat].append(mean_latency)

# Convert lists to numpy arrays
ratwise_departure = {rat: np.array(values) for rat, values in ratwise_departure.items()}


#%% Extract poke time, dwell time and their ratio for a cohort

# Initialize dictionaries
ratwise_poke = {}
ratwise_dwell = {}
ratwise_ratio = {}

# Iterate through each entry in sorted_data
for entry in sorted_data:
    rat1_df, rat2_df = entry['ratsamples']
    rat1, rat2 = entry['ratnames']  # Assuming 'ratnames' key stores their names
    
    # Apply the filtering function
    rat1_df, rat2_df = filter_following_trials(rat1_df, rat2_df)

    # Store required data for each rat
    for rat, df in zip([rat1, rat2], [rat1_df, rat2_df]):
        # Initialize lists for each rat if not already present
        if rat not in ratwise_poke:
            ratwise_poke[rat] = []
            ratwise_dwell[rat] = []
            ratwise_ratio[rat] = []
        
        # Convert to numpy arrays and store
        poke_time = np.nanmean(np.array(df["poke time"]))
        dwell_time = np.nanmean(np.array(df["dwell_time"]))
        poke_dwell_ratio = np.divide(poke_time, dwell_time, where=dwell_time != 0)  # Avoid division by zero
        
        ratwise_poke[rat].append(poke_time)
        ratwise_dwell[rat].append(dwell_time)
        ratwise_ratio[rat].append(poke_dwell_ratio)

# Convert lists to numpy arrays
ratwise_poke = {rat: np.array(values) for rat, values in ratwise_poke.items()}
ratwise_dwell = {rat: np.array(values) for rat, values in ratwise_dwell.items()}
ratwise_ratio = {rat: np.array(values) for rat, values in ratwise_ratio.items()}

#%% Extract data from cohort ER1

from collections import defaultdict

# Define the dtype for the structured array
dtype = np.dtype([
    ('name', 'U100'),      # File name
    ('folder', 'U255'),    # Folder path
    ('date', 'U50'),       # Extracted date
    ('cohortnum', 'U50'),  # Cohort number
    ('runum', 'U50'),      # Run number
    ('ratnames', 'O'),     # List of rat names (object)
    ('ratsamples', 'O'),   # Tuple of two DataFrames (object)
    ('match', 'O'),        # matches for rats
    ('reward', 'O'),       # rewards for rats
    ('nTransitions', 'O'), # transitions for rats
    ('perf', 'U50'),       # performance metric for pair
    ('duration', 'U50'),   # session duration
    ('transition sequence', 'O'), # transition sequence
    ('triplet counts', 'O') # triplet counts
])


# Function to extract data from filename
def parse_filename(filename):
    """
    Function to extract data from filename.
    """
    match = re.search(r"log(\d{2}-\d{2}-\d{4})\((\d+)-([A-Z]+\d+)-([A-Z]+\d+)\)", filename)
    
    if match:
        date, runum, rat1, rat2 = match.groups()
        return date, runum, f"{rat1},{rat2}", [rat1, rat2]
    
    return None, None, None, None


# Function to load .stateScriptLog files
def process_social_csvFiles(base_dir):
    
    """
    Function to extract data from filename
    """
    
    struct_data = []
    
    # Regex pattern to extract (date, run_number, rat1, rat2, rat_in_filename)
    pattern = re.compile(r"log(\d{2}-\d{2}-\d{4})\((\d+)-([A-Z]+\d+)-([A-Z]+\d+)\)-Rat([A-Z]+\d+)\.csv")

    # Dictionary to store grouped files by (date, run_number)
    file_groups = defaultdict(list)

    # Step 1: Recursively collect files based on (date, run number)
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                match = pattern.match(file)
                if match:
                    date, runum, rat1, rat2, rat_in_filename = match.groups()
                    key = (date, runum)  # Group files by (date, run_number)
                    full_path = os.path.join(root, file)  # Store full path
                    file_groups[key].append(full_path)
    
    # Step 2: Process paired files
    for (date, runum), file_list in file_groups.items():
        if len(file_list) == 2:  # Ensure there are exactly 2 paired files
            file_rat1, file_rat2 = sorted(file_list)  # Sort to maintain order
            
            rat1 = file_rat1[-7:-4]
            rat2 = file_rat2[-7:-4]
            ratnames = [rat1, rat2]
            
            # Load DataFrames
            df_rat1 = pd.read_csv(file_rat1)
            df_rat2 = pd.read_csv(file_rat2)
            
            df_rat1.rename(columns={'thiswell': 'well', 'Reward': 'reward'}, inplace=True)
            df_rat2.rename(columns={'thiswell': 'well', 'Reward': 'reward'}, inplace=True)
            

    
            print(f"📌 Processing paired files for Date: {date}, Run: {runum}")
            print(f"  🐀 Rat 1 file: {file_rat1}")
            print(f"  🐀 Rat 2 file: {file_rat2}")

                
            match1 = df_rat1['match'].sum()
            match2 = df_rat2['match'].sum()
            
            reward1 = df_rat1['reward'].sum()
            reward2 = df_rat2['reward'].sum()
            
            session_length = (max(df_rat1['end'].max(), df_rat2['end'].max()) 
                              - min(df_rat1['start'].min(), df_rat2['start'].min())) / 60
            
            rat1_df = combine_consecutive_wells(df_rat1)
            rat2_df = combine_consecutive_wells(df_rat2)
            
            count1 = count_valid_triplets(rat1_df, column_name='well')
            count2 = count_valid_triplets(rat2_df, column_name='well')
            count = [count1, count2]
            
            seq1 = rat1_df['well'].astype(int).astype(str).str.cat()
            seq2 = rat2_df['well'].astype(int).astype(str).str.cat()
            seq = [seq1, seq2]
            
            transitions1 = len(rat1_df)
            transitions2 = len(rat2_df)
            nTransitions = [transitions1, transitions2]
            
            if np.any((transitions1 > 150) | (transitions2 > 150)):
                transitions1 = np.nan
                transitions2 = np.nan
                nTransitions = [transitions1, transitions2]
                perf = np.nan
            else:
                perf = 200*(match1 / np.sum(nTransitions))


            struct_data.append((
                file_rat1[:-11],  # name
                root,  # folder
                date,  # date
                root.split("/")[-2],  # cohortnum (assuming one level up)
                runum,  # runum
                ratnames,  # ratnames
                (df_rat1, df_rat2),  # ratsamples (tuple of DataFrames)
                (match1, match2),    # matches for rats
                (reward1, reward2),  # rewards for rats
                nTransitions,        # transitions for rats
                perf,                # pair performance
                session_length,      # session duration
                seq,                 # transition sequences
                count                # triplet counts
            ))
            
            # Initialize an empty list to store the data
            structured_data = []

            for entry in struct_data:
                data_entry = {
                    'name': entry[0],
                    'folder': entry[1],
                    'date': entry[2],
                    'cohortnum': entry[3],
                    'runum': entry[4],
                    'ratnames': entry[5],
                    'ratsamples': entry[6],  # DataFrame
                    'match': entry[7],
                    'reward': entry[8],
                    'nTransitions': entry[9],
                    'perf': entry[10],
                    'duration': entry[11],
                    'transition_sequence': entry[12],  # List
                    'triplet counts': entry[13]  # Dict
                }
                structured_data.append(data_entry)
    
    return structured_data

# Define base directory
base_dir = "E:/Jadhav lab data/Behavior/CohortER1/Social W/ratsamples/Opaque"

# Load the structured array
data = process_social_csvFiles(base_dir)

# Sort the dict by date and run numbers (ascending order)
sorted_data = sorted(data, key=lambda x: (datetime.strptime(x['date'], "%m-%d-%Y"), int(x['runum'])))


#%% Batch process individual rat experiments

# Define the dtype for the structured array (single rat)
dtype = np.dtype([
    ('name', 'U100'),        # File name
    ('folder', 'U255'),      # Folder path
    ('date', 'U50'),         # Extracted date
    ('cohortnum', 'U50'),    # Cohort number
    ('runum', 'U50'),        # Run number
    ('ratnum', 'U50'),       # Single rat number as string
    ('ratnames', 'U100'),     # Single rat name as string
    ('ratsamples', 'O'),     # Single sample (float)
    ('pokeData',  'O'),
    ('match', 'f8'),         # Single match value (float)
    ('reward', 'f8'),        # Single reward value (float)
    ('nTransitions', 'f8'),  # Single transition count (float)
    ('perf', 'f8'),          # Performance metric (float)
    ('duration', 'f8'),      # Session duration (float)
    ('transition_sequence', 'O'),  # Transition sequence (still an object if a list is expected)
    ('triplet_counts', 'O') # Single triplet count (float)
])



# Function to extract data from filename
def parse_individual_filename(filename):
    """
    Function to extract data from filename
    """
    match = re.search(r"log(\d{2}-\d{2}-\d{4})\((\d+)-([A-Z]+\d+)\)\.stateScriptLog", filename)
    
    if match:
        date, runum, rat1 = match.groups()
        return date, runum, f"{rat1}", [rat1]
    return None, None, None, None

# Function to load .stateScriptLog files
def process_individual_stateScriptLogs(base_dir):
    
    """
    Function to extract data from filename
    """
    
    struct_data = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".stateScriptLog"):
                full_path = os.path.join(root, file)
                date, runum, ratnums, ratnames = parse_individual_filename(file)

                # Dummy DataFrames for example
                df_rat = parse_individual_events(full_path)
                
                match = df_rat['match'].sum()
                reward = df_rat['reward'].sum()
                
                
                session_length = (df_rat['end'].max() - df_rat['start'].min()) / 60
                
                rat_df = combine_consecutive_wells(df_rat)
                
                count = count_valid_triplets(rat_df, column_name='well')
                
                seq = rat_df['well'].dropna().astype(int).astype(str).str.cat()
                
                
                nTransitions = float(len(rat_df))
                
                
                if nTransitions > 150:
                    nTransitions = np.nan
                    perf = np.nan
                else:
                    perf = 100*(reward / nTransitions)


                struct_data.append((
                    file,  # name
                    root,  # folder
                    date,  # date
                    root.split("/")[-1],  # cohortnum (assuming one level up)
                    runum,  # runum
                    str(ratnums),  # ratnums
                    str(ratnames),  # ratnames
                    df_rat,  # ratsamples (tuple of DataFrames)
                    rat_df,  # consolidated ratsamples
                    match,    # matches for rats
                    reward,  # rewards for rats
                    nTransitions,        # transitions for rats
                    perf,                # pair performance
                    session_length,      # session duration
                    seq,                 # transition sequences
                    count                # triplet counts
                ))
                
                # Initialize an empty list to store the data
                structured_data = []

                for entry in struct_data:
                    data_entry = {
                        'name': entry[0],
                        'folder': entry[1],
                        'date': entry[2],
                        'cohortnum': entry[3],
                        'runum': entry[4],
                        'ratnum': entry[5],
                        'ratnames': entry[6],
                        'ratsamples': entry[7],  # DataFrame
                        'pokeData': entry[8],
                        'match': entry[9],
                        'reward': entry[10],
                        'nTransitions': entry[11],
                        'perf': entry[12],
                        'duration': entry[13],
                        'transition sequence': entry[14],  # List
                        'triplet counts': entry[15]  # Dict
                    }
                    structured_data.append(data_entry)


    
    return structured_data

# Define base directory
base_dir = "E:\\Jadhav lab data\\Behavior\\CohortAS5"

# Load the structured array
data = process_individual_stateScriptLogs(base_dir)

# Sort by date and run numbers
sorted_data = sorted(data, key=lambda x: (x['date'], x['runum']))

#%% Extract performance for individual rats (probabilistic foraging experiments)

# Dictionary to store performance values for each rat
rat_performance = {}

for entry in sorted_data:
    rat = entry['ratnum']  # Extract rat name
    perf_value = entry['perf']  # Extract performance value

    if rat not in rat_performance:
        rat_performance[rat] = []
    
    rat_performance[rat].append(perf_value)  # Append performance value

# Convert lists to NumPy arrays
for rat in rat_performance:
    rat_performance[rat] = np.array(rat_performance[rat])

#%% Extract ratwise transition rates, triplet counts and ratio of optimal triplets

ratwise_transition_rate = {}
ratwise_triplet_count = {}

for entry in sorted_data:
    rat = entry['ratnames']  # Single rat name
    duration = float(entry['duration'])

    transition_rates = entry['nTransitions']  # Single value
    triplet_counts = entry['triplet counts']  # Single value or structure

    if rat not in ratwise_transition_rate:
        ratwise_transition_rate[rat] = []
        ratwise_triplet_count[rat] = []

    ratwise_transition_rate[rat].append(transition_rates / duration)  # Compute rate
    ratwise_triplet_count[rat].append(triplet_counts)

# Convert to NumPy arrays
ratwise_transition_rate = {rat: np.array(rates, dtype=float) for rat, rates in ratwise_transition_rate.items()}

ratwise_optimal_triplet_ratio = {}

# Convert list of dictionaries to DataFrame, filling missing values with 0
for key in ratwise_triplet_count.keys():
    
    df = pd.DataFrame(ratwise_triplet_count[key]).T
    ratwise_optimal_triplet_ratio[key]= []
    

    ratio = ((df.loc['123', :] + df.loc['132', :]) / df.sum()).to_numpy()
    ratwise_optimal_triplet_ratio[key].append(ratio)
    
    # Plot heatmap
    plt.figure()
    sns.heatmap(df, annot = False, cmap = "coolwarm", cbar = True, vmin = 0.0, vmax = 10.0)
    
    # Labels and title
    plt.xlabel("Session")
    plt.ylabel("Triplet")
    plt.title(key)
    plt.xlim((0, 10))
    
    plt.show()
    
#%% Extract data from cohort AJ1


# Function to extract data from filename
def parse_individual_filename(filename):
    """
    Function to extract data from filename
    """
    match = re.search(r"log(\d{2}-\d{2}-\d{4})\((\d+)-([A-Z]+\d+)\)\.csv", filename)
    
    if match:
        date, runum, rat1 = match.groups()
        return date, runum, f"{rat1}", [rat1]
    
    return None, None, None, None


# base directory where .h264 files are located
base_dir = "E:/Jadhav lab data/Behavior/CohortAJ1"  

struct_data = []
# Loop through all subdirectories and find .h264 files
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".csv"):
            full_path = os.path.join(root, file)  # Full path to .csv file
            
            date, runum, ratnums, ratnames = parse_individual_filename(file)

            # Dummy DataFrames for example
            df_rat = pd.read_csv(full_path)
            
            df_rat.rename(columns={'current_well':'well'}, inplace=True)
            
                
            reward = df_rat['reward'].sum()
            
            
            session_length = (df_rat['end'].max() - df_rat['start'].min()) / 60
            
            rat_df = combine_consecutive_wells(df_rat)
            
            count = count_valid_triplets(rat_df, column_name='well')
            
            seq = rat_df['well'].dropna().astype(int).astype(str).str.cat()
            
            
            nTransitions = float(len(rat_df))
            
            
            if nTransitions > 150:
                nTransitions = np.nan
                perf = np.nan
            else:
                perf = 100*(reward / nTransitions)
                
         


            struct_data.append((
                file,  # name
                root,  # folder
                date,  # date
                root.split("/")[-1],  # cohortnum (assuming one level up)
                runum,  # runum
                str(ratnums),  # ratnums
                str(ratnames),  # ratnames
                df_rat,  # ratsamples (tuple of DataFrames)
                rat_df,  # consolidated ratsamples
                reward,  # rewards for rats
                nTransitions,        # transitions for rats
                perf,                # pair performance
                session_length,      # session duration
                seq,                 # transition sequences
                count                # triplet counts
            ))
            
            # Initialize an empty list to store the data
            structured_data = []

            for entry in struct_data:
                data_entry = {
                    'name': entry[0],
                    'folder': entry[1],
                    'date': entry[2],
                    'cohortnum': entry[3],
                    'runum': entry[4],
                    'ratnum': entry[5],
                    'ratnames': entry[6],
                    'ratsamples': entry[7],  # DataFrame
                    'pokeData': entry[8],
                    'reward': entry[9],
                    'nTransitions': entry[10],
                    'perf': entry[11],
                    'duration': entry[12],
                    'transition sequence': entry[13],  # List
                    'triplet counts': entry[14]  # Dict
                }
                structured_data.append(data_entry)

data = structured_data

# Sort by date and run numbers
sorted_data = sorted(data, key=lambda x: (x['date'], x['runum']))         
