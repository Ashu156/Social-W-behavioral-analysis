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
# file_path = "E:/Jadhav lab data/Behavior/CohortAS6/Social W/50%/01-15-2025/log01-15-2025(3-FX115-FX117).stateScriptLog"
# df_rat1, df_rat2 = parse_social_events(file_path)
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

#%%

def combine_consecutive_wells2(df, threshold=1.0):
    """
    Extract transition sequence from rat poking data by combining consecutive rows
    with the same well. Also attaches match and reward info per visit,
    and filters out transitions with time gap < threshold.
    """
    # Calculate poke_time if not present
    if 'poke_time' not in df.columns:
        df['poke_time'] = df['end'] - df['start']
    
    # Identify groups of consecutive rows with the same well
    df['group'] = (df['well'] != df['well'].shift()).cumsum()

    grouped = (
        df.groupby('group')
        .agg(
            start=('start', 'first'),
            end=('end', 'last'),
            well=('well', 'first'),
            dwell_time=('poke_time', 'sum'),
            match=('match', lambda x: int(x.any())),
            reward=('reward', lambda x: int(x.any()))
        )
        .reset_index(drop=True)
    )

    # Apply threshold on time between end of current visit and start of next
    records = []
    for i in range(len(grouped) - 1):
        curr = grouped.loc[i]
        next_ = grouped.loc[i + 1]
        gap = next_['start'] - curr['end']

        if gap >= threshold:
            records.append(curr)

    # Always keep the last visit if it wasn’t appended yet
    if len(grouped) >= 2 and (grouped.iloc[-1]['start'] - grouped.iloc[-2]['end']) >= threshold:
        records.append(grouped.iloc[-1])
    elif len(grouped) == 1:
        records.append(grouped.iloc[0])  # handle single visit edge case

    return pd.DataFrame(records)


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
                    total_transitions = np.sum(nTransitions)
                    if np.isnan(total_transitions) or total_transitions == 0:
                        print(f"Warning: Division issue in file {file}. nTransitions: {nTransitions}, match1: {match1}")
                        perf = np.nan
                    else:
                        perf = 200 * (match1 / total_transitions)



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
base_dir = "E:/Jadhav lab data/Behavior/CohortAS6/Social W/100%"

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
pairwise_inter_match_interval = {}
pairwise_inter_reward_interval = {}

for entry in sorted_data:
    ratnames = tuple(sorted(entry['ratnames']))  # entry[6] contains ['Rat1', 'Rat2']
    perf = entry['perf']  # Assuming perf is the second last column
    duration = float(entry['duration'])
    match_rate = entry['match'][0] / duration
    reward_rate = entry['reward'][0] / duration
    df = entry['ratsamples'][0]
    match_times = df.loc[df['match'] == 1, 'start'].to_numpy()
    reward_times = df.loc[df['reward'] == 1, 'start'].to_numpy()

    if ratnames not in pairwise_perf:
        pairwise_perf[ratnames] = []
        pairwise_match_rate[ratnames] = []
        pairwise_reward_rate[ratnames] = []
        pairwise_inter_match_interval[ratnames] = []
        pairwise_inter_reward_interval[ratnames] = []

    pairwise_perf[ratnames].append(perf)
    pairwise_match_rate[ratnames].append(match_rate)
    pairwise_reward_rate[ratnames].append(reward_rate)
    pairwise_inter_match_interval[ratnames].append(np.mean(np.diff(match_times)))
    pairwise_inter_reward_interval[ratnames].append(np.mean(np.diff(reward_times)))

# Convert to NumPy arrays
pairwise_perf = {pair: np.array(perfs, dtype=float) for pair, perfs in pairwise_perf.items()}
pairwise_match_rate = {pair: np.array(perfs, dtype=float) for pair, perfs in pairwise_match_rate.items()}
pairwise_reward_rate = {pair: np.array(perfs, dtype=float) for pair, perfs in pairwise_reward_rate.items()}
pairwise_inter_match_interval = {pair: np.array(perfs, dtype=float) for pair, perfs in pairwise_inter_match_interval.items()}
pairwise_inter_reward_interval = {pair: np.array(perfs, dtype=float) for pair, perfs in pairwise_inter_reward_interval.items()}

#%% Parse performance, match rate, etc. for each rat separately

# Dictionary to store performance data for each individual rat
ratwise_perf = {}
ratwise_match_rate = {}
ratwise_reward_rate = {}
ratwise_transition_rate = {}
ratwise_triplet_count = {}

for entry in sorted_data:
    rat1, rat2 = entry['ratnames']  # entry['ratnames'] contains ['Rat1', 'Rat2']
    perf = entry['perf']  # performance value for that session
    duration = float(entry['duration'])

    # Extract values separately for each rat
    matches = entry['match']  # Assuming entry['match'] is [rat1_match, rat2_match]
    rewards = entry['reward']
    transitions = entry['nTransitions']
    triplet_counts = entry['triplet counts']

    if np.isnan(perf):
        matches = (np.nan, np.nan)
        rewards = (np.nan, np.nan)
                
        for d in triplet_counts:
            for key in d:
                d[key] = np.nan
        

    for i, rat in enumerate([rat1, rat2]):  # Iterate over both rats with index
        if rat not in ratwise_perf:
            ratwise_perf[rat] = []
            ratwise_match_rate[rat] = []
            ratwise_reward_rate[rat] = []
            ratwise_transition_rate[rat] = []
            ratwise_triplet_count[rat] = []

        # Compute rates while avoiding division by zero
        match_rate = matches[i] / duration #if perf else np.nan
        reward_rate = rewards[i] / duration #if perf else np.nan
        transition_rate = transitions[i] / duration 

        ratwise_perf[rat].append(perf)  # Assuming performance is the same for both rats
        ratwise_match_rate[rat].append(match_rate)  # Extract individual match rate
        ratwise_reward_rate[rat].append(reward_rate)  # Extract individual reward rate
        ratwise_transition_rate[rat].append(transition_rate)  # Extract individual transition rate

        # Check if triplet_counts exists and has the right structure
        if isinstance(triplet_counts, (list, np.ndarray)) and len(triplet_counts) > i:
            ratwise_triplet_count[rat].append(triplet_counts[i])
        else:
            ratwise_triplet_count[rat].append(np.nan)  # Assign NaN if triplet count is missing
        
# Convert to NumPy arrays
ratwise_perf = {rat: np.array(perfs, dtype = float) for rat, perfs in ratwise_perf.items()}
ratwise_match_rate = {rat: np.array(rates, dtype = float) for rat, rates in ratwise_match_rate.items()}
ratwise_reward_rate = {rat: np.array(rates, dtype = float) for rat, rates in ratwise_reward_rate.items()}
ratwise_transition_rate = {rat: np.array(rates, dtype = float) for rat, rates in ratwise_transition_rate.items()}
# ratwise_triplet_count = {rat: np.array(rates, dtype=float) for rat, rates in ratwise_triplet_count.items()}


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

n_sess = 34
v_max = 10.0

ratwise_optimal_triplet_ratio = {}

for key in ratwise_triplet_count.keys():
    df = pd.DataFrame(ratwise_triplet_count[key]).T
    
    # Drop sessions where all triplet counts are NaN
    df_filtered = df.dropna(axis=1, how='all')

    # Initialize empty list for storage
    ratwise_optimal_triplet_ratio[key] = []

    if df_filtered.empty:
        # If all sessions were NaN, store NaN for consistency
        ratwise_optimal_triplet_ratio[key].append(np.nan)
        continue

    # Compute ratio, keeping NaN where all values in a session are NaN
    ratio = ((df.loc['123', :] + df.loc['132', :]) / df.sum(skipna=False)).to_numpy()
    
    ratwise_optimal_triplet_ratio[key].append(ratio)

    # Plot heatmap only for valid data
    plt.figure()
    sns.heatmap(df_filtered, annot=False, cmap="coolwarm", cbar=True, vmin=0.0, vmax=v_max)

    plt.xlabel("Session")
    plt.ylabel("Triplet")
    plt.title(key)
    plt.xlim((0, n_sess))
    plt.xticks((np.arange(0,n_sess,10)))
    plt.show()

#%% Plot optimal_triplet ratio for each rat

for key in ratwise_optimal_triplet_ratio.keys():  
    tempData = ratwise_optimal_triplet_ratio[key][0]
    tempData = tempData[~np.isnan(tempData)]  # Remove NaN values
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
    perf = entry['perf']

    if np.isnan(perf):
        if ratnames not in pairwise_mi:
            pairwise_mi[ratnames] = []
        pairwise_mi[ratnames].append(np.nan)    
        
    else:
        ratnames = tuple(sorted(entry['ratnames']))  # ratnames
        rat1 = entry['ratsamples'][0]  # rat1 data
        rat2 = entry['ratsamples'][1]  # rat2 data

        rat1_df = combine_consecutive_wells(rat1) # get transition sequence for rat 1
        rat2_df = combine_consecutive_wells(rat2) # get transition sequence for rat 2

        # Assign 'hiswell' column values
        for tr in range(len(rat1_df)):
            his_last_poke = rat2_df[rat2_df["start"] < rat1_df["start"].iloc[tr]].index.max()
            if his_last_poke is not None and not np.isnan(his_last_poke):
                rat1_df.at[tr, "hiswell"] = rat2_df.at[his_last_poke, "well"]

        if rat1_df.shape[0] > 5:
            try:
                # Drop rows where either 'well' or 'hiswell' has NaN
                filtered_df = rat1_df.dropna(subset=['well', 'hiswell'])

                if filtered_df.empty:
                    mi = np.nan  # Set NaN if no valid data remains
                else:
                    # Convert to NumPy arrays
                    a = filtered_df['well'].to_numpy().reshape(-1, 1).astype(float)
                    target = filtered_df['hiswell'].to_numpy().astype(float)

                    # Calculate mutual information using mutual_info_regression
                    mi = mutual_info_regression(a, target, random_state=42)
                    mi = mi[0] if isinstance(mi, np.ndarray) and mi.size > 0 else np.nan

            except ValueError as e:
                print(f"Error processing file {entry['name']}: {e}")
                mi = np.nan
        else:
            mi = np.nan  # Explicitly set NaN if not enough data

    # Store results
    if ratnames not in pairwise_mi:
        pairwise_mi[ratnames] = []
    
    pairwise_mi[ratnames].append(mi)

# Convert lists to NumPy arrays, ensuring NaNs are properly assigned
pairwise_mi = {pair: np.array(mis, dtype=float) for pair, mis in pairwise_mi.items()}

#%% Leader-follower relation

from collections import defaultdict
import itertools

def calculate_transition_probabilities_single_vector(vector, pattern_lengths=[1, 2, 3]):
    """
    Calculate transition probabilities to 0 or 1 after various patterns in a single binary vector.
    
    Parameters:
    -----------
    vector : list or numpy array
        Binary vector (0s and 1s)
    pattern_lengths : list, optional
        Lengths of patterns to analyze, by default [1, 2, 3]
    
    Returns:
    --------
    dict
        Dictionary with pattern lengths as keys and pattern statistics as values
    """
    # Convert input to numpy array if it isn't already
    if not isinstance(vector, np.ndarray):
        v = np.array(vector, dtype = int)
    else:
        v = vector.astype(int)
    
    # Check if input is valid
    if not set(np.unique(v)).issubset({0, 1}):
        raise ValueError("Vector must contain only binary values (0 and 1)")
    
    results = {}
    
    # Process each pattern length
    for length in pattern_lengths:
        # Skip if vector is too short for this pattern length
        if len(v) <= length:
            continue
            
        # Generate all possible patterns for this length
        all_possible_patterns = [''.join(map(str, p)) for p in itertools.product([0, 1], repeat=length)]
        
        # Create dictionaries to store counts
        pattern_followed_by_0 = defaultdict(int)
        pattern_followed_by_1 = defaultdict(int)
        pattern_total = defaultdict(int)
        
        # Iterate through the vector
        for i in range(len(v) - length):
            # Extract the pattern
            pattern = tuple(v[i:i+length])
            pattern_str = ''.join(map(str, pattern))
            
            # Get the next value
            next_value = v[i+length]
            
            # Update counts
            pattern_total[pattern_str] += 1
            if next_value == 0:
                pattern_followed_by_0[pattern_str] += 1
            else:  # next_value == 1
                pattern_followed_by_1[pattern_str] += 1
        
        # Calculate probabilities for all possible patterns
        pattern_probs = {}
        for pattern in all_possible_patterns:
            total = pattern_total[pattern]
            prob_0 = pattern_followed_by_0[pattern] / total if total > 0 else 0
            prob_1 = pattern_followed_by_1[pattern] / total if total > 0 else 0
            pattern_probs[pattern] = {'P(0)': prob_0, 'P(1)': prob_1, 'count': total}
        
        results[f"Length-{length} patterns"] = pattern_probs
    
    return results

# def display_results(results):
#     """
#     Display the results in a formatted way.
    
#     Parameters:
#     -----------
#     results : dict
#         Dictionary containing the transition probability results
#     """
#     for length_key, patterns in results.items():
#         print(f"\n{length_key}:")
#         print(f"{'Pattern':<10} {'P(0)':<10} {'P(1)':<10} {'Count':<10}")
#         print("-" * 40)
        
#         # Sort patterns for more organized display
#         sorted_patterns = sorted(patterns.keys())
#         for pattern in sorted_patterns:
#             probs = patterns[pattern]
#             print(f"{pattern:<10} {probs['P(0)']:<10.4f} {probs['P(1)']:<10.4f} {probs['count']:<10}")



# Initialize dictionaries
ratwise_leader = {}
ratwise_pSwitch = {}
ratwise_lag = {}  # New dictionary to store the pairwise lag between rat1 and rat2 for each match
ratwise_tpm = {}  # New dictionary to store the probability to switch between states given last n states
ratwise_departure_lag = {}
ratwise_arrival_lags_all = {}
ratwise_departure_lags_all = {}
ratwise_lead_vector = {}

# Iterate through each entry in sorted_data
for entry in sorted_data:
    rat1_df, rat2_df = entry['ratsamples']
    rat1, rat2 = entry['ratnames']  # Assuming 'rats' key stores their names

    # if rat1_df['match'].sum() == rat2_df['match'].sum():
    idx1 = rat1_df.index[rat1_df['match'] == 1]
    idx2 = rat2_df.index[rat2_df['match'] == 1]
        
    r1 = np.zeros(len(idx1))
    r2 = np.zeros(len(idx2))
    pairwise_lags = []  # List to store pairwise lag for each match
    pairwise_dep_lags = []  # List to store pairwise lag for each match
    
    
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
        
    def next_departure_time(df, row_index):
        """
        Returns the departure time (i.e., end time) from the current well at row_index.
        """
        current_well = df.loc[row_index, 'well']
        next_rows = df.loc[row_index + 1:]
        next_different = next_rows[next_rows['well'] != current_well]
    
        if not next_different.empty:
            return df.loc[row_index, 'end']  # Departure is at the current row's end
        else:
            return None  # No known departure

    

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
        
        r1_depart = next_departure_time(rat1_df, idx1[match])
        r2_depart = next_departure_time(rat2_df, idx2[match])
        
        if r1_arrival is None:
            r1_arrival = rat1_df.loc[idx1[match], 'start']
            
        if r2_arrival is None:
            r2_arrival = rat2_df.loc[idx2[match], 'start']
            
        if r1_depart is None:
            r1_depart = rat1_df.loc[idx1[match], 'end']
            
        if r2_depart is None:
            r2_depart = rat2_df.loc[idx2[match], 'end']
        
        # Calculate the absolute lag (time difference) between the two rats' arrivals
        lag = abs(r1_arrival - r2_arrival)
        pairwise_lags.append(lag)  # Store only the lag value
        
        depart_lag = abs(r1_depart - r2_depart)
        pairwise_dep_lags.append(depart_lag)
        
        # Determine which rat is the leader based on arrival times
        if r1_arrival < r2_arrival:
            r1[match] = 1
        elif r1_arrival > r2_arrival:
            r2[match] = 1
        
    # Check if r1 and r2 are empty before division and mean calculation
    if len(r1) > 0:
        r1_Lead = np.sum(r1) / len(r1)
    else:
        r1_Lead = np.nan

    if len(r2) > 0:
        r2_Lead = np.sum(r2) / len(r2)
    else:
        r2_Lead = np.nan
    
    # Calculate probability of leadership switch between consecutive matches
    r1_pSwitch = pSwitch_leader(r1) if len(r1) > 0 else np.nan
    r2_pSwitch = pSwitch_leader(r2) if len(r2) > 0 else np.nan
    
    # Compute the mean arrival lag for the pair, ensuring no empty list is passed
    mean_arrival_lag = np.mean(np.array(pairwise_lags)) if pairwise_lags else np.nan
    
    # Compute the mean arrival lag for the pair, ensuring no empty list is passed
    mean_departure_lag = np.mean(np.array(pairwise_dep_lags)) if pairwise_dep_lags else np.nan
    
    pairwise_arrival_lags_all = np.array(pairwise_lags) if pairwise_lags else np.nan
    
    pairwise_departure_lags_all = np.array(pairwise_dep_lags) if pairwise_dep_lags else np.nan
    
    
    r1_results = calculate_transition_probabilities_single_vector(r1, pattern_lengths=[1, 2, 3])
    r2_results = calculate_transition_probabilities_single_vector(r2, pattern_lengths=[1, 2, 3])
    
    # Check if 'perf' for the entry is NaN
    perf = entry['perf']
    if np.isnan(perf):
        r1_Lead = np.nan
        r2_Lead = np.nan
        r1_pSwitch = np.nan
        r2_pSwitch = np.nan
        mean_arrival_lag = np.nan
        mean_departure_lag = np.nan
        r1_results = np.nan
        r2_results = np.nan
    
    # Assign the arrays to the correct rat IDs in a single loop
    for rat, r_lead, r_pSwitch, r_tpm, rat_lead_vector in zip(entry['ratnames'], [r1_Lead, r2_Lead], [r1_pSwitch, r2_pSwitch], [r1_results, r2_results], [r1, r2]):
        ratwise_leader.setdefault(rat, []).append(r_lead)
        ratwise_pSwitch.setdefault(rat, []).append(r_pSwitch)
        ratwise_lag.setdefault(rat, []).append(mean_arrival_lag)  # Store the same mean arrival lag for both rats
        ratwise_tpm.setdefault(rat, []).append(r_tpm)
        ratwise_departure_lag.setdefault(rat, []).append(mean_departure_lag)
        ratwise_arrival_lags_all.setdefault(rat, []).append(pairwise_arrival_lags_all)
        ratwise_departure_lags_all.setdefault(rat, []).append(pairwise_departure_lags_all)
        ratwise_lead_vector.setdefault(rat, []).append(rat_lead_vector)

# Convert lists to numpy arrays
ratwise_leader = {rat: np.array(values) for rat, values in ratwise_leader.items()}
ratwise_pSwitch = {rat: np.array(values) for rat, values in ratwise_pSwitch.items()}
ratwise_lag = {rat: np.array(values) for rat, values in ratwise_lag.items()}
ratwise_departure_lag = {rat: np.array(values) for rat, values in ratwise_departure_lag.items()}


ratwise_arrival_lags_all = {
    rat: np.concatenate([np.atleast_1d(arr) for arr in values]) if values else np.array([]) 
    for rat, values in ratwise_arrival_lags_all.items()
}


ratwise_departure_lags_all = {
    rat: np.concatenate([np.atleast_1d(arr) for arr in values]) if values else np.array([]) 
    for rat, values in ratwise_departure_lags_all.items()
}

ratwise_lead_vector = {
    rat: np.concatenate([np.atleast_1d(arr) for arr in values]) if values else np.array([]) 
    for rat, values in ratwise_lead_vector.items()
}



# Get sorted keys
rat_keys = list(ratwise_leader.keys())

# Dictionary to store pairwise differences
pairwise_leader = {}

# Iterate through keys in steps of 2
for i in range(0, len(rat_keys), 2):
    rat1, rat2 = rat_keys[i], rat_keys[i + 1]
    pair_name = f"{rat1} & {rat2}"
    pairwise_leader[pair_name] = np.array(ratwise_leader[rat1] - ratwise_leader[rat2])

###############################################################################
################## Extract transition probabilities of interest ###############
###############################################################################

# Define the keys you are interested in
target_keys = ['0', '1', '00', '11', '000', '111']

# Initialize a new dictionary to store the extracted probabilities
extracted_probs = {}

# Iterate through each rat in the original dictionary
for rat, sessions in ratwise_tpm.items():
    # For each rat, create a list to hold data from each session
    rat_sessions = []

    for session in sessions:
        session_data = {}

        # If session is NaN, add NaNs for all target keys
        if not isinstance(session, dict):
            session_data = {key: {'P(0)': np.nan, 'P(1)': np.nan} for key in target_keys}
        else:
            # Check all pattern lengths (1 to 3)
            for pattern_length_key in ['Length-1 patterns', 'Length-2 patterns', 'Length-3 patterns']:
                pattern_dict = session.get(pattern_length_key, {})

                for key in target_keys:
                    if key in pattern_dict:
                        probs = pattern_dict[key]
                        session_data[key] = {
                            'P(0)': probs.get('P(0)', np.nan),
                            'P(1)': probs.get('P(1)', np.nan)
                        }

            # Ensure all target keys are present (fill with NaNs if missing)
            for key in target_keys:
                if key not in session_data:
                    session_data[key] = {'P(0)': np.nan, 'P(1)': np.nan}

        rat_sessions.append(session_data)

    extracted_probs[rat] = rat_sessions
    

# Initialize separate dicts for P(0) and P(1)
p0_dict = {}
p1_dict = {}

# Loop through each rat and session
for rat, sessions in extracted_probs.items():
    p0_sessions = []
    p1_sessions = []

    for session in sessions:
        p0_session = {}
        p1_session = {}

        for key, probs in session.items():
            p0_session[key] = probs.get('P(0)', np.nan)
            p1_session[key] = probs.get('P(1)', np.nan)

        p0_sessions.append(p0_session)
        p1_sessions.append(p1_session)

    p0_dict[rat] = p0_sessions
    p1_dict[rat] = p1_sessions
    

import numpy as np

# Define the order of pattern keys (columns)
pattern_keys = ['0', '1', '00', '11', '000', '111']

# Function to convert dict to numpy arrays
def dict_to_array(prob_dict):
    rat_arrays = {}

    for rat, sessions in prob_dict.items():
        session_array = []

        for session in sessions:
            # Ensure consistent ordering of columns
            row = [session.get(key, np.nan) for key in pattern_keys]
            session_array.append(row)

        rat_arrays[rat] = np.array(session_array)

    return rat_arrays

# Convert both dicts
p0_arrays = dict_to_array(p0_dict)
p1_arrays = dict_to_array(p1_dict)

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
    perf = entry['perf']
    
    if np.isnan(perf):  # If 'perf' is NaN, set NaN for both rats' data in this session
        # Append NaN for both rats in this session
        if rat1 not in ratwise_poke:
            ratwise_poke[rat1] = []
            ratwise_dwell[rat1] = []
            ratwise_ratio[rat1] = []
        
        if rat2 not in ratwise_poke:
            ratwise_poke[rat2] = []
            ratwise_dwell[rat2] = []
            ratwise_ratio[rat2] = []
        
        ratwise_poke[rat1].append(np.nan)
        ratwise_poke[rat2].append(np.nan)
        ratwise_dwell[rat1].append(np.nan)
        ratwise_dwell[rat2].append(np.nan)
        ratwise_ratio[rat1].append(np.nan)
        ratwise_ratio[rat2].append(np.nan)
        
    else:
    
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
            poke_dwell_ratio = np.divide(poke_time, dwell_time, where = dwell_time != 0)  # Avoid division by zero
            
            ratwise_poke[rat].append(poke_time)
            ratwise_dwell[rat].append(dwell_time)
            ratwise_ratio[rat].append(poke_dwell_ratio)

# Convert lists to numpy arrays
ratwise_poke = {rat: np.array(values) for rat, values in ratwise_poke.items()}
ratwise_dwell = {rat: np.array(values) for rat, values in ratwise_dwell.items()}
ratwise_ratio = {rat: np.array(values) for rat, values in ratwise_ratio.items()}

#%% Concatenate all the following trial information

# Initialize dictionary to store rat-specific DataFrames
ratwise_data = {}
ratwise_session_count = {}  # Dictionary to track session count for each rat

# Iterate through each entry in sorted_data
for entry in sorted_data:
    rat1_df, rat2_df = entry['ratsamples']
    rat1, rat2 = entry['ratnames']  # Assuming 'ratnames' key stores their names
    
    # Apply the filtering function
    rat1_df, rat2_df = filter_following_trials(rat1_df, rat2_df)

    # Process each rat separately
    for rat, df in zip([rat1, rat2], [rat1_df, rat2_df]):
        # Ensure 'following trial' exists before filtering
        if "following trial" not in df:
            continue  # Skip if the column is missing

        # Filter only rows where 'following trial' is True
        df = df[df["following trial"]].copy()

        # Select required columns if they exist
        selected_columns = ["following trial"]
        for col in ["well", "hiswell","match", "reward"]:
            if col in df:
                selected_columns.append(col)

        df = df[selected_columns]

        # If no valid rows remain, create an empty DataFrame with the required columns
        if df.empty:
            df = pd.DataFrame(columns=selected_columns)

        # Assign a session number, counting separately for each rat
        if rat not in ratwise_session_count:
            ratwise_session_count[rat] = 0  # Initialize session count

        df["session"] = ratwise_session_count[rat]  # Assign session ID
        ratwise_session_count[rat] += 1  # Increment session count for the rat

        # Store data in the dictionary, concatenating across sessions
        if rat not in ratwise_data:
            ratwise_data[rat] = df  # Initialize with first session
        else:
            ratwise_data[rat] = pd.concat([ratwise_data[rat], df], ignore_index=True)


# # Convert ratwise_data into a single DataFrame
combined_df = pd.concat(
    [df.assign(rat=rat) for rat, df in ratwise_data.items()], 
    ignore_index=True
)

# Create a combined DataFrame with numeric rat IDs
# combined_df = pd.concat(
#     [df.assign(rat=i + 1) for i, (_, df) in enumerate(ratwise_data.items())],
#     ignore_index=True
# )

# Rename 'match' to 'state'
combined_df = combined_df.rename(columns={"match": "state", "well":"mywell"})

# Select only required columns
combined_df = combined_df[["rat", "mywell", "hiswell","state", "reward", "session"]]

# Save to CSV
csv_path = "E:/Jadhav lab data/Behavior/CohortAS6/Social W/cohortAS1_following_trials_100.csv"  # Modify path if needed
combined_df.to_csv(csv_path, index = False)

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
                total_transitions = np.sum(nTransitions)
                if np.isnan(total_transitions) or total_transitions == 0:
                    print(f"Warning: Division issue in file {file}. nTransitions: {nTransitions}, match1: {match1}")
                    perf = np.nan
                else:
                    perf = 200 * (match1 / total_transitions)


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
base_dir = "E:/Jadhav lab data/Behavior/CohortAS3/Social W/ratsamples/100%"

# Load the structured array
data = process_social_csvFiles(base_dir)

# Sort the dict by date and run numbers (ascending order)
sorted_data = sorted(data, key=lambda x: (datetime.strptime(x['date'], "%m-%d-%Y"), int(x['runum'])))

#%% Calculate distance between choice sequences of rats

from collections import defaultdict
import numpy as np
import textdistance as td

# Dictionary to store data for each unique rat pair
pairwise_dist = defaultdict(list)

# Iterate through sorted_data and group by unique rat pair
for entry in sorted_data:
    rat_pair = tuple(sorted(entry['ratnames']))  # Ensure order consistency by sorting the pair
    pairwise_dist[rat_pair].append(entry)  # Store the entry in the corresponding pair's list

# Convert lists to NumPy arrays
pairwise_arrays = {pair: np.array(entries, dtype=object) for pair, entries in pairwise_dist.items()}
    
    
# Dictionary to store performance data for each unique rat pair
pairwise_d = {}


for entry in sorted_data:
    ratnames = tuple(sorted(entry['ratnames']))  # ('Rat1', 'Rat2')
    ratsamples = tuple(entry['ratsamples'])  
    perf = entry['perf']
    
    if np.isnan(perf):
        
        if ratnames not in pairwise_d:
            pairwise_d[ratnames] = []
            
        pairwise_d[ratnames].append(np.nan)
            
            

    if ratnames not in pairwise_d:
        pairwise_d[ratnames] = []
        
    rat1 = combine_consecutive_wells(ratsamples[0])
    rat2 = combine_consecutive_wells(ratsamples[1])

    seq1 = rat1['well'].dropna()
    seq1 = seq1.astype(int)
    seq1 = seq1.astype(str).str.cat()

    seq2 = rat2['well'].dropna()
    seq2 = seq2.astype(int)
    seq2 = seq2.astype(str).str.cat()

    dist = td.jaccard(seq1, seq2)

    pairwise_d[ratnames].append(dist)
    

# Convert to NumPy arrays
pairwise_dist = {pair: np.array(perfs, dtype=float) for pair, perfs in pairwise_d.items()}

#%% Calculate similarity between choice sequences of a rat across all sessions

import textdistance as td

# Step 1: Identify all unique rat names and collect their data
ratwise_data = {}

for session in sorted_data:
    rat1, rat2 = session['ratnames']
    rat1_df, rat2_df = session['ratsamples']

    for rat, df in zip([rat1, rat2], [rat1_df, rat2_df]):
        if rat not in ratwise_data:
            ratwise_data[rat] = []
        ratwise_data[rat].append(df)  # Store data from each session

# Step 2: Compute Jaccard distances across sessions for each rat
ratwise_distances = {}

for rat, dfs in ratwise_data.items():
    num_sessions = len(dfs)
    matrix = pd.DataFrame(np.nan, index=range(num_sessions), columns=range(num_sessions))

    # Extract sequences for each session
    sequences = []
    for df in dfs:
        df = combine_consecutive_wells(df)  # Apply processing
        seq = df['well'].dropna().astype(int).astype(str).tolist()
        sequences.append(''.join(seq))  # Convert to string format

    # Compute Jaccard distance between sessions
    for i in range(num_sessions):
        for j in range(i, num_sessions):  # Only compute upper triangle (matrix is symmetric)
            distance = td.jaccard(sequences[i], sequences[j])
            matrix.loc[i, j] = distance
            matrix.loc[j, i] = distance  # Fill lower triangle

    ratwise_distances[rat] = matrix

# Print or save results
for rat, matrix in ratwise_distances.items():
    # print(f"Jaccard Distance Matrix for {rat}:\n", matrix, "\n")
    plt.figure()
    heatmap = sns.heatmap(matrix, cmap = 'RdBu_r')
    # Get the current axes
    ax = plt.gca()

    # Reverse the y-axis
    ax.invert_yaxis()
    plt.title(f"Rat {rat}")
    # Show the plot
    plt.show()
    
#%% Global transition probability and diversity

from scipy.stats import entropy


###############################################################################
#################### Compute transition matrix ################################

def compute_transition_matrix(sequence):
    """Computes the transition probability matrix from a sequence of well visits."""
    unique_wells = sorted(set(sequence))  # Get all unique dispensers
    n_wells = len(unique_wells)
    well_index = {well: idx for idx, well in enumerate(unique_wells)}

    # Initialize transition count matrix
    transition_counts = np.zeros((n_wells, n_wells))

    # Count transitions
    for i in range(len(sequence) - 1):
        from_well = sequence[i]
        to_well = sequence[i + 1]
        transition_counts[well_index[from_well], well_index[to_well]] += 1

    # Convert to probability matrix
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transition_counts, row_sums, where=row_sums != 0)  # Avoid division by zero

    return pd.DataFrame(transition_counts, index=unique_wells, columns=unique_wells), pd.DataFrame(transition_probs, index=unique_wells, columns=unique_wells)

###############################################################################
#################### Compute global entropy ###################################

def compute_global_entropy(transition_matrix):
    """Computes Shannon entropy from a transition probability matrix."""
    probabilities = transition_matrix.values.flatten()
    probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
    return -np.sum(probabilities * np.log(probabilities))

###############################################################################
#################### Compute transition diversity #############################

def simpson_di(transition_counts):
    """
    Computes the Simpson Diversity Index from a transition count matrix.

    Parameters:
    transition_counts (pd.DataFrame): A DataFrame where rows represent source dispensers
                                      and columns represent target dispensers, with values
                                      indicating the number of transitions.

    Returns:
    float: Simpson's Diversity Index
    """
    # Flatten the matrix and sum the total transitions
    total_transitions = transition_counts.values.sum()
    
    if total_transitions == 0:
        return 0  # Avoid division by zero if no transitions exist

    # Compute Simpson's Diversity Index
    return 1 - np.sum((transition_counts.values / total_transitions) ** 2)

###############################################################################
# Step 1: Identify all unique rat names and collect their data
ratwise_data = {}
condition = 'single'

if condition == 'paired':
    for session in sorted_data:
        rat1, rat2 = session['ratnames']
        rat1_df, rat2_df = session['ratsamples']
    
        for rat, df in zip([rat1, rat2], [rat1_df, rat2_df]):
            if rat not in ratwise_data:
                ratwise_data[rat] = []
            ratwise_data[rat].append(df)  # Store data from each session
            
elif condition == 'single':
    for session in sorted_data:
        rat = session['ratnames']         # string
        df = session['ratsamples']        # DataFrame

        if rat not in ratwise_data:
            ratwise_data[rat] = []
        ratwise_data[rat].append(df)


# Step 2: Compute Jaccard distances across sessions for each rat
ratwise_transition_matrix = {}
ratwise_entropy = {}
ratwise_diversity = {}

for rat, dfs in ratwise_data.items():
    num_sessions = len(dfs)
    

    # Extract sequences for each session
    sequences = []
    for df in dfs:
        df = combine_consecutive_wells(df)  # Apply processing
        seq = df['well'].dropna().astype(int).astype(str).tolist()
        sequences.append(''.join(seq))  # Convert to string format
    
    transition_matrix = []
    global_entropy = []
    global_diversity = []
    
    # Compute Jaccard distance between sessions
    for i in range(num_sessions):
        
        tc, matrix = compute_transition_matrix(sequences[i])
        transition_matrix.append(matrix)
        global_entropy.append(compute_global_entropy(matrix))
        
        if np.isclose(tc.values.sum(), 0.0): # checks if sum of total counts is cose to 0
            simpson_div = 0.0
        else:
            simpson_div = float(1) / simpson_di(tc)
            
        global_diversity.append(simpson_div)

    ratwise_transition_matrix[rat] = transition_matrix
    ratwise_entropy[rat] = global_entropy
    ratwise_diversity[rat] = global_diversity

# Convert to NumPy arrays
ratwise_diversity = {rat: np.array(values) for rat, values in ratwise_diversity.items()}
ratwise_entropy = {rat: np.array(values) for rat, values in ratwise_entropy.items()}



#%% Markov model predictions

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

def compute_transition_matrix(sequences):
    """Computes the transition probability matrix from a list of dispenser visit sequences."""
    unique_wells = sorted(set([well for seq in sequences for well in seq]))  # Get all unique dispensers
    n_wells = len(unique_wells)
    well_index = {well: idx for idx, well in enumerate(unique_wells)}

    # Initialize transition count matrix
    transition_counts = np.zeros((n_wells, n_wells))

    # Count transitions
    for seq in sequences:
        for i in range(len(seq) - 1):
            from_well = seq[i]
            to_well = seq[i + 1]
            transition_counts[well_index[from_well], well_index[to_well]] += 1

    # Normalize to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transition_counts, row_sums, where=row_sums != 0)  # Avoid division by zero

    return pd.DataFrame(transition_probs, index=unique_wells, columns=unique_wells), well_index

def predict_next_state(transition_matrix, current_state):
    """Predicts the next state using the Markov model."""
    if current_state not in transition_matrix.index:
        return np.random.choice(transition_matrix.columns)  # If unseen state, pick random
    return transition_matrix.loc[current_state].idxmax()  # State with highest transition probability

def random_baseline_prediction(unique_wells):
    """Predicts the next state randomly as a baseline."""
    return np.random.choice(unique_wells)

def evaluate_predictions(true_next, predicted_next):
    """Computes prediction accuracy."""
    return np.mean(np.array(true_next) == np.array(predicted_next))

def cross_validate_markov(sequences, k=5):
    """Performs 10-fold cross-validation to evaluate the Markov model."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    sequences = np.array(sequences, dtype=object)  # Ensure array format

    markov_accuracies = []
    random_accuracies = []
    confusion_matrices = []

    for train_index, test_index in kf.split(sequences):
        train_sequences = sequences[train_index]
        test_sequences = sequences[test_index]

        # Compute transition matrix on training set
        transition_matrix, well_index = compute_transition_matrix(train_sequences)
        unique_wells = list(well_index.keys())

        true_next, predicted_next_markov, predicted_next_random = [], [], []

        for seq in test_sequences:
            for i in range(len(seq) - 1):
                current_state = seq[i]
                true_next.append(seq[i + 1])

                predicted_next_markov.append(predict_next_state(transition_matrix, current_state))
                predicted_next_random.append(random_baseline_prediction(unique_wells))

        # Compute accuracy
        markov_acc = evaluate_predictions(true_next, predicted_next_markov)
        random_acc = evaluate_predictions(true_next, predicted_next_random)

        markov_accuracies.append(markov_acc)
        random_accuracies.append(random_acc)

        # Compute confusion matrix
        cm = confusion_matrix(true_next, predicted_next_markov, labels=unique_wells)
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalize rows
        confusion_matrices.append(cm)

    # Average confusion matrix
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

    # Print results
    print(f"Markov Model Accuracy: {np.mean(markov_accuracies):.3f} ± {np.std(markov_accuracies):.3f}")
    print(f"Random Baseline Accuracy: {np.mean(random_accuracies):.3f} ± {np.std(random_accuracies):.3f}")

    return avg_confusion_matrix, markov_accuracies, random_accuracies



# Run Cross-Validation
avg_cm, markov_acc, random_acc = cross_validate_markov(sequences)

# Display averaged confusion matrix
print("\nAveraged Confusion Matrix:")
print(pd.DataFrame(avg_cm))


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
base_dir = "E:/Jadhav lab data/Behavior/CohortAS4"

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

#%% Preparing data for mHMM 

import pandas as pd
import os

# Define file paths for the two workbooks
file_path_1 = "E:/Jadhav lab data/Behavior/unmixed_pairs_following_trials_original.xlsx"  # Modify this path
file_path_2 = "E:/Jadhav lab data/Behavior/mixed_pairs_following_trials_original.xlsx"  # Modify this path

# Load the WT and KO sheets from both workbooks
sheets_to_load = ["WT", "KO"]
dataframes = []

for file_path in [file_path_1, file_path_2]:
    sheets = pd.read_excel(file_path, sheet_name=sheets_to_load)  # Load only WT and KO
    for sheet_name in sheets_to_load:
        df = sheets[sheet_name].copy()
        df["group"] = sheet_name  # Add a column to track WT or KO
        df["rat"] = df["rat"].astype(str)  # Convert rat IDs to strings
        dataframes.append(df)

# Combine all data
combined_df = pd.concat(dataframes, ignore_index=True)

# Create a unique mapping for rat IDs
wt_rats = sorted(combined_df[combined_df["group"] == "WT"]["rat"].unique(), key=str)
ko_rats = sorted(combined_df[combined_df["group"] == "KO"]["rat"].unique(), key=str)

rat_mapping = {rat: i + 1 for i, rat in enumerate(wt_rats)}  # WT gets IDs 1-15
rat_mapping.update({rat: i + 16 for i, rat in enumerate(ko_rats)})  # KO gets IDs 16-30

# Apply mapping
combined_df["rat"] = combined_df["rat"].map(rat_mapping)

# Drop the group column as it's no longer needed
combined_df.drop(columns=["group"], inplace=True)

# Sort the dataframe first by rat (ascending), then by session (ascending)
combined_df.sort_values(by=["rat", "session"], ascending=[True, True], inplace=True)

# Define output file paths
output_file = "E:/Jadhav lab data/Behavior/combined_rats.csv"  # Modify if needed
mapping_file = "E:/Jadhav lab data/Behavior/rat_mapping.csv"  # Modify if needed

# Save the sorted data to CSV
combined_df.to_csv(output_file, index=False)

# Save the rat mapping as a CSV file
rat_mapping_df = pd.DataFrame(list(rat_mapping.items()), columns=["Original_Rat_ID", "Mapped_Rat_ID"])
rat_mapping_df.to_csv(mapping_file, index=False)

print(f"Combined CSV file saved at: {output_file}")
print(f"Rat mapping file saved at: {mapping_file}")


#%% Plot raster for match events for example pairs

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
pairwise_df = {}
pairwise_match_rate = {}
pairwise_reward_rate = {}


for entry in sorted_data:
    ratnames = tuple(sorted(entry['ratnames']))  # entry[6] contains ['Rat1', 'Rat2']
    df = entry['ratsamples'][0]  # Assuming perf is the second last column
    duration = float(entry['duration'])
    match_rate = entry['match'][0] / duration
    reward_rate = entry['reward'][0] / duration

    if ratnames not in pairwise_df:
        pairwise_df[ratnames] = []
        

    pairwise_df[ratnames].append(df)
    

import matplotlib.pyplot as plt

def plot_raster(match_dict, key, max_idx):
    plt.figure()

    df_list = pairwise_df[key]
    # Combine the first 35 DataFrames into one
    df_combined = df_list[:max_idx]  # Select the first 35 entries (DataFrames)

    for sess in range(len(df_combined)):
        df = df_combined[sess]
        df['start'] = df['start'] - df.loc[0, 'start']
        # Extract 'start' timestamps where match == 1 and within max_idx
        match_times = df.loc[df['match'] == 1, 'start']
        # Plot raster lines
        plt.scatter(match_times, [sess] * len(match_times), marker='|', color = 'red', s=20, label=f"{key}")

        plt.xlabel("Time")
        plt.ylabel("Session")
        plt.yticks(np.arange(0, 35, 5))
        plt.xlim((-50, 1250))
        plt.xticks(np.arange(0, 1205, 300))
        plt.show()

# Example usage:
plot_raster(pairwise_df, ('ER1', 'ER2'), max_idx = 34)


#%% Plot cross-correlation between arrival times of 2 rats (example sessions)

import numpy as np
from numpy.lib.stride_tricks import as_strided
import smoothfit

def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

def crosscorrelation(x, y, maxlag, mode='corr'):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                    strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    if mode == 'dot':       # get lagged dot product
        return T.dot(px)
    elif mode == 'corr':    # gets Pearson correlation
        return (T.dot(px)/px.size - (T.mean(axis=1)*px.mean())) / \
                (np.std(T, axis=1) * np.std(px))



def interpolate_dfs(rat1, rat2, sample_rate):
    # Set sample rate to 24 (e.g., 24 frames per second)
    # sample_rate = 1
    
    # Create a uniformly sampled time vector (sampling rate = 1 unit of time)
    time_vector = np.arange(min(rat1['start'].min(), rat2['start'].min()), max(rat1['end'].max(), rat2['end'].max()),  1/sample_rate)
    
    # Initialize a list to store event assignment for each time point
    event_assignment = []
    
    # Loop over the time vector and assign the closest event when no event is found
    for t in time_vector:
        # Find the active event(s) at time t
        active_event_at_t = rat1[(rat1['start'] <= t) & (rat1['end'] >= t)]
        
        if not active_event_at_t.empty:
            # If there is an event at this time, assign it
            event_assignment.append(active_event_at_t['well'].values[0])
        else:
            # Find the closest event in time when no event is active at this time
            # closest_event = rat1.iloc[(np.abs(rat1[['start', 'end']] - t).min(axis=1)).argmin()]
            # event_assignment.append(closest_event['thiswell'])
            event_assignment.append(0)
            
    
    # Create a DataFrame with time points and assigned events
    rat1wells = pd.DataFrame({'time': time_vector, 'wells': event_assignment})
    
    
    # Initialize a list to store event assignment for each time point
    event_assignment = []
    
    # Loop over the time vector and assign the closest event when no event is found
    for t in time_vector:
        # Find the active event(s) at time t
        active_event_at_t = rat2[(rat2['start'] <= t) & (rat2['end'] >= t)]
        
        if not active_event_at_t.empty:
            # If there is an event at this time, assign it
            event_assignment.append(active_event_at_t['well'].values[0])
        else:
            # Find the closest event in time when no event is active at this time
            # closest_event = rat2.iloc[(np.abs(rat2[['start', 'end']] - t).min(axis=1)).argmin()]
            # event_assignment.append(closest_event['thiswell'])
            event_assignment.append(0)
    
    # Create a DataFrame with time points and assigned events
    rat2wells = pd.DataFrame({'time': time_vector, 'wells': event_assignment})
    
    return rat1wells, rat2wells

# Dictionary to store performance data for each individual rat
ratwise_df = {}


for entry in sorted_data:
    rat1, rat2 = entry['ratnames']  # entry[6] contains ['Rat1', 'Rat2']
    ratsamples = entry['ratsamples']  # Assuming perf is the second last column
    

    


    for i, rat in enumerate([rat1, rat2]):  # Iterate over both rats with index
        if rat not in ratwise_df:
            ratwise_df[rat] = []
            
        ratwise_df[rat].append(ratsamples[i])  # Extract individual match rate
        



key1 = 'FXM108' 
key2 = 'FXM109'
       
sessions = [0, 32]

maxlags = 30

for sess in sessions:
    rat1 = ratwise_df[key1][sess]
    rat2 = ratwise_df[key2][sess]
    
    rat1wells, rat2wells = interpolate_dfs(rat1, rat2, sample_rate = 1)
    
    
    
    corr = crosscorrelation(rat1wells['wells'].to_numpy(), rat2wells['wells'].to_numpy(), maxlags)
    delays = np.arange(-maxlags, maxlags+1, 1)
    
    
    lmbda = 5.0e-1
    
    # Ensure the interval (a, b) fully contains all x-values in `delays`
    a, b = min(delays), max(delays)
    
    basis, coeffs = smoothfit.fit1d(delays, corr, a, b, 1000, degree=1, lmbda=lmbda)
    
    plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], linestyle = '-')




#%% Plot raster plot for well visits and matches

# Step 1: Identify all unique rat names and collect their data
ratwise_data = {}

for session in sorted_data:
    rat1, rat2 = session['ratnames']
    rat1_df, rat2_df = session['ratsamples']
    

    
    # rat1_df = combine_consecutive_wells(rat1_df)
    # rat2_df = combine_consecutive_wells(rat2_df)
    
    start_min = min(rat1_df['start'].min(), rat2_df['start'].min())
    
    rat1_df[['start', 'end']] = rat1_df[['start', 'end']] - start_min
    rat2_df[['start', 'end']] = rat2_df[['start', 'end']] - start_min
    
    for rat, df in zip([rat1, rat2], [rat1_df, rat2_df]):
        if rat not in ratwise_data:
            ratwise_data[rat] = []
        ratwise_data[rat].append(df)  # Store data from each session
        


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_raster(df1, df2):
    wells = sorted(set(df1["well"]).union(set(df2["well"])))  # Unique well IDs
    fig, ax = plt.subplots()

    well_height = 0.2  # Height of each patch
    spacing = 0.8  # Vertical spacing between wells

    for i, well in enumerate(wells):
        y_base = i * spacing  # Base row index for each well
        
        # Plot df1 events
        df1_well = df1[df1["well"] == well]
        for _, row in df1_well.iterrows():
            rect = patches.Rectangle((row["start"], y_base - well_height), 
                                     row["end"] - row["start"], well_height, 
                                     facecolor='black', edgecolor='black', alpha = 0.5)
            ax.add_patch(rect)

        # Plot df2 events
        df2_well = df2[df2["well"] == well]
        for _, row in df2_well.iterrows():
            rect = patches.Rectangle((row["start"], y_base), 
                                     row["end"] - row["start"], well_height, 
                                     facecolor='gray', edgecolor='gray', alpha = 0.5)
            ax.add_patch(rect)

        # Label wells on the y-axis
        ax.text(df1["start"].min() - 2, y_base, f"Well {well}", va="center", fontsize=12)

    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.set_xlim(df1["start"].min() - 2, max(df1["end"].max(), df2["end"].max()) + 2)
    ax.set_ylim(-1, len(wells) * spacing)
    # ax.set_title("Raster Plot of Well Occupancy")

    plt.show()

# Example usage
plot_raster(ratwise_data['FXM102'][34], ratwise_data['FXM103'][34])



df1 = ratwise_data['FXM102'][34]
df2 = ratwise_data['FXM103'][34]

#%%

import matplotlib.pyplot as plt

def plot_raster(df1, df2):
    wells = sorted(set(df1["well"]).union(set(df2["well"])))  # Unique well IDs
    fig, ax = plt.subplots()

    well_offset = 0.3  # Space between subrows
    match_offset = 0.6  # Match row offset

    for i, well in enumerate(wells):
        y_base = i * 3  # Increase spacing to accommodate match row

        # Plot df1 events as vertical lines
        df1_well = df1[df1["well"] == well]
        ax.vlines(df1_well["start"], ymin=y_base - well_offset - 0.2, ymax=y_base - well_offset + 0.2, 
                  color='black', linewidth=2, label="DF1" if i == 0 else "")

        # Plot df2 events as vertical lines
        df2_well = df2[df2["well"] == well]
        ax.vlines(df2_well["start"], ymin=y_base + well_offset - 0.2, ymax=y_base + well_offset + 0.2, 
                  color='red', linewidth=2, label="DF2" if i == 0 else "")

        # Plot match events in a separate row below df2
        df_match = df1_well[df1_well["match"] == 1]  # Assuming match info is in df1
        ax.scatter(df_match["start"], [y_base - match_offset] * len(df_match), 
                   color='green', s=20, label="Match" if i == 0 else "")

        # Label wells on the y-axis
        ax.text(df1["start"].min() - 2, y_base, f"Well {well}", va="center", fontsize=12)

    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 1250, 300))
    # ax.legend()
    # ax.set_title("Raster Plot of Well Occupancy (Match Events Below DF2)")

    plt.show()

# Example usage
plot_raster(ratwise_data['FXM102'][34], ratwise_data['FXM103'][34])


#%% Plot example for timescales

import matplotlib.pyplot as plt

def plot_timescale_example(df1, df2):
    # wells = sorted(set(df1["well"]).union(set(df2["well"])))  # Unique well IDs
    fig, ax = plt.subplots()

    well_offset = 0.3  # Space between subrows
    match_offset = 0.3  # Match row offset
    y_base = 3

    # Plot df1 events as vertical lines
    ax.plot(df1['start'], df1['well'])
    ax.plot(df2['start'], df2['well'])

        

    # Plot match events in a separate row below df2
    df_match = df1[df1["match"] == 1]  # Assuming match info is in df1
    # ax.scatter(df_match["start"], [y_base + match_offset] * len(df_match), 
    #             color='black', s=20, label="Match" if i == 0 else "")
    
    ax.vlines(df_match["start"], ymin=y_base + match_offset - 0.2, ymax=y_base + match_offset + 0.1, 
              color='black', linewidth=2, label="Match")
    
    df_reward = df1[df1["reward"] == 1]  # Assuming match info is in df1
    ax.scatter(df_reward["start"], [y_base + 2*match_offset] * len(df_reward), 
                color='green', s=20, label="Reward" if i == 0 else "")
    
    df_noreward = df1[(df1["match"] == 1) & (df1["reward"] == 0)] # Assuming match info is in df1
    ax.scatter(df_noreward["start"], [y_base + 2*match_offset] * len(df_noreward), 
                color='red', s=20, label="Reward omitted" if i == 0 else "")

    # Label wells on the y-axis
    # ax.text(df1["start"].min() - 2, 3, f"Well {well}", va="center", fontsize=12)

    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 1250, 300))
    # ax.legend()
    # ax.set_title("Raster Plot of Well Occupancy (Match Events Below DF2)")

    plt.show()

# Example usage
plot_timescale_example(ratwise_data['FXM102'][34], ratwise_data['FXM103'][34])

#%%

def analyze_leader_follower(rat1_df, rat2_df):
    i, j = 0, 0
    events = []

    while i < len(rat1_df) and j < len(rat2_df):
        # Step 1: Wait for a match event
        while i < len(rat1_df) and j < len(rat2_df):
            r1 = rat1_df.iloc[i]
            r2 = rat2_df.iloc[j]
            if (r1['match'] == 1 and r2['match'] == 1 and r1['well'] == r2['well']):
                match_well = r1['well']
                match_start_time = min(r1['start'], r2['start'])
                match_end_time = max(r1['end'], r2['end'])
                break
            if r1['start'] < r2['start']:
                i += 1
            else:
                j += 1
        else:
            break  # No more matches

        # Step 2: Determine who leaves first
        i += 1
        j += 1
        leader, follower = None, None
        leader_entry, follower_action = None, None
        leader_depart_time = None
        follower_depart_time = None

        r1_move = rat1_df.iloc[i:] if i < len(rat1_df) else pd.DataFrame()
        r2_move = rat2_df.iloc[j:] if j < len(rat2_df) else pd.DataFrame()

        r1_next = next((row for _, row in r1_move.iterrows() if row['well'] != match_well), None)
        r2_next = next((row for _, row in r2_move.iterrows() if row['well'] != match_well), None)

        if r1_next is None or r2_next is None:
            break  # No further moves

        if r1_next['start'] < r2_next['start']:
            leader = 'rat1'
            follower = 'rat2'
            leader_entry = r1_next
            leader_depart_time = r1_next['start']
            leader_idx = rat1_df.index.get_loc(r1_next.name)
            follower_df = rat2_df
            follower_start_idx = j
        else:
            leader = 'rat2'
            follower = 'rat1'
            leader_entry = r2_next
            leader_depart_time = r2_next['start']
            leader_idx = rat2_df.index.get_loc(r2_next.name)
            follower_df = rat1_df
            follower_start_idx = i

        # Step 4: What did the follower do?
        follow_status = 'stayed'
        for k in range(follower_start_idx, len(follower_df)):
            f_event = follower_df.iloc[k]
            if f_event['start'] > leader_entry['end']:
                break
            if f_event['well'] == leader_entry['well']:
                follow_status = 'followed_leader' if f_event['match'] == 1 else 'matched_wrongly'
                follower_depart_time = f_event['start']
                break
            elif f_event['well'] != match_well:
                follow_status = 'chose_other_well'
                follower_depart_time = f_event['start']
                break

        # Record event
        events.append({
            'match_well': match_well,
            'match_start': match_start_time,
            'leader': leader,
            'follower': follower,
            'leader_next_well': leader_entry['well'],
            'leader_depart_time': leader_depart_time,
            'follower_action': follow_status,
            'follower_depart_time': follower_depart_time
        })

        # Move i/j forward to after the latest visit we've seen
        i = rat1_df.index.get_loc(r1_next.name) + 1
        j = rat2_df.index.get_loc(r2_next.name) + 1

    return pd.DataFrame(events)


#%% 
    
def get_visit_sequence_df(df, start_idx, threshold):
    """
    Returns a DataFrame of well visits after a given index.
    Each row includes:
        - current_well: well that was departed from
        - next_well: well that was visited
        - depart_time: time of leaving current_well
        - arrive_time: time of arriving at next_well
        - match: 1 if match occurred during that visit to next_well
        - reward: 1 if reward occurred during that visit to next_well
    Filters out visits where arrival occurs less than 2 seconds after departure.
    """
    threshold = 1.0
    records = []
    current_well = df.loc[start_idx, 'well']
    idx = start_idx

    while True:
        # Find index where well changes from current_well
        next_idx = df.loc[idx+1:][df['well'] != current_well].first_valid_index()
        if next_idx is None:
            break

        next_well = df.loc[next_idx, 'well']

        # Identify the visit to next_well: consecutive rows with the same well
        visit_start = next_idx
        visit_mask = df.loc[visit_start:, 'well'] == next_well
        visit_end_idx = visit_mask[visit_mask == False].index.min()

        if pd.isna(visit_end_idx):
            visit_end_idx = df.index[-1] + 1  # cover until the end if well never changes again

        visit_df = df.loc[visit_start:visit_end_idx - 1]

        depart_time = df.loc[next_idx - 1, 'end']
        arrive_time = df.loc[visit_start, 'start']

        # Skip if time difference is less than 2 seconds
        if (arrive_time - depart_time) < threshold:
            # Still need to update idx/current_well to proceed to the next visit
            current_well = next_well
            idx = visit_end_idx - 1
            continue

        # Collect info
        record = {
            'current_well': current_well,
            'next_well': next_well,
            'depart_time': depart_time,
            'arrive_time': arrive_time,
            'match': int((visit_df['match'] == 1).any()),
            'reward': int((visit_df['reward'] == 1).any())
        }
        records.append(record)

        # Update for next iteration
        current_well = next_well
        idx = visit_end_idx - 1

    return pd.DataFrame(records)



# start_idx1 = rat1_df[rat1_df['match'] == 1].index[0]
start_idx1 = 0
rat1_seq_df = get_visit_sequence_df(rat1_df, start_idx1, threshold = 1.0)

# start_idx2 = rat2_df[rat2_df['match'] == 1].index[0]
start_idx2 = 0
rat2_seq_df = get_visit_sequence_df(rat2_df, start_idx2, threshold = 1.0)



def analyze_leader_follower_dynamics(rat1_seq_df, rat2_seq_df):
    events = []
    leader_stats = {
        'rat1': {
            'led': 0, 'led_followed': 0, 'followed': 0, 'followed_back': 0, 'diverged': 0,
            'led_and_matched': 0, 'led_and_rewarded': 0,
            'followed_and_matched': 0, 'followed_and_rewarded': 0
        },
        'rat2': {
            'led': 0, 'led_followed': 0, 'followed': 0, 'followed_back': 0, 'diverged': 0,
            'led_and_matched': 0, 'led_and_rewarded': 0,
            'followed_and_matched': 0, 'followed_and_rewarded': 0
        }
    }

    i, j = 0, 0

    while i < len(rat1_seq_df) and j < len(rat2_seq_df):
        r1 = rat1_seq_df.iloc[i]
        r2 = rat2_seq_df.iloc[j]

        dep1 = r1['depart_time']
        dep2 = r2['depart_time']
        arr1 = r1['arrive_time']
        arr2 = r2['arrive_time']

        # Determine leader and follower
        if dep1 < dep2 and (dep2 - dep1 > 1.0):
            leader = 'rat1'
            follower = 'rat2'
            leader_row = r1
            follower_idx = j + 1
            i += 1
        elif dep2 < dep1 and (dep1 - dep2 > 1.0):
            leader = 'rat2'
            follower = 'rat1'
            leader_row = r2
            follower_idx = i + 1
            j += 1
        else:
            i += 1
            j += 1
            continue

        leader_stats[leader]['led'] += 1
        led_to = leader_row['next_well']
        leader_depart = leader_row['depart_time']

        if leader_row['match']:
            leader_stats[leader]['led_and_matched'] += 1
        if leader_row['reward']:
            leader_stats[leader]['led_and_rewarded'] += 1

        # Get follower's next move
        if follower == 'rat1' and follower_idx < len(rat1_seq_df):
            f_next = rat1_seq_df.iloc[follower_idx]
        elif follower == 'rat2' and follower_idx < len(rat2_seq_df):
            f_next = rat2_seq_df.iloc[follower_idx]
        else:
            f_next = None

        if f_next is not None:
            follower_arrives = f_next['arrive_time']
            follower_next_well = f_next['next_well']
            leader_stats[follower]['followed'] += 1

            if f_next['match']:
                leader_stats[follower]['followed_and_matched'] += 1
            if f_next['reward']:
                leader_stats[follower]['followed_and_rewarded'] += 1

            if follower_next_well == led_to and follower_arrives > leader_depart:
                leader_stats[leader]['led_followed'] += 1
                events.append((follower_arrives, f"{follower} followed {leader} to well {led_to}"))
            elif follower_next_well != led_to:
                leader_stats[follower]['diverged'] += 1
                events.append((follower_arrives, f"{follower} diverged to well {follower_next_well}"))

                # Check if original leader follows back
                if leader == 'rat1' and i < len(rat1_seq_df):
                    l_next = rat1_seq_df.iloc[i]
                elif leader == 'rat2' and j < len(rat2_seq_df):
                    l_next = rat2_seq_df.iloc[j]
                else:
                    l_next = None

                if l_next is not None and l_next['next_well'] == follower_next_well:
                    leader_stats[leader]['followed_back'] += 1
                    leader_stats[follower]['led'] += 1
                    leader_stats[follower]['led_followed'] += 1
                    events.append((l_next['arrive_time'], f"{leader} followed {follower} to well {follower_next_well}"))
        else:
            events.append((leader_depart, f"{leader} moved to well {led_to} but {follower} had no further entries"))

    # Create event log DataFrame
    event_df = pd.DataFrame(sorted(events, key=lambda x: x[0]), columns=['time', 'event'])

    # Compute summary ratios
    for rat in ['rat1', 'rat2']:
        led = leader_stats[rat]['led']
        followed = leader_stats[rat]['followed']
        leader_stats[rat]['lead_followed_ratio'] = leader_stats[rat]['led_followed'] / led if led else 0
        leader_stats[rat]['follow_back_ratio'] = leader_stats[rat]['followed_back'] / followed if followed else 0
        leader_stats[rat]['divergence_ratio'] = leader_stats[rat]['diverged'] / followed if followed else 0

    return event_df, leader_stats



event_df, leader_stats = analyze_leader_follower_dynamics(rat1_seq_df, rat2_seq_df)

#%% Exploration efficiency

import numpy as np
import matplotlib.pyplot as plt
import random

def calculate_efficiency(visits, n_nodes=3):
    """
    Calculate exploration efficiency based on the sequence of node visits.
    
    Parameters:
    visits (list): List of nodes visited in sequence (e.g., [1, 2, 1, 3, 2])
    n_nodes (int): Total number of nodes in the environment
    
    Returns:
    float: Efficiency score
    dict: Additional metrics including N_half
    """
    # Convert to numpy array for easier manipulation
    visits = np.array(visits)
    
    # Keep track of unique nodes encountered
    unique_nodes_over_time = []
    unique_nodes_set = set()
    
    for node in visits:
        unique_nodes_set.add(node)
        unique_nodes_over_time.append(len(unique_nodes_set))
    
    unique_nodes_over_time = np.array(unique_nodes_over_time)
    
    # Calculate N_half (visits needed to discover half the nodes)
    # For 3 nodes, half is 1.5, round up to 2
    # half_nodes = n_nodes // 2 if n_nodes % 2 == 0 else (n_nodes // 2) + 1
    half_nodes = 3
    
    try:
        # Find the first position where we've seen half_nodes
        N_half = np.where(unique_nodes_over_time >= half_nodes)[0][0] + 1
        efficiency = half_nodes / N_half
    except IndexError:
        # If we never reached half the nodes
        efficiency = 0
        N_half = np.inf
    
    return efficiency, {
        "N_half": N_half,
        "unique_nodes_over_time": unique_nodes_over_time,
        "half_nodes": half_nodes
    }

def simulate_optimal_agent(n_nodes=3):
    """
    Simulate an optimal agent that visits each node exactly once,
    with the constraint that it cannot visit the same node twice in succession.
    """
    # For an optimal agent with no successive repeats, with 3 nodes:
    # We can simply visit the nodes in order, e.g., [1, 2, 3]
    # Since there are no constraints that would make this path non-optimal
    return list(range(1, n_nodes + 1))

def simulate_random_agent(n_steps=20, n_nodes=3):
    """
    Simulate a random agent that makes random decisions at each step,
    with the constraint that it cannot visit the same node twice in succession.
    """
    visits = []
    
    # First visit is random
    current_node = random.randint(1, n_nodes)
    visits.append(current_node)
    
    # Remaining visits must be different from the previous
    for _ in range(n_steps - 1):
        # Get all possible next nodes (excluding the current one)
        possible_next_nodes = [node for node in range(1, n_nodes + 1) if node != current_node]
        
        # Randomly select one of the possible next nodes
        next_node = random.choice(possible_next_nodes)
        visits.append(next_node)
        
        # Update current node
        current_node = next_node
    
    return visits

def plot_exploration_efficiency(visits_list, labels, n_nodes=3):
    """
    Plot the exploration efficiency curves for different agents
    
    Parameters:
    visits_list (list): List of visit sequences for different agents
    labels (list): Labels for each agent
    n_nodes (int): Total number of nodes
    """
    plt.figure(figsize=(10, 6))
    
    for visits, label in zip(visits_list, labels):
        efficiency, metrics = calculate_efficiency(visits, n_nodes)
        
        # Plot the curve
        x = range(1, len(visits) + 1)
        y = metrics["unique_nodes_over_time"]
        
        plt.plot(x, y, label=f"{label} (E={efficiency:.2f})")
        
        # Mark N_half
        if metrics["N_half"] < np.inf:
            plt.scatter([metrics["N_half"]], [metrics["half_nodes"]], 
                        marker='o', s=100, edgecolors='black')
    
    # Plot reference line for optimal agent if not already included
    if "Optimal Agent" not in labels:
        optimal_visits = simulate_optimal_agent(n_nodes)
        opt_efficiency, opt_metrics = calculate_efficiency(optimal_visits, n_nodes)
        
        x_opt = range(1, len(optimal_visits) + 1)
        y_opt = opt_metrics["unique_nodes_over_time"]
        
        plt.plot(x_opt, y_opt, 'k--', 
                 label=f"Optimal Agent (E={opt_efficiency:.2f})")
    
    plt.xlabel("Number of Visits")
    plt.ylabel("Number of Distinct Nodes Encountered")
    plt.title("Exploration Efficiency in a 3-Node Environment")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Set axis limits
    # plt.xlim(0, max(len(v) for v in visits_list) + 1)
    plt.ylim(0, n_nodes + 0.5)
    
    # Add horizontal line at half_nodes
    half_nodes = n_nodes // 2 if n_nodes % 2 == 0 else (n_nodes // 2) + 1
    plt.axhline(y=half_nodes, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    n_nodes = 3
    
    # Simulate agents
    optimal_agent = simulate_optimal_agent(n_nodes)
    random_agent = simulate_random_agent(20, n_nodes)
    
    # Example empirical data - replace with your actual observations
    # Ensuring no successive repeats in this example
    empirical_data = combine_consecutive_wells2(sorted_data[65]['ratsamples'][0], threshold = 1.0)['well'].to_list()
    # Calculate efficiencies
    opt_efficiency, _ = calculate_efficiency(optimal_agent, n_nodes)
    rand_efficiency, _ = calculate_efficiency(random_agent, n_nodes)
    emp_efficiency, _ = calculate_efficiency(empirical_data, n_nodes)
    
    print(f"Optimal Agent Efficiency: {opt_efficiency:.2f}")
    print(f"Random Agent Efficiency: {rand_efficiency:.2f}")
    print(f"Empirical Data Efficiency: {emp_efficiency:.2f}")
    
    # Plot results
    plot_exploration_efficiency(
        [optimal_agent, random_agent, empirical_data],
        ["Optimal Agent", "Random Agent", "Empirical Data"],
        n_nodes
    )
    
    # Additional analysis - run multiple simulations of random agent
    random_efficiencies = []
    for _ in range(1000):
        random_visits = simulate_random_agent(20, n_nodes)
        efficiency, _ = calculate_efficiency(random_visits, n_nodes)
        random_efficiencies.append(efficiency)
    
    print(f"Average Random Agent Efficiency (100 runs): {np.mean(random_efficiencies):.2f}")
    print(f"Standard Deviation: {np.std(random_efficiencies):.2f}")
    
#%% state space model for leading vector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import integrate
from numba import jit  # For JIT compilation
from tqdm import tqdm  # For progress tracking
import seaborn as sns

# JIT compile the state update function for faster execution
@jit(nopython=True)
def get_state_update(x_pred, v_pred, b0, n):
    """
    Newton-Raphson algorithm to update the state - optimized with Numba
    """
    M = 50  # maximum iterations
    
    it = np.zeros(M)
    it[0] = x_pred
    
    for i in range(M-1):
        exp_term = np.exp(b0 + it[i])
        denom = (1 + exp_term)
        
        func = it[i] - x_pred - v_pred * (n - exp_term / denom)
        df = 1 + v_pred * exp_term / (denom ** 2)
        it[i + 1] = it[i] - func / df
        
        if abs(it[i + 1] - it[i]) < 1e-14:
            return it[i + 1]
    
    # If we didn't converge, return the last value instead of raising an error
    return it[M-1]

# Optimize confidence limit calculation
@jit(nopython=True)
def calculate_integrand(p, v, b0, x):
    """Calculate the integrand for confidence limits"""
    valid_indices = (p > 0) & (p < 1)
    result = np.zeros_like(p)
    
    # Use log for numerical stability
    log_p = np.log(p[valid_indices])
    log_1_minus_p = np.log(1 - p[valid_indices])
    
    # Calculate the integrand
    exponent = (-1) / (2 * v) * (log_p - log_1_minus_p - b0 - x) ** 2
    result[valid_indices] = 1 / (np.sqrt(2 * np.pi * v) * p[valid_indices] * (1 - p[valid_indices])) * np.exp(exponent)
    
    return result

def get_pk_conf_lims(v, b0, x):
    """
    Calculate confidence limits for p_k - more efficient implementation
    Uses adaptive step sizes for the p values
    """
    # Use fewer points for faster integration, but concentrate them where needed
    # Start with a coarse grid
    p_coarse = np.concatenate([
        np.linspace(1e-6, 0.01, 50),
        np.linspace(0.01, 0.99, 500),
        np.linspace(0.99, 1-1e-6, 50)
    ])
    
    # Calculate integrand on coarse grid
    integrand = calculate_integrand(p_coarse, v, b0, x)
    
    # Compute CDF
    fp = integrate.cumtrapz(integrand, p_coarse, initial=0)
    fp = fp / fp[-1]  # Normalize to ensure it reaches 1
    
    # Find indices where CDF crosses our thresholds
    n_indices = np.where(fp <= 0.975)[0]
    m_indices = np.where(fp < 0.025)[0]
    
    ucl = p_coarse[n_indices[-1]] if len(n_indices) > 0 else 1
    lcl = p_coarse[m_indices[-1]] if len(m_indices) > 0 else 0
    
    return lcl, ucl

def run_em_algorithm(n, K, max_iterations=20000, tol=1e-8, base_prob=0.25, 
                     initial_ve=0.005, batch_conf_lims=True):
    """
    Run the EM algorithm with optimizations
    """
    M = max_iterations
    ve = np.zeros(M)
    
    x_pred = np.zeros(K)
    v_pred = np.zeros(K)
    x_updt = np.zeros(K)
    v_updt = np.zeros(K)
    x_smth = np.zeros(K)
    v_smth = np.zeros(K)
    p_updt = np.zeros(K)
    
    A = np.zeros(K)
    W = np.zeros(K)
    CW = np.zeros(K)
    
    ve[0] = initial_ve
    x_smth[0] = 0
    b0 = np.log(base_prob / (1 - base_prob))
    
    # Pre-allocate memory for the exp terms to avoid recomputation
    exp_b0_x = np.zeros(K)
    
    # Use tqdm for progress tracking
    with tqdm(total=M, desc="EM Algorithm Progress") as pbar:
        for m in range(M):
            # Forward pass - can be vectorized in parts but the update step requires iteration
            for k in range(K):
                if k == 0:  # boundary condition
                    x_pred[k] = x_smth[0]
                    v_pred[k] = 2 * ve[m]  # Simplified from ve[m] + ve[m]
                else:
                    x_pred[k] = x_updt[k - 1]
                    v_pred[k] = v_updt[k - 1] + ve[m]
                
                x_updt[k] = get_state_update(x_pred[k], v_pred[k], b0, n[k])
                
                # Precompute the exponential term
                exp_term = np.exp(b0 + x_updt[k])
                exp_b0_x[k] = exp_term
                
                p_updt[k] = 1 / (1 + exp_term)
                v_updt[k] = 1 / ((1 / v_pred[k]) + p_updt[k] * (1 - p_updt[k]))
            
            # Copy values for the last element
            x_smth[K-1] = x_updt[K-1]
            v_smth[K-1] = v_updt[K-1]
            W[K-1] = v_smth[K-1] + x_smth[K-1]**2
            
            # Compute A efficiently
            A[:(K-1)] = v_updt[:(K-1)] / v_pred[1:]
            x0_prev = x_smth[0]
            
            # Backward smoothing pass
            for k in range(K-2, -1, -1):
                x_smth[k] = x_updt[k] + A[k] * (x_smth[k + 1] - x_pred[k + 1])
                v_smth[k] = v_updt[k] + A[k]**2 * (v_smth[k + 1] - v_pred[k + 1])
                
                CW[k] = A[k] * v_smth[k + 1] + x_smth[k] * x_smth[k + 1]
                W[k] = v_smth[k] + x_smth[k]**2
            
            if m < M - 1:
                # More efficient computation of ve update
                ve[m + 1] = (np.sum(W[1:]) + np.sum(W[:(K-1)]) - 2 * np.sum(CW) + 0.5 * W[0]) / (K + 1)
                x0 = x_smth[0] / 2
                
                if (abs(ve[m + 1] - ve[m]) < tol) and (abs(x0 - x0_prev) < tol):
                    print(f'm = {m}\nx0 = {x_smth[0]:.18f}\nve = {ve[m]:.18f}\n')
                    print(f'Converged at m = {m}\n')
                    break
                else:
                    # Only print status every 100 iterations to reduce overhead
                    if m % 100 == 0:
                        print(f'm = {m}\nx0 = {x_smth[0]:.18f}\nve = {ve[m+1]:.18f}\n')
                    
                    # Reset arrays for next iteration - reuse existing arrays
                    x_pred.fill(0)
                    v_pred.fill(0)
                    x_updt.fill(0)
                    v_updt.fill(0)
                    
                    # Keep x_smth[0] for next iteration
                    x0_temp = x0
                    x_smth.fill(0)
                    x_smth[0] = x0_temp
                    
                    v_smth.fill(0)
                    p_updt.fill(0)
                    A.fill(0)
                    W.fill(0)
                    CW.fill(0)
            
            pbar.update(1)
            if m < M-1 and abs(ve[m + 1] - ve[m]) < tol and abs(x0 - x0_prev) < tol:
                pbar.update(M - m - 1)  # Jump to end if converged
                break
    
    # Calculate smoothed probability
    p_smth = 1 / (1 + np.exp((-1) * (b0 + x_smth)))
    
    # Calculate confidence limits for state
    lcl_x = norm.ppf(0.025, x_smth, np.sqrt(v_smth))
    ucl_x = norm.ppf(0.975, x_smth, np.sqrt(v_smth))
    
    # Calculate certainty metric
    median_x = np.median(x_smth)
    certainty = 1 - norm.cdf(median_x * np.ones(K), x_smth, np.sqrt(v_smth))
    
    # Calculate confidence limits for probability
    lcl_p = np.zeros(K)
    ucl_p = np.zeros(K)
    
    if batch_conf_lims:
        # Calculate confidence limits for p_k in batches for better performance
        batch_size = 10  # Process confidence limits in batches
        print('Calculating the pk confidence limits in batches...')
        
        with tqdm(total=K, desc="Confidence Limits Progress") as pbar:
            for i in range(0, K, batch_size):
                end_idx = min(i + batch_size, K)
                for k in range(i, end_idx):
                    lcl_p[k], ucl_p[k] = get_pk_conf_lims(v_smth[k], b0, x_smth[k])
                pbar.update(end_idx - i)
    else:
        # Original sequential calculation
        print('Calculating the pk confidence limits...')
        with tqdm(total=K, desc="Confidence Limits Progress") as pbar:
            for k in range(K):
                lcl_p[k], ucl_p[k] = get_pk_conf_lims(v_smth[k], b0, x_smth[k])
                pbar.update(1)
    
    print('Finished calculating the pk confidence limits.')
    
    return x_smth, v_smth, p_smth, lcl_x, ucl_x, lcl_p, ucl_p, certainty, ve[:m+1]

# Main code
if __name__ == "__main__":
    # Close all figures
    # plt.close('all')
    # base_prob = np.sum(u) / len(u)
    base_prob = 0.5
    
    data = ratwise_lead_vector
    rats = data.keys()
    
    # Dictionary to store t and p_smth per rat
    smoothed_results = {}

    for rat in rats:
        u = data[rat]
        K = len(u)
        
        # Extract binary outcomes
        n = np.zeros(K)
        pt = np.where(u > 0)[0]
        n[pt] = 1
        
        # Run the EM algorithm with the optimized implementation
        # Use fewer iterations for faster execution during testing
        x_smth, v_smth, p_smth, lcl_x, ucl_x, lcl_p, ucl_p, certainty, ve_history = run_em_algorithm(
            n, K, max_iterations=20000, tol=1e-8, base_prob=base_prob, initial_ve=0.005
        )
        
        # Plotting
        fs = 1
        t = np.arange(K) / fs
        tr = np.arange(K-1, -1, -1) / fs
        
        u_plot = np.full(K, np.nan)
        u_plot[pt] = u[pt]
        
        smoothed_results[rat] = {
        't': t,
        'p_smth': p_smth,
        'certainty': certainty}
        
        sns.set(style='ticks')
        sns.set_context('poster')
        plt.figure()
        plt.plot(t, p_smth, 'r', linewidth=1.5)
        plt.fill_between(t, lcl_p, ucl_p, color=(1, 0, 127/255), alpha=0.3)
        plt.ylim([0, 1])
        plt.ylabel('(d) probability (p_{k})')
        plt.tick_params(labelbottom=True)
        plt.xlim([0, t[-1]])
        plt.axhline(y=base_prob, linestyle='--', color='k')
        plt.title(rat)
        plt.show()

#%% Running entropy for choice sequences

import numpy as np
from scipy.stats import entropy

def running_entropy(sequence, window_size, overlap=0.5):
    """
    Compute entropy in a sliding window over the input sequence.

    Parameters:
    - sequence: list or array of well numbers (e.g., [1, 2, 3, 1, ...])
    - window_size: size of the sliding window
    - overlap: fraction of overlap between windows (e.g., 0.5 = 50%)

    Returns:
    - entropies: list of entropy values
    - centers: indices of the center of each window (for plotting)
    """
    entropies = []
    centers = []
    step = int(window_size * (1 - overlap))
    if step < 1:
        raise ValueError("Overlap too high, step becomes zero or negative.")

    for i in range(0, len(sequence) - window_size + 1, step):
        window = sequence[i:i + window_size]
        counts = np.bincount(window, minlength=4)[1:]  # skip index 0
        probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
        ent = entropy(probs, base=2)
        entropies.append(ent)
        centers.append(i + window_size // 2)

    return np.array(entropies), np.array(centers)


def map_triplets_to_sequence(rat_df, column_name='well'):
    """
    Maps each valid, non-overlapping triplet in the well sequence to an integer code (1–8).

    Parameters:
        rat_df (pd.DataFrame): DataFrame with a well sequence.
        column_name (str): Name of the column containing well values.

    Returns:
        list: Sequence of integers (1–8), each representing a valid triplet.
    """

    # Define valid triplets and assign them unique IDs (1–8)
    keys = ['121', '123', '131', '132', '212', '232', '313', '323']
    triplet_id_map = {k: i+1 for i, k in enumerate(keys)}

    # Map of equivalent permutations to canonical triplets
    triplet_map = {
        '213': '123',
        '312': '123',
        '231': '132',
        '321': '132'
    }

    seq = rat_df[column_name].dropna().astype(int).astype(str).tolist()
    triplet_sequence = []

    # Iterate over non-overlapping triplets
    for i in range(0, len(seq) - 2, 3):
        triplet = ''.join(seq[i:i+3])
        canonical = triplet_map.get(triplet, triplet)

        if canonical in triplet_id_map:
            triplet_sequence.append(triplet_id_map[canonical])

    return triplet_sequence


rat1_df = combine_consecutive_wells(sorted_data[71]['ratsamples'][0])
rat2_df = combine_consecutive_wells(sorted_data[71]['ratsamples'][1])

sequence1 = map_triplets_to_sequence(rat1_df, column_name='well')
sequence2 = map_triplets_to_sequence(rat2_df, column_name='well')
window_size = 2

entropy_values1, centers1 = running_entropy(sequence1, window_size, overlap = 0.0)
entropy_values2, centers2 = running_entropy(sequence2, window_size, overlap = 0.0)

import matplotlib.pyplot as plt
plt.plot(entropy_values1)
plt.plot(entropy_values2)
plt.ylabel('Entropy')
plt.xlabel('Time (Index)')
plt.title('Running Entropy of Well Visits')
plt.show()

