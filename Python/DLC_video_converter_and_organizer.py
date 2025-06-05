# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:51:16 2025

@author: shukl
"""



#%% Import libraries

import os
import subprocess
import shutil  
import re

#%% Convert .h64 fles into  .mp4 format

# Base directory where .h264 files are located
base_dir = "E:/Jadhav lab data/Behavior/Observational learning/Demonstrator training"  

# Define the target directory for DLC videos
dlc_folder = os.path.join(base_dir, "videos_for_DLC")
os.makedirs(dlc_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Specify the full path to FFmpeg
ffmpeg_path = r"C:\Users\shukl\ffmpeg\bin\ffmpeg.exe"

# Loop through all subdirectories and find .h264 files
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".h264"):
            h264_path = os.path.join(root, file)  # Full path to .h264 file
            mp4_path = os.path.join(root, file.replace(".h264", ".mp4"))  # Output .mp4 path

            # **Skip conversion if .mp4 already exists**
            if os.path.exists(mp4_path):
                print(f"⏭️ Skipping: {h264_path} (MP4 already exists)")
                continue

            # FFmpeg command with full path
            cmd = [
                ffmpeg_path, "-y", "-i", h264_path, "-c:v", "libx264", "-preset", "fast", "-crf", "23", mp4_path
            ]

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"✅ Converted: {h264_path} -> {mp4_path}")

                # Copy the converted .mp4 file to the "videos_for_DLC" folder
                dlc_path = os.path.join(dlc_folder, os.path.basename(mp4_path))
                shutil.copy(mp4_path, dlc_path)
                print(f"📂 Copied: {mp4_path} -> {dlc_path}")

            except subprocess.CalledProcessError as e:
                print(f"❌ Error converting {h264_path}: {e.stderr.decode()}")

print("🎥✅ All conversions completed! All MP4 files are now in 'videos_for_DLC'.")


#%% Arrange the .mp4 videos in subfolders

# base directory where files are stored
base_dir = "E:/Jadhav lab data/Behavior/Observational learning/Demonstrator training/videos_for_DLC"
# Regular expression to extract dates in the format MM-DD-YYYY
date_pattern = re.compile(r"\d{2}-\d{2}-\d{4}")

# Loop through all files in the base directory
for file in os.listdir(base_dir):
    file_path = os.path.join(base_dir, file)

    # Ensure it's a file (not a directory)
    if os.path.isfile(file_path):
        match = date_pattern.search(file)

        if match:
            date_str = match.group()  # Extracted date from filename
            date_folder = os.path.join(base_dir, date_str)  # Folder path for this date
            
            # Create the subfolder if it doesn't exist
            os.makedirs(date_folder, exist_ok=True)

            # Move the file to the respective date folder
            dest_path = os.path.join(date_folder, file)
            shutil.move(file_path, dest_path)

            print(f"📂 Moved: {file} -> {date_folder}")

print("✅ All files have been organized into date-based subfolders!")

# %% Copy files from Social W experiments for NWB conversions

import os
import shutil
import re

# Define source and destination directories
source_dir = "E:/Jadhav lab data/Behavior/CohortAS1/Social W"  # Change this to your main data directory
dest_dir = "E:/Jadhav lab data/Behavior/For NWB conversions/CohortAS1/Social W"  # Change this to where you want rat-pair folders


# Regular expression to extract rat pair names from filenames
rat_pattern = re.compile(r'\(\d+-(FXM\d+|XFN\d+)-?(FXM\d+|XFN\d+)?\)')

# Walk through the nested subfolders
for root, _, files in os.walk(source_dir):
    for file in files:
        match = rat_pattern.search(file)
        if match:
            rat1, rat2 = match.groups()
            if rat2:  
                rat_pair = "-".join(sorted([rat1, rat2]))  # Sort to handle XFN2-XFN4 == XFN4-XFN2
            else:
                rat_pair = rat1  

            # Extract the percentage and date subfolders
            relative_path = os.path.relpath(root, source_dir)  
            parts = relative_path.split(os.sep)
            if len(parts) < 2:
                continue  

            percentage_folder = parts[0]  # reward contingency folder
            date_folder = parts[1]  # 06-20-2023

            # Corrected folder structure
            rat_folder = os.path.join(dest_dir, percentage_folder, rat_pair, date_folder)
            os.makedirs(rat_folder, exist_ok=True)  

            # Copy the file
            shutil.copy(os.path.join(root, file), os.path.join(rat_folder, file))
            print(f"📂 Copied {file} to {rat_folder}")


