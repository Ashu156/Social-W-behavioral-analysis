# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:51:16 2025

@author: shukl
"""



#%%

import os
import subprocess
import shutil  # For copying files
import re

#%%
# Define the base directory where .h264 files are located
base_dir = "E:/Jadhav lab data/Behavior/CohortAS6/50%"  # Change this to your actual base directory

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

#%%

# base directory where files are stored
base_dir = "E:/Jadhav lab data/Behavior/CohortAS6/50%/videos_for_DLC2"
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
