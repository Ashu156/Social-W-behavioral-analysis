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


#%% 

import os
import shutil

# Root directory where all the files are stored
root_dir = r"E:\Jadhav lab data\Behavior\For NWB conversions\CohortAS1\Social W\Opaque"



# File extensions to move to DLC data
dlc_extensions = {'.mp4', '.csv', '.pickle', '.h5', '.jpeg', '.npy'}

# Walk through all directories and files
for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
    if not filenames:
        continue  # Skip if no files in the folder

    # Separate DLC and DIO files
    dlc_files = [f for f in filenames if os.path.splitext(f)[1].lower() in dlc_extensions]
    dio_files = [f for f in filenames if os.path.splitext(f)[1].lower() not in dlc_extensions]

    # Create and move DLC files
    if dlc_files:
        dlc_data_path = os.path.join(dirpath, "DLC data")
        os.makedirs(dlc_data_path, exist_ok=True)
        for file in dlc_files:
            src = os.path.join(dirpath, file)
            dest = os.path.join(dlc_data_path, file)
            shutil.move(src, dest)
            print(f" 📂 Moved to DLC data: {src} -> {dest}")

    # Create and move DIO files
    if dio_files:
        dio_data_path = os.path.join(dirpath, "DIO data")
        os.makedirs(dio_data_path, exist_ok=True)
        for file in dio_files:
            src = os.path.join(dirpath, file)
            dest = os.path.join(dio_data_path, file)
            shutil.move(src, dest)
            print(f" 📂 Moved to DIO data: {src} -> {dest}")

print("File organization complete!")


# %% Delete old tracking data from cohortAS1

import os

# Start from the parent folder to include all subdirectories
root_dir = r"E:\Jadhav lab data\Behavior\For NWB conversions\CohortAS1"

# Keyword to search for
keyword = "DLC_resnet50_SocialWSep18shuffle5"

# Walk through all subdirectories
for folder_path, _, files in os.walk(root_dir):
    for file in files:
        if keyword.lower() in file.lower():  # Case-insensitive match
            file_path = os.path.join(folder_path, file)
            print(f"Found: {file_path}")  # Debugging: Verify files

            try:
                os.chmod(file_path, 0o777)  # Ensure file is writable
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

print("\nCleanup completed!")




# %% Parse files in DLC subfolders into raw and processed

import os
import shutil

# Define the root directory where the search should begin
root_dir = r"E:/Jadhav lab data/Behavior/For NWB conversions/CohortAS1"

# Define criteria for processed files
processed_extensions = {".jpeg", ".npy"}
keyword = "Rat"

# Walk through the directory tree
for folder_path, _, files in os.walk(root_dir):
    # Create 'Processed' and 'Raw' subfolders if they don't exist
    processed_folder = os.path.join(folder_path, "Processed")
    raw_folder = os.path.join(folder_path, "Raw")

    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)

    for file in files:
        file_path = os.path.join(folder_path, file)
        file_ext = os.path.splitext(file)[1].lower()

        # Check if the file meets processed criteria
        if file_ext in processed_extensions or "Rat" in file:
            destination = os.path.join(processed_folder, file)
        else:
            destination = os.path.join(raw_folder, file)

        # Move the file
        try:
            shutil.move(file_path, destination)
            print(f"Moved: {file_path} -> {destination}")
        except Exception as e:
            print(f"Error moving {file_path}: {e}")


# %% Delete empty Processed and Raw subfolders

import os

# Define the root directory where the search should begin
root_dir = r"E:/Jadhav lab data/Behavior/For NWB conversions/CohortAS1"

# Walk through the directory tree
for folder_path, subfolders, files in os.walk(root_dir, topdown=False):  # Process from bottom up
    for subfolder in ["Processed", "Raw"]:
        subfolder_path = os.path.join(folder_path, subfolder)
        
        # Check if the subfolder exists and is empty
        if os.path.exists(subfolder_path) and not os.listdir(subfolder_path):
            try:
                os.rmdir(subfolder_path)
                print(f"Deleted empty folder: {subfolder_path}")
            except Exception as e:
                print(f"Error deleting {subfolder_path}: {e}")


# %%
