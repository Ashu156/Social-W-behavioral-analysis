%% Description: 
% This function (1) uses linear interpolation to fill out values that DLC model is uncertain about (2) organizes data based on bodyparts , and (3) segregates data according to rat

% Input: Raw DLC csv file 
% Output: Preprocessed csv files, 1 for each rat and columns organized by
% bodyparts

% CHECK THE FOLLOWING LINES BELOW RUNNING THIS. VARIABLES IN THESE LINES
% NEED TO BE ADJUSTED AS PER COHORT REQUIREMENTS.
% 37-42, 61, 73

% Written by:  Edward L. Rivera
% Modified by: Ashutosh Shukla

%% Load tracked .csv files

tic; % start timer

clear;
close all; 
clc;

%%
projectFolder='E:\Jadhav lab data\Behavior\Cohort 2\Social W'; % master folder with all subfolders
% projectFolder='F:\Jadhav lab data\Behavior\Cohort N\Social W'; % master folder with all subfolders
cohortfolder = projectFolder; % define cohort folder (here same as master folder since I have only one cohort)
cd(string(cohortfolder));  % change current directory to the master folder


directory = dir(cohortfolder); % get all the subfolders in the master folder
directory(1:2,:) = [];    % Get rid of the top two entries as they are blank
directory(57:end, :) = []; % This is required if there are other subfolders in the master folder


%% Define a few parameters to be used subsequently

threshold = 0.9; % cutoff threshold
bodyparts = ["" "snout" "head1" "head2" "body1" "body2" "body3" "tail base" "tail tip"]; % body parts tracked
rat1_ID = 'FXM108'; % ID of the 1st rat
rat2_ID = 'FXM109'; % ID of the 2nd rat
rat1_col_end = 25;   % depends on the number of body parts tracked (so edit this carefully)
rat2_col_start = 26; % depends on the number of body parts tracked (so edit this carefully)

%% Process data and write new .csv files

for day = 1:height(directory) % iterate across directory 
    
     sessions = dir(directory(day).name); % session ID (same as date of experiment)
     sessions(1:2) = []; % eliminate first two elements, which do not contain anything 
     sessions = sessions(contains({sessions.name},'.csv') & contains({sessions.name}, rat1_ID) &...
                         contains({sessions.name}, rat2_ID),:); % get only those files with .csv extension and of given animals 

     if ~isempty(sessions) % if you have data from run experiments 
         for session = 1:height(sessions) % iterate across those session/files 
        
             filename_r1 = [cohortfolder filesep directory(day).name filesep sessions(session).name(1:29) '-Rat1-SnoutTracking.csv']; % get file name of DLC csv file
             filename_r2 = [cohortfolder filesep directory(day).name filesep sessions(session).name(1:29) '-Rat2-SnoutTracking.csv'];
             try
                 % if ~isfile(filename_r1) || ~isfile(replace(filename_r1,"Snout","All")) % evaluate if file has already been proccesed 
                    % Adjust the length of file name for every cohort
                    file = [cohortfolder filesep directory(day).name filesep sessions(session).name]; % get filename of unprocessed DLC csv file
                    data = readtable(file); % read data from DLC csv file 
        
                    rat1 = data(:, 1:rat1_col_end); % split DLC csv file into rat 1 and rat2
                    rat2 = [data(:, 1) data(:, rat2_col_start:end)];
                    
                    
                    rat1.Properties.VariableNames(1) = "frame"; rat1.frame = rat1.frame+1; % Add frame column 
                    rat2.Properties.VariableNames(1) = "frame"; rat2.frame = rat2.frame+1;
                    
                    

                    for bodypart = 2:9 % iterate across body parts 
                        rat1 = mergevars(rat1, [bodypart bodypart+1 bodypart+2], 'NewVariableName', bodyparts(bodypart)); % merge bodyparts-specific x,y,p data
                        rat2 = mergevars(rat2, [bodypart bodypart+1 bodypart+2], 'NewVariableName', bodyparts(bodypart));
                    
                        start1 = find(~isnan(rat1{:, bodypart}(:, 1))); start1 = start1(1); % look when data starts
                        start2 = find(~isnan(rat2{:, bodypart}(:, 1))); start2 = start2(1);
                    
                        end1 = find(~isnan(rat1{:, bodypart}(:, 1))); end1 = end1(end); % look when data ends for that animal 
                        end2 = find(~isnan(rat2{:, bodypart}(:, 1))); end2 = end2(end);
                    
                        interpolate1 = find(rat1{:,bodypart}(:, 3) < threshold); % get the indexes for those points that DLC model is ~uncerstain about prob. <0.9 
                        interpolate2 = find(rat2{:,bodypart}(:, 3) < threshold);

                        rat1{interpolate1, bodypart}(:, 1:2) = nan; % nan those values
                        rat2{interpolate2, bodypart}(:, 1:2) = nan;
                        % 
                        % rat1{start1:end1,bodypart}(:, 1:2) = fillmissing(rat1{start1:end1,bodypart}(:,1:2), "linear"); % use linear interpolation to fill those nan values 
                        % rat2{start2:end2,bodypart}(:, 1:2) = fillmissing(rat2{start2:end2,bodypart}(:,1:2), "linear");
                    
                    end
                    
                   
                    rat1Data = table(rat1.snout(:, 1:2), 'VariableNames', "xy");
                    rat2Data = table(rat2.snout(:, 1:2), 'VariableNames', "xy");
                    rat1Data = splitvars(rat1Data, "xy", 'NewVariableNames', ["x","y"]);
                    rat2Data = splitvars(rat2Data, "xy", 'NewVariableNames', ["x","y"]);
                    writetable(rat1Data,filename_r1);
                    writetable(rat2Data,filename_r2);
                    writetable(rat1,replace(filename_r1, "SnoutTracking", "AllTracking"));
                    writetable(rat2,replace(filename_r2, "SnoutTracking", "AllTracking"));
                 % end
             catch
                 continue
             end
         end
     end
end

toc;

%%