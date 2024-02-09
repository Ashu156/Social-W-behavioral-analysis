% Original author: John Blandon, PhD
% The purpose of this script is to iterate across stateScript log files (sessions) to generate a data structure containing relevant information of each behavioral session like e.g., rat names/numbers, run number, total number of well transitions for each rat, etc.
% ... importantly, this data structure will contain DIO-poking data of each
% rat of a given coperating pair in a field called ratsamples

% Modified by: Ashutosh Shukla

%% 
tic;
clear; close all; clc;

%% Load data and preprocess

% mydir = uigetdir();
% mydir='/media/rig5/df7ae62d-51d5-46e9-a6fb-e38430127df9/AS/Social W'; % directory where the stateScript log files are located
mydir = 'E:\Jadhav lab data\Behavior\Cohort AS1\Social W';
 GroupPrefix={'XFN'}; % Cohort names
 fileList = dir(fullfile(mydir, '**\*.*'));  %get list of files and folders in any subfolder

 okfiles = cellfun(@(a) contains(a, 'stateScriptLog') & ~contains(a, 'LED') & ~contains(a, 'Image'), {fileList.name}); % vector of 0s and 1s stating which files are state script logs
 fileList = fileList(okfiles); % get only the state script log files

 okfiles=ones(length(fileList),1); % create vector of 1s

 for i=1:length(fileList) % iterate across files
     try
         % kill test files
         if contains(fileList(i).name,'test')
             okfiles(i)=0; % invalidate and move on
             continue
         else

             % namePat=['(?(log).*>)(?<month>[0-9]+)-(?<day>[0-9]+)-(?<year>[0-9]+)',...
             %     '((?<run>[0-9]+)-(?<rat1>/w+)-(?<rat2>/w+)']; % naming of the files
             namePat = ['(?(log).*>)(?<month>[0-9]+)-(?<day>[0-9]+)-(?<year>[0-9]+)',...
                 '\((?<run>[0-9]+)-(?<rat1>[A-Za-z0-9]+)-(?<rat2>[A-Za-z0-9]+)'];

             myName=fileList(i).name; % get state script file name
             myName(myName=='_')='-'; % replace character
             fileData=regexpi(myName,namePat,'names'); % get file with corresponding name
             % first get cohort
             % cohortNum=find(contains(GroupPrefix,fileData.rat1(1:end-1),'IgnoreCase',true)); % identify cohort number (1-6)
             % if ~isempty(cohortNum)
             fileList(i).cohortname=GroupPrefix; % get cohort name e.g., 'XFK'
             fileList(i).cohortnum= 1; % get cohort number e.g., 6
             fileList(i).runum=str2double(fileData.run); % get run number e.g., 2
             fileList(i).ratnums=[str2double(fileData.rat1(end)),...
                 str2double(fileData.rat2(end))]; % get rat numbers (1-4)
             fileList(i).ratnames={fileData.rat1 fileData.rat2}; % get rat names e.g., XFK1 or XFK2 (this information is obtained from the filename [fileData])
             % else
             %     fprintf('Cant parse this, the filename is... \n     %s \n',...
             %         fileList(i).name);
             %     okfiles(i)=0;
         end
         % end
     catch
         okfiles(i)=0;
         fprintf('Cant parse this, the filename is... \n     %s \n',...
             fileList(i).name);
         disp(i)
     end

 end


%% Extract data from text files

verbose=0;
errorIndices = [];

if true %~isfield(ratinfo,'ratsamples')
    fileList=fileList(cellfun(@(a) ~isempty(a), {fileList.cohortnum}));
    %fileList=fileList(sum([fileList.cohortnum]==[1; 3])'>0); % right now its all but you'll want to pull only certain groups
    opts=setStateScriptOpts();


    ratinfo=fileList; % re-name structure to ratinfo (this will be the final data structure)
    %ratinfo=ratinfo(cellfun(@(a) ~contains(lower(a),'z'),{ratinfo.name}));
    ratinfo=ratinfo(cellfun(@(a) a>0, {ratinfo.bytes}));
    %
    for i=1:length(ratinfo) % iterate across sessions (stateScript log files)
        try

            DataFile = readtable(fullfile(ratinfo(i).folder,ratinfo(i).name), opts);
            % Convert to output type

            myevents={};
            % here we must swap for parseSocialEvents I think...
            % we split up into two sets of events, and we debounce repeated hits on
            % the same well (if the last end and the next start are within a
            % second, its the same sample
            debounce=1; % 1 second debounce...
            [myevents{1},myevents{2}] = parseSocialEvents(DataFile,debounce); %

            if verbose
                % make a huge plot here
                % on left y maybe a moving average of total success rate
                figure;
                for rt=1:length(myevents) % for each rat
                    for it=1:length(myevents{rt}) % for each event
                        if myevents{rt}(it,5)==0
                            plot([myevents{rt}(it,1) myevents{rt}(it,2)],repmat(myevents{rt}(it,3),1,2)+rt*.25,'k-','LineWidth',3);
                            hold on
                        else
                            plot([myevents{rt}(it,1) myevents{rt}(it,2)],repmat(myevents{rt}(it,3),1,2)+rt*.25,'r-','LineWidth',3);
                            hold on
                        end
                    end
                end
            end

            % tables are easier to read...

            for rt=1:2 % iterate across rats
                ratinfo(i).ratsamples{rt}=table(myevents{rt}(:,1),myevents{rt}(:,2),...
                    myevents{rt}(:,3),myevents{rt}(:,4),myevents{rt}(:,5),...
                    myevents{rt}(:,6), myevents{rt}(:,7),'VariableNames',...
                    {'start','end','thiswell','lastwell','Reward','match','Goal Well'});

                % now report
                firsthit=myevents{rt}(diff(myevents{rt}(:,[3 4]),1,2)~=0,:);

            end
        catch exception
            % Handle the error here
            fprintf('Error processing file %s, index %d: %s\n', ratinfo(i).name, i, exception.message);
            % Store the index for later analysis
            errorIndices = [errorIndices, i];  % Assuming errorIndices is initialized outside the loop
        end
    end

end

%% sort by dates 
Dates = {ratinfo.date};  % 'date' field in ratinfo
Dates = cellfun(@(x) datestr(datenum(x), 'dd-mmm-yyyy'), Dates, 'UniformOutput', false); % remove time information

% Convert date strings in ratinfo to serial date numbers
allSerialDates = datenum(Dates, 'dd-mmm-yyyy');

% Find the sorted order indices
[~, sortedIndices] = sort(allSerialDates);

% Use the sorted indices to reorder the rows of ratinfo
ratinfo = ratinfo(sortedIndices);

%% Get data from the google sheet ledger

% for cohort 1

google_sheet_url ='https://docs.google.com/spreadsheets/d/e/2PACX-1vSTWbeF7xlI00dSLofuNj0StunEG0K5rT4Q_sSRVPIfIibaKQCXHl1lbJbwSENqyJiuR0hYIrP_kW0d/pub?gid=190373958&single=true&output=csv';

% for cohort 2
% google_sheet_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSTWbeF7xlI00dSLofuNj0StunEG0K5rT4Q_sSRVPIfIibaKQCXHl1lbJbwSENqyJiuR0hYIrP_kW0d/pub?gid=483643765&single=true&output=csv';

% Specify the local file path where you want to save the CSV file
local_csv_file_path = 'Cohort AS1.csv'; % Replace with your desired file path

% Use websave to download the Google Sheet as a CSV
websave(local_csv_file_path, google_sheet_url);

% Specify the starting row you want to read
start_row = 229; % Replace with your desired start row

% Load the CSV file into a table
google_sheet_table = readtable(local_csv_file_path, 'Range', ['A' num2str(start_row)]);

%% Get rid of runs that are not in the google sheet

% Extract logfilenames from ratinfo
ratinfo_logfilenames = {ratinfo.name};

% Remove '.stateScriptLog' from the end of elements in anotherList
ratinfo_logfilenames = cellfun(@(x) regexprep(x, '\.stateScriptLog$', ''), ratinfo_logfilenames, 'UniformOutput', false);

% Extract logfilenames from google_sheet_table
google_sheet_logfilenames = google_sheet_table.Var2; % logfilenames are in column Var2

% Find and remove the empty cells
google_sheet_logfilenames = google_sheet_table.Var2;

% List to compare
matches = ismember(ratinfo_logfilenames, google_sheet_logfilenames);

% Find indices of non-matching elements
nonMatchingIndices = find(~matches);

ratinfo(nonMatchingIndices, :) = [];

%% Update reward contingency of runs

rewardCont = google_sheet_table.Var13;

% Define a function to clean and convert the strings
cleanAndConvert = @(str) str2double(strtrim(strrep(str, '%', '')));
% get
resultArray = cellfun(cleanAndConvert, rewardCont);

for i = 1:numel(ratinfo)
    ratinfo(i).rewardContingency = resultArray(i);
end

%% Update whether a run is Opaque control or not

opqCont = google_sheet_table.Var3;


for i = 1:numel(ratinfo)
    ratinfo(i).opqCont = opqCont(i);
end

%% Get rid of runs marked to be discraded in the google sheet

% Find rows to be discarded (marked as 'Yes' by experimenterunde rthe column 'Discard run')
rows_to_discard = strcmpi(google_sheet_table.Var18, 'Discard run') | strcmpi(google_sheet_table.Var18, 'Yes');

% Get the log filenames for discarded rows
discarded_log_filenames = google_sheet_table{rows_to_discard, 'Var2'};

% Find rows to discard based on filenames
rows_to_discard_filenames = ismember(ratinfo_logfilenames, discarded_log_filenames);
% Remove corresponding rows from ratinfo
ratinfo(rows_to_discard_filenames, :) = [];

% Find rows to discard based on filenames
% rows_to_discard_filenames = ismember(google_sheet_logfilenames, discarded_log_filenames);
% Remove corresponding rows from google_sheet_table
% google_sheet_table(rows_to_discard_filenames, :) = [];

%%
% for each session, keep track of your peers well at all times:

for i=1:length(ratinfo) % iterate across sessions

    % Skip rows in errorIndices
    if isempty(ratinfo(i).ratsamples)
        continue;  % Skip this iteration
    end

    for tr=1:height(ratinfo(i).ratsamples{1}) % for each poke sample
        % find his last well poke
        hislastpoke=find(ratinfo(i).ratsamples{2}.start<ratinfo(i).ratsamples{1}.start(tr),1,'last'); % find last poke-in event of partner
        if ~isempty(hislastpoke)
            ratinfo(i).ratsamples{1}.hiswell(tr)=ratinfo(i).ratsamples{2}.thiswell(hislastpoke); % extract well of such poke-in event 
        else
            ratinfo(i).ratsamples{1}.hiswell(tr)=nan; 
        end
    end

    for tr=1:height(ratinfo(i).ratsamples{2}) % repeat code line 111-120 for the other rat
        hislastpoke=find(ratinfo(i).ratsamples{1}.start<ratinfo(i).ratsamples{2}.start(tr),1,'last');
        if ~isempty(hislastpoke)
            ratinfo(i).ratsamples{2}.hiswell(tr)=ratinfo(i).ratsamples{1}.thiswell(hislastpoke);
        else
            ratinfo(i).ratsamples{2}.hiswell(tr)=nan;
        end
    end

end

% for each session, keep track of your peers well at all times:


% lets tabulate in the ratinfo struct
%%

 %%%%%%%%%%% ANALYSIS STARTS HERE %%%%%%%%%%%%%%%%


 % genotypetable=table([2 3 5 7],[8 9],...
 %    'VariableNames',{'WT','FX'}); % genotype information for all 6 cohorts



for i=1:length(ratinfo) % iterate across poke/un-poke samples

    % Skip rows in errorIndices
    if isempty(ratinfo(i).ratsamples)
        continue;  % Skip this iteration
    end

    ratinfo(i).tries=[nan nan]; % pre-allocate tries; tries are the "trials" 
    ratinfo(i).matches=[nan nan]; % pre-allocate matches; matches are "correct" choices
    for j=1:2 % iterate across rats
        mysamples=ratinfo(i).ratsamples{j};
        % candidates are when i change wells and i am not leaving his well
        candidates=mysamples.thiswell~=mysamples.lastwell & ... % this translates to: when rat X transitions between wells and is not departing from a well occupied by its partner
            mysamples.hiswell~=mysamples.lastwell;
        % wins are when i change wells and my well is now his well
        wins=mysamples.thiswell~=mysamples.lastwell & ...
            mysamples.hiswell==mysamples.thiswell; % this is consider a match (when rat transitions and it visits a well occupied by its partner)
        ratinfo(i).tries(j)=sum(candidates); % total number of trials for that rat 
        ratinfo(i).matches(j)=sum(wins); % total number of correct choices
        ratinfo(i).wins(j)=sum(mysamples.match); % total number of matches for the pair
        ratinfo(i).nTransitions(j)=sum(mysamples.thiswell~=mysamples.lastwell); % total number of well transitions 

    end
  
    % session duration and total arm transitions
    ratinfo(i).duration=(max([ratinfo(i).ratsamples{1}.end(end) ratinfo(i).ratsamples{2}.end(end)])-...
        min([ratinfo(i).ratsamples{1}.start(1) ratinfo(i).ratsamples{2}.start(1)]))/60; % duration of the session

end

%% Assign genopair IDs

for i = 1:length(ratinfo)
    % Check if both rats are WT
    if all(ismember([2, 4], ratinfo(i).ratnums)) || all(ismember([5, 7], ratinfo(i).ratnums))
        ratinfo(i).genopair = 2; % Both rats are WT
    % Check if both rats are FX
    elseif all(ismember([1, 3], ratinfo(i).ratnums))
        ratinfo(i).genopair = 0; % Both rats are FX
    else
        ratinfo(i).genopair = 1; % One rat is WT and the other is FX
    end
end


%% Change the ID flips during experiments
% % Assuming column S contains the flip information ('Yes' or 'No')
% rat_flips = strcmpi(google_sheet_table.Var15, 'Yes');
% 
% % Iterate through each row
% for i = 1:length(ratinfo)
%     % Check if the rat IDs need to be flipped
%     if rat_flips(i)
%         % Flip the rat IDs
%         flipped_ratnums = fliplr(ratinfo(i).ratnums);
%         % Update the ratnums field
%         ratinfo(i).ratnums = flipped_ratnums;
% 
%         % Flip the rat names
%         flipped_ratnames = fliplr(ratinfo(i).ratnames);
%         % Update the ratnames field
%         ratinfo(i).ratnames = flipped_ratnames;
%     end
% end

%% Only for Cohort AS1 where dates were wrong in the date field

entries = {ratinfo.name};
datePattern = '(\d{2}-\d{2}-\d{4})';
dates = cellfun(@(entry) regexp(entry, datePattern, 'match', 'once'), entries, 'UniformOutput', false);
[ratinfo.date] = deal(dates{:});

%%
Dates = {ratinfo.date};  % 'date' field in ratinfo
Dates = cellfun(@(x) datestr(datenum(x), 'dd-mmm-yyyy'), Dates, 'UniformOutput', false); % remove time information
uniqueDates= unique(Dates); % extract unique dates
% dates = datetime(Dates, 'InputFormat', 'dd-mmm-yyyy');

% Convert date strings to serial date numbers
serialDates = datenum(uniqueDates, 'dd-mmm-yyyy');

% Create a matrix with two columns: one for year-month and one for day
dateMatrix = [year(serialDates), month(serialDates), day(serialDates)];

% Sort by year-month and then by day within each month
sortedDateMatrix = sortrows(dateMatrix, [1, 2, 3]);

% Convert back to date strings
sortedUniqueDates = cellstr(datestr(datenum(sortedDateMatrix), 'dd-mmm-yyyy'));

% Display the sorted result
disp('Sorted Unique Dates (First by Month, Then by Day):');
disp(sortedUniqueDates);


% Assuming Dates and sortedUniqueDates are your cell arrays

% Initialize cell array to store indices
indicesCellArray = cell(size(sortedUniqueDates));

% Find indices of each element in sortedUniqueDates in Dates
for i = 1:numel(sortedUniqueDates)
    indicesCellArray{i} = find(ismember(Dates, sortedUniqueDates{i}));
end

date = datestr(datetime);
date = strrep(date, ':', '_');
% save(strcat(mydir, '\Cohort AS1_', date, '.mat'), 'ratinfo', 'errorIndices')


%% Sort by pair ID

% pairsToCheck = {[3, 2], [7, 5], [9, 8]};
pairsToCheck = {[2, 4], [1, 3]};
for pair = 1:numel(pairsToCheck)

    pairToCheck = pairsToCheck{pair};
    pairData{pair} = [];
    emptyIndices{pair} = [];


    for i = 1:numel(uniqueDates)
        winsArray{i} = [];
        pairfound{i} = [];

        for j = 1:numel(indicesCellArray{i})

            pairmatch = isequal(ratinfo(indicesCellArray{i}(j)).ratnums, pairToCheck)...
                || isequal(ratinfo(indicesCellArray{i}(j)).ratnums, fliplr(pairToCheck));

            pairfound{i} = [pairfound{i}, pairmatch];
            winsValue = ratinfo(indicesCellArray{i}(j)).wins;
            winsArray{i} = [winsArray{i}, winsValue];
        end

    end




    % Sort by pair ID (create a different struct array for each pair)


    pairmatch = [];
    for i = 1:numel(uniqueDates)

        for j = 1:numel(indicesCellArray{i})

            pairmatch = [pairmatch, isequal(ratinfo(indicesCellArray{i}(j)).ratnums, pairToCheck)...
                || isequal(ratinfo(indicesCellArray{i}(j)).ratnums, fliplr(pairToCheck))];

        end

    end

    pairmatch = logical(pairmatch);
    pairData{pair} = [pairData{pair}, ratinfo(pairmatch)];

    % Check whether any of the foeld entries are blank
    temp_data = ratinfo(pairmatch);
    % Initialize a logical array to store results
    isEmptyArray = false(size(temp_data));

    % Loop through each struct in the array
    for i = 1:numel(ratinfo(pairmatch))
        % Get the fieldnames of the current struct
        fieldnamesArray = fieldnames(temp_data(i));

        % Loop through each field

        % Check if the field is empty
        isEmptyField = isempty(temp_data(i).ratsamples);

        % Update the logical array
        isEmptyArray(i) = isEmptyArray(i) || isEmptyField;

    end

    emptyidx = find(isEmptyArray);
    emptyIndices{pair} = [emptyIndices{pair}, emptyidx];

end



% save(strcat(mydir, '\Cohort 2 _', 'pairData_', date, '.mat'), 'pairsToCheck', 'pairData', 'emptyIndices')
save(strcat('Cohort AS1_', 'pairData_', date, '.mat'), 'pairsToCheck', 'pairData')


%% Sort by rat ID

% ratsToCheck = [2, 3, 5, 7, 8, 9]; % for cohort AS2
ratsToCheck = [1, 2, 3, 4];    % for cohort AS1

for rt = 1:length(ratsToCheck)
    ratToCheck = ratsToCheck(rt);
    ratfound{rt} = [];

    for i = 1:numel(uniqueDates)

        for j = 1:numel(indicesCellArray{i})

            ratmatch = ismember(ratinfo(indicesCellArray{i}(j)).ratnums, ratToCheck);
            ratfound{rt} = [ratfound{rt}; ratmatch];

        end

    end
    ratfound{rt} = logical(ratfound{rt});
    [rowIndices, colIndices] = find(ratfound{rt}== 1);

    for rr = 1:numel(rowIndices)

        [sortedRowIndices, sortOrder] = sort(rowIndices);
        sortedColIndices = colIndices(sortOrder);
        % Skip rows in errorIndices
        if isempty(ratinfo(sortedRowIndices(rr)).ratsamples)
            continue;  % Skip this iteration
        end

        
        
        singleRat(rr).name = ratinfo(sortedRowIndices(rr)).name;
        singleRat(rr).folder = ratinfo(sortedRowIndices(rr)).folder;
        singleRat(rr).bytes = ratinfo(sortedRowIndices(rr)).bytes;
        singleRat(rr).isdir = ratinfo(sortedRowIndices(rr)).isdir;
        singleRat(rr).datenum = ratinfo(sortedRowIndices(rr)).datenum;
        singleRat(rr).cohortname = ratinfo(sortedRowIndices(rr)).cohortname;
        singleRat(rr).cohortnum = ratinfo(sortedRowIndices(rr)).cohortnum;
        singleRat(rr).runum = ratinfo(sortedRowIndices(rr)).runum;
        singleRat(rr).ratnames = ratinfo(sortedRowIndices(rr)).ratnames{sortedColIndices(rr)};
        singleRat(rr).ratsamples = ratinfo(sortedRowIndices(rr)).ratsamples{sortedColIndices(rr)};
        singleRat(rr).tries = ratinfo(sortedRowIndices(rr)).tries(sortedColIndices(rr));
        singleRat(rr).matches = ratinfo(sortedRowIndices(rr)).matches(sortedColIndices(rr));
        singleRat(rr).wins = ratinfo(sortedRowIndices(rr)).wins(sortedColIndices(rr));
        singleRat(rr).nTransitions = ratinfo(sortedRowIndices(rr)).nTransitions(sortedColIndices(rr));
        singleRat(rr).duration = ratinfo(sortedRowIndices(rr)).duration;
        singleRat(rr).genopair = ratinfo(sortedRowIndices(rr)).genopair;
        singleRat(rr).rewardContingency= ratinfo(sortedRowIndices(rr)).rewardContingency;
    end
    RatData{rt} = singleRat;
    clearvars singleRat
end



% mydir = cd;
% save(strcat(mydir, '\Cohort AS1_', 'ratwiseData', '_', date, '.mat'), 'RatData')

toc;

%% end of script