%%
% This script extracts the trials from the session runs of each rat and
% saves it as a .csv file (cohortID_ratID_sessionID.csv).
% Rat IDs are as in the ratinfo struct array

% Author: Ashutosh Shukla

%%
clear;
close all;
clc;

%% This extracts trials from RatData struct array if it exists

tries = cell(1,1);
ratTrials = cell(1, numel(RatData));

for rt = 1:numel(RatData)
    temp_data = RatData{rt};
    trials = cell(3, length(temp_data));

    for i = 1:length(temp_data) % iterate across poke/un-poke samples

        % Skip rows in errorIndices
        if isempty(temp_data(i).ratsamples)
            continue;  % Skip this iteration
        end


        mysamples=temp_data(i).ratsamples;
        % candidates are when i change wells and i am not leaving his well
        candidates=mysamples.thiswell~=mysamples.lastwell & ... % this translates to: when rat X transitions between wells and is not departing from a well occupied by its partner
            mysamples.hiswell~=mysamples.lastwell;

        tries = candidates;
        well1 = mysamples.thiswell(candidates);
        well2 = mysamples.hiswell(candidates);
        reward = mysamples.Reward(candidates);
        match = mysamples.match(candidates);
        well = table(well1, well2, reward, match, 'VariableNames', {'thiswell', 'hiswell', 'Reward', 'match'});
        wells = well;
        nTrials = height(wells);
        nMatches = sum(wells.match);
        pMatches = (nMatches / nTrials);


        trials{1,i} = tries;
        trials{2,i} = wells;
        trials{3,i} = pMatches;
    end
    ratTrials{rt} = trials;
end

% mydir = cd; % directory where the stateScript log files are located
% save(strcat(mydir, '\CohortAS2_trials_', '.mat'), 'ratTrials')

% Plot probability of matching
figure('Color', [1 1 1]);
hold on;
for ii = 1:length(ratTrials)
    plot(smoothdata(cell2mat({ratTrials{1,ii}{3, :}}), "gaussian", 10));
end

xlabel("Session #")
ylabel("p(Matching)")
ylim([0 1])
legend()

%% Extract trials from socialW_50, socialW_100 cell arrays

% tries = cell(1,1);
% ratTrials = cell(1, numel(socialW_100));


% for pair = 1:1 %numel(socialW_100)
    temp_data = socialW_100{1};
    % trials = cell(3, length(temp_data));

    ratsToCheck = sort(temp_data(1).ratnums);

    ratnums = cell2mat({temp_data.ratnums});
    rat1ID = ratnums(1,1:2:end)';
    rat2ID = ratnums(1,2:2:end)';

    for i = 1:length(temp_data) % iterate across poke/un-poke samples

        % Skip rows in errorIndices
        % if isempty(temp_data(i).ratsamples)
        %     continue;  % Skip this iteration
        % end

        

        
% end
        %% ratData = {};

        for rt = 1:2


            mysamples=temp_data(i).ratsamples{rt};
            % candidates are when i change wells and i am not leaving his well
            candidates=mysamples.thiswell~=mysamples.lastwell & ... % this translates to: when rat X transitions between wells and is not departing from a well occupied by its partner
                mysamples.hiswell~=mysamples.lastwell;

            tries = candidates;
            well1 = mysamples.thiswell(candidates);
            well2 = mysamples.hiswell(candidates);
            reward = mysamples.Reward(candidates);
            match = mysamples.match(candidates);
            well = table(well1, well2, reward, match, 'VariableNames', {'thiswell', 'hiswell', 'Reward', 'match'});
            wells = well;
            nTrials = height(wells);
            nMatches = sum(wells.match);
            pMatches = (nMatches / nTrials);


            trials{1,i} = tries;
            trials{2,i} = wells;
            trials{3,i} = pMatches;
        end
        ratTrials{rt} = trials;
  

    mydir = cd; % directory where the stateScript log files are located
    % save(strcat(mydir, '\CohortAS2_trials_', '.mat'), 'ratTrials')

    % Plot probability of matching
    figure('Color', [1 1 1]);
    hold on;
    for ii = 1:length(ratTrials)
        plot(smoothdata(cell2mat({ratTrials{1,ii}{3, :}}), "gaussian", 10));
    end

    xlabel("Session #")
    ylabel("p(Matching)")
    ylim([0 1])
    legend()

    %% Make a supertable for each rat

    allTables = cell(1, numel(ratTrials));

    for rt = 1:numel(ratTrials)
        dt = ratTrials{rt};
        concatenatedTable = [];
        for col = 1:size(dt, 2)
            if istable(dt{2, col})

                concatenatedTable = vertcat(concatenatedTable, dt{2, col});
            end

        end
        allTables{rt} = concatenatedTable;
    end

    for i = 1:numel(allTables)
        temp_tbl = allTables{i};

        temp_tbl.prefer = zeros(size(temp_tbl, 1), 1);  % Initialize with zeros
        temp_tbl.prefer(temp_tbl.thiswell == temp_tbl.hiswell) = 1;
        allTables{i} = temp_tbl;

    end

    %% save tables as .csv fils for further processing in python

    % Assuming RatData is your cell array
    for cellIndex = 1:length(RatData)
        % Access the struct array from the cell array
        currentStructArray = RatData{cellIndex};

        % Access the 'ratsamples' field
        % ratsamplesField = currentStructArray.ratsamples;

        % Iterate through the tables in the 'ratsamples' field
        for tableIndex = 1:numel(currentStructArray)
            % Access the current table
            if ~isempty(currentStructArray(tableIndex).ratsamples)
                currentTable = currentStructArray(tableIndex).ratsamples;
            end

            % Generate a unique file name (adjust as needed)
            fileName = sprintf('cohortAS2_rat_%d_session%d.csv', cellIndex, tableIndex);


            % Save the table as a CSV file
            writetable(currentTable, strcat(mydir, fileName));

            fprintf('Saved table %d from cell %d as %s\n', tableIndex, cellIndex, fileName);
        end
    end

    %% Calculate probability of matching based on the trial-based data

