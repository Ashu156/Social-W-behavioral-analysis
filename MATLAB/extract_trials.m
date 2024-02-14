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

if exist('ratTrials', 'var')
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
else
    %% Extract trials from socialW_50, socialW_100 cell arrays
    tries = cell(1,1);
    ratTrials_100 = cell(1, 2*numel(socialW_100));

    for pair = 1:numel(socialW_100)

        for i = 1:length(socialW_100{pair}) % iterate across

            % Skip rows in errorIndices
            if isempty(socialW_100{pair}(i).ratsamples) || sum(socialW_100{pair}(i).nTransitions) > 400
                continue;  % Skip this iteration
            end

            mysamples = socialW_100{pair}(i).ratsamples;
            ratnums = socialW_100{pair}(i).ratnums;

            if socialW_100{pair}(i).ratnums(1) == min(ratnums)

                table1 = mysamples{1};
                table2 = mysamples{2};

            else
                table1 = mysamples{2};
                table2 = mysamples{1};
            end
            % candidates are when rat 1 changes wells and rat2 is not leaving
            % his well
            candidates1 = table1.thiswell ~= table1.lastwell & ... % this translates to: when rat X transitions between wells and is not departing from a well occupied by its partner
                table1.hiswell ~= table1.lastwell;

            tries1 = candidates1;
            thiswell1 = table1.thiswell(candidates1);
            thiswell2 = table1.hiswell(candidates1);
            reward1 = table1.Reward(candidates1);
            match1 = table1.match(candidates1);
            well1 = table(thiswell1, thiswell2, reward1, match1, 'VariableNames', {'thiswell', 'hiswell', 'Reward', 'match'});
            wells1 = well1;
            nTrials1 = height(wells1);
            nMatches1 = sum(wells1.match);
            pMatches1 = (nMatches1 / nTrials1);


            trials1{1,i} = tries1;
            trials1{2,i} = wells1;
            trials1{3,i} = pMatches1;

            clearvars thiswell1 thiswell2


            % candidates are when rat 1 changes wells and rat2 is not leaving
            % his well
            candidates2 = table2.thiswell ~= table2.lastwell & ... % this translates to: when rat X transitions between wells and is not departing from a well occupied by its partner
                table2.hiswell ~= table2.lastwell;

            tries2 = candidates2;
            thiswell1 = table2.thiswell(candidates2);
            thiswell2 = table2.hiswell(candidates2);
            reward2 = table2.Reward(candidates2);
            match2 = table2.match(candidates2);
            well2 = table(thiswell1, thiswell2, reward2, match2, 'VariableNames', {'thiswell', 'hiswell', 'Reward', 'match'});
            wells2 = well2;
            nTrials2 = height(wells2);
            nMatches2 = sum(wells2.match);
            pMatches2 = (nMatches2 / nTrials2);


            trials2{1,i} = tries2;
            trials2{2,i} = wells2;
            trials2{3,i} = pMatches2;


        end
        ratTrials_100{2*pair - 1} = trials1;
        ratTrials_100{2*pair} = trials2;

        clearvars trials1 trials2

    end







    % mydir = cd; % directory where the stateScript log files are located
    % save(strcat(mydir, '\CohortAS2_trials_', '.mat'), 'ratTrials')

    % Plot probability of matching


    for kk = 1:numel(ratTrials_100)
        figure;
        hold on;
        for ii = 1:length(ratTrials_100)
            plot(smoothdata(cell2mat({ratTrials_100{ii}{3, :}}), "gaussian", 10));
        end

        xlabel("Session #")
        ylabel("p(Matching)")
        ylim([0 1])
        legend()
    end


     %% Extract trials from socialW_50 cell array
     tries = cell(1,1);
    ratTrials_50 = cell(1, 2*numel(socialW_50));

    for pair = 1:numel(socialW_50)

        for i = 1:length(socialW_50{pair}) % iterate across

            % Skip rows in errorIndices
            if isempty(socialW_50{pair}(i).ratsamples) || sum(socialW_50{pair}(i).nTransitions) > 400
                continue;  % Skip this iteration
            end

            mysamples = socialW_50{pair}(i).ratsamples;
            ratnums = socialW_50{pair}(i).ratnums;

            if socialW_50{pair}(i).ratnums(1) == min(ratnums)

                table1 = mysamples{1};
                table2 = mysamples{2};

            else
                table1 = mysamples{2};
                table2 = mysamples{1};
            end
            % candidates are when rat 1 changes wells and rat2 is not leaving
            % his well
            candidates1 = table1.thiswell ~= table1.lastwell & ... % this translates to: when rat X transitions between wells and is not departing from a well occupied by its partner
                table1.hiswell ~= table1.lastwell;

            tries1 = candidates1;
            thiswell1 = table1.thiswell(candidates1);
            thiswell2 = table1.hiswell(candidates1);
            reward1 = table1.Reward(candidates1);
            match1 = table1.match(candidates1);
            well1 = table(thiswell1, thiswell2, reward1, match1, 'VariableNames', {'thiswell', 'hiswell', 'Reward', 'match'});
            wells1 = well1;
            nTrials1 = height(wells1);
            nMatches1 = sum(wells1.match);
            pMatches1 = (nMatches1 / nTrials1);


            trials1{1,i} = tries1;
            trials1{2,i} = wells1;
            trials1{3,i} = pMatches1;

            clearvars thiswell1 thiswell2


            % candidates are when rat 1 changes wells and rat2 is not leaving
            % his well
            candidates2 = table2.thiswell ~= table2.lastwell & ... % this translates to: when rat X transitions between wells and is not departing from a well occupied by its partner
                table2.hiswell ~= table2.lastwell;

            tries2 = candidates2;
            thiswell1 = table2.thiswell(candidates2);
            thiswell2 = table2.hiswell(candidates2);
            reward2 = table2.Reward(candidates2);
            match2 = table2.match(candidates2);
            well2 = table(thiswell1, thiswell2, reward2, match2, 'VariableNames', {'thiswell', 'hiswell', 'Reward', 'match'});
            wells2 = well2;
            nTrials2 = height(wells2);
            nMatches2 = sum(wells2.match);
            pMatches2 = (nMatches2 / nTrials2);


            trials2{1,i} = tries2;
            trials2{2,i} = wells2;
            trials2{3,i} = pMatches2;


        end
        ratTrials_50{2*pair - 1} = trials1;
        ratTrials_50{2*pair} = trials2;

        clearvars trials1 trials2

    end







    % mydir = cd; % directory where the stateScript log files are located
    % save(strcat(mydir, '\CohortAS2_trials_', '.mat'), 'ratTrials')

    % Plot probability of matching


    for kk = 1:numel(ratTrials_50)
        figure;
        hold on;
        for ii = 1:length(ratTrials_50)
            plot(smoothdata(cell2mat({ratTrials_50{ii}{3, :}}), "gaussian", 10));
        end

        xlabel("Session #")
        ylabel("p(Matching)")
        ylim([0 1])
        legend()
    end
end
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
% end
%% Calculate probability of matching based on the trial-based data

