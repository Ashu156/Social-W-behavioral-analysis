%%
clear;
close all;
clc;

%% load data

cohort = 'Cohort ER1';
load('cohortER1_pairData_segregated_rewardCont.mat')


%% Leader-follower relationship and lag between arrival of leader and arrival (100% contingency)

global_leadFoll1 = [];
global_leadFoll2 = [];
all_lags = {};

for rt = 1:numel(socialW_100)

    for i = 1:length(socialW_100{rt}) % iterate across

        % Skip rows in errorIndices
        if isempty(socialW_100{rt}(i).ratsamples) || sum(socialW_100{rt}(i).nTransitions) > 400
            continue;  % Skip this iteration
        end

        mysamples = socialW_100{rt}(i).ratsamples;
        ratnums = socialW_100{rt}(i).ratnums;

        if socialW_100{rt}(i).ratnums(1) == min(ratnums)

            table1 = mysamples{1};
            table2 = mysamples{2};

        else
            table1 = mysamples{2};
            table2 = mysamples{1};
        end

        table1.leader = NaN(size(table1,1),1);
        table2.leader = NaN(size(table2,1),1);

        matches1 = table1.match;
        matches2 = table2.match;

        % if sum(matches1) == sum(matches2)
        %     matches = sum(matches1);
        % 
        % else
        %     matches = min(sum(matches1), sum(matches2));
        % end

        match1_idx = find(matches1 == 1);
        match2_idx = find(matches2 == 1);

        if length(match1_idx) == length(match2_idx)
            matches = length(match1_idx);
        else
            matches = min(length(match1_idx), length(match2_idx));
        end

        lag = [];

        for match = 1:matches
            rat1_entry = table1.start(match1_idx(match));
            rat2_entry = table2.start(match2_idx(match));
            delta = rat1_entry - rat2_entry;
            lag = [lag delta];
            if delta < 0
                table1.leader(match1_idx(match)) = 1;
            elseif delta > 0
                table2.leader(match2_idx(match)) = 1;
            end
        end

        if sum(~isnan(table1.leader)) > sum(~isnan(table2.leader))
            global_leadFoll1(i) = 1;
        else sum(~isnan(table1.leader)) < sum(~isnan(table2.leader));
            global_leadFoll2(i) = 1;

        end

        socialW_100{rt}(i).leaDFoll = {table1, table2};
        socialW_100{rt}(i).lagMatches = lag;

        all_lags{i} = lag;

        leader1{i} = table1.leader;
        leader2{i} = table2.leader;

        lead1_proportion(i) = sum(~isnan(table1.leader)) / matches;
        lead2_proportion(i) = sum(~isnan(table2.leader)) / matches;

        

    end

    %% Plot global leader-follower relationship as a function of session #
    % figure;
    % plot(cumsum(global_leadFoll1))
    % hold on
    % plot(cumsum(global_leadFoll2))

    figure;
    plot(lead1_proportion - lead2_proportion)
    hold on
    plot([0 i], [0 0], '--r')
    plot(smoothdata((lead1_proportion - lead2_proportion), "gaussian", 5))
    xlabel('Session #')
    ylabel('Proportion of trials led')
    ylim([-1 1])

    zcd = dsp.ZeroCrossingDetector;
    numZeroCross(rt) = zcd((lead1_proportion - lead2_proportion)');
    numZeroCrossProp(rt) = double(numZeroCross(rt)) / length(socialW_100{rt});

    %% Plot leader-follower relationship for all matches

    % % Determine the maximum length among non-empty vectors
    % maxLen1 = max(cellfun(@numel, leader1));
    % maxLen2 = max(cellfun(@numel, leader2));
    % 
    % % Initialize a matrix to store padded vectors
    % paddedNonEmptyMatrix1 = NaN(maxLen1, numel(maxLen1));
    % paddedNonEmptyMatrix2 = NaN(maxLen2, numel(maxLen2));
    % 
    % % Pad or truncate non-empty vectors and store in the matrix
    % for i = 1:numel(leader1)
    %     currentVector = leader1{i};
    %     paddedNonEmptyMatrix1(1:numel(currentVector), i) = currentVector;
    % end
    % 
    % for i = 1:numel(leader2)
    %     currentVector = leader2{i};
    %     paddedNonEmptyMatrix2(1:numel(currentVector), i) = currentVector;
    % end
    % 
    % % Concatenate non-empty vectors into a single array
    % leadValues1 = paddedNonEmptyMatrix1(:);
    % leadValues2 = paddedNonEmptyMatrix2(:);
    % 
    % figure;
    % plot(cumsum(~isnan(leadValues1)))
    % hold on
    % plot(cumsum(~isnan(leadValues2)))


    %% Plot median of lag between arrival times as a function of session

    % % Initialize an array to store the modes
    % modesArray = zeros(size(all_lags));
    % % Loop through each vector and find the mode
    % for i = 1:numel(all_lags)
    %     modesArray(i) = median(abs(all_lags{i}));
    % end
    % 
    % % figure;
    % plot(smoothdata(modesArray, "gaussian", 5));
    % hold on

    %% Plot distribution of lag between arrival times of a match

    % % Remove empty entries from the cell array
    % nonEmptyCells = all_lags(~cellfun('isempty', all_lags));
    % 
    % % Determine the maximum length among non-empty vectors
    % maxLen = max(cellfun(@numel, nonEmptyCells));
    % 
    % % Initialize a matrix to store padded vectors
    % paddedNonEmptyMatrix = NaN(maxLen, numel(nonEmptyCells));
    % 
    % % Pad or truncate non-empty vectors and store in the matrix
    % for i = 1:numel(nonEmptyCells)
    %     currentVector = nonEmptyCells{i};
    %     paddedNonEmptyMatrix(1:numel(currentVector), i) = currentVector;
    % end
    % 
    % % Concatenate non-empty vectors into a single array
    % allValues = paddedNonEmptyMatrix(:);
    % % Specify the bin edges (adjust as needed)
    % binEdges = linspace(min(allValues), max(allValues), 101);
    % % Create a subplot and plot the histogram
    % subplot(numel(socialW_100),1,rt);
    % hist(abs(allValues), binEdges);
    % xlim([-100 100])

end
% legend()
hold off






%% Leader-follower relationship and lag between arrivals (50% contingency)

% figure;
% hold on
global_leadFoll1 = [];
global_leadFoll2 = [];
all_lags = {};

for rt = 1:numel(socialW_50)

    for i = 1:length(socialW_50{rt}) % iterate across

        % Skip rows in errorIndices
        if isempty(socialW_50{rt}(i).perf) || sum(socialW_50{rt}(i).nTransitions) > 400
            continue;  % Skip this iteration
        end

        mysamples = socialW_50{rt}(i).ratsamples;
        ratnums = socialW_50{rt}(i).ratnums;

        if socialW_50{rt}(i).ratnums(1) == min(ratnums)

            table1 = mysamples{1};
            table2 = mysamples{2};

        else
            table1 = mysamples{2};
            table2 = mysamples{1};
        end

        table1.leader = NaN(size(table1,1),1);
        table2.leader = NaN(size(table2,1),1);

        matches1 = table1.match;
        matches2 = table2.match;

        % if sum(matches1) == sum(matches2)
        %     matches = sum(matches1);
        %
        % else
        %     matches = min(sum(matches1), sum(matches2));
        % end

        match1_idx = find(matches1 == 1);
        match2_idx = find(matches2 == 1);

        if length(match1_idx) == length(match2_idx)
            matches = length(match1_idx);
        else
            matches = min(length(match1_idx), length(match2_idx));
        end


        lag = [];

        for match = 1:matches
            rat1_entry = table1.start(match1_idx(match));
            rat2_entry = table2.start(match2_idx(match));
            delta = rat1_entry - rat2_entry;
            lag = [lag; delta];
            if delta < 0
                table1.leader(match1_idx(match)) = 1;
            elseif delta > 0
                table2.leader(match2_idx(match)) = 1;
            end
        end

        if sum(~isnan(table1.leader)) > sum(~isnan(table2.leader))
            global_leadFoll1(i) = 1;
        else sum(~isnan(table1.leader)) < sum(~isnan(table2.leader));
            global_leadFoll2(i) = 1;

        end

        socialW_50{rt}(i).leadFoll = {table1, table2};
        socialW_50{rt}(i).lagMatches = lag;

        all_lags{i} = lag;

        leader1{i} = table1.leader;
        leader2{i} = table2.leader;

        lead1_proportion(i) = sum(~isnan(table1.leader)) / matches;
        lead2_proportion(i) = sum(~isnan(table2.leader)) / matches;

    end

     %% Plot global leader-follower relationship as a function of session #
    % figure;
    % plot(cumsum(global_leadFoll1))
    % hold on
    % plot(cumsum(global_leadFoll2))

    zcd = dsp.ZeroCrossingDetector;
    numZeroCross(rt) = zcd((lead1_proportion - lead2_proportion)');
    numZeroCrossProp(rt) = double(numZeroCross(rt)) / length(socialW_100{rt});

    figure;
    plot(lead1_proportion - lead2_proportion)
    hold on
    plot([0 length(socialW_50{rt})], [0 0], '--r')
    plot(smoothdata((lead1_proportion - lead2_proportion), "gaussian", 5))
    xlabel('Session #')
    ylabel('Proportion of trials led')
    ylim([-1 1])

    %% Plot leader-follower relationship for all matches

    % % Determine the maximum length among non-empty vectors
    % maxLen1 = max(cellfun(@numel, leader1));
    % maxLen2 = max(cellfun(@numel, leader2));
    % 
    % % Initialize a matrix to store padded vectors
    % paddedNonEmptyMatrix1 = NaN(maxLen1, numel(maxLen1));
    % paddedNonEmptyMatrix2 = NaN(maxLen2, numel(maxLen2));
    % 
    % % Pad or truncate non-empty vectors and store in the matrix
    % for i = 1:numel(leader1)
    %     currentVector = leader1{i};
    %     paddedNonEmptyMatrix1(1:numel(currentVector), i) = currentVector;
    % end
    % 
    % for i = 1:numel(leader2)
    %     currentVector = leader2{i};
    %     paddedNonEmptyMatrix2(1:numel(currentVector), i) = currentVector;
    % end
    % 
    % % Concatenate non-empty vectors into a single array
    % leadValues1 = paddedNonEmptyMatrix1(:);
    % leadValues2 = paddedNonEmptyMatrix2(:);
    % 
    % figure;
    % plot(cumsum(~isnan(leadValues1)))
    % hold on
    % plot(cumsum(~isnan(leadValues2)))

    %% Plot median of lag between arrival times as a function of session

    % % Initialize an array to store the modes
    % modesArray = zeros(size(all_lags));
    % % Loop through each vector and find the mode
    % for i = 1:numel(all_lags)
    %     modesArray(i) = median(abs(all_lags{i}));
    % end
    % 
    % 
    % plot(smoothdata(modesArray, "gaussian", 10));

    %% Plot distribution of lag between arrival times of a match

    % % Remove empty entries from the cell array
    % nonEmptyCells = all_lags(~cellfun('isempty', all_lags));
    % 
    % % Determine the maximum length among non-empty vectors
    % maxLen = max(cellfun(@numel, nonEmptyCells));
    % 
    % % Initialize a matrix to store padded vectors
    % paddedNonEmptyMatrix = NaN(maxLen, numel(nonEmptyCells));
    % 
    % % Pad or truncate non-empty vectors and store in the matrix
    % for i = 1:numel(nonEmptyCells)
    %     currentVector = nonEmptyCells{i};
    %     paddedNonEmptyMatrix(1:numel(currentVector), i) = currentVector;
    % end
    % 
    % % Concatenate non-empty vectors into a single array
    % allValues = paddedNonEmptyMatrix(:);
    % 
    % 
    % % Specify the bin edges (adjust as needed)
    % binEdges = linspace(min(allValues), max(allValues), 101);
    % % Create a subplot and plot the histogram
    % subplot(numel(socialW_50),1,rt);
    % hist(abs(allValues), binEdges);
    % xlim([-100 100])

end

% legend()

%%


