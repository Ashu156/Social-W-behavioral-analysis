%%
clear;
close all;
clc;

%%
Data = cell(1, numel(pairData));

for j = 1:numel(pairData)

    rat = pairData{j};

    for i = 1:length(rat) % iterate across poke/un-poke samples

        % Skip rows in errorIndices
        if isempty(rat(i).ratsamples)
            continue;  % Skip this iteration
        end

        rat(i).tries=[nan nan]; % pre-allocate tries; tries are the "trials"
        rat(i).matches=[nan nan]; % pre-allocate matches; matches are "correct" choices
        for rt = 1:2
            mysamples = rat(i).ratsamples{rt};
            % candidates are when i change wells and i am not leaving his well
            candidates = mysamples.thiswell~=mysamples.lastwell; % this translates to: when rat X transitions between wells and is not departing from a well occupied by its partner
            % wins are when i change wells and my well is now his well
            % wins = mysamples.thiswell~=mysamples.lastwell & ...
            %     mysamples.hiswell==mysamples.thiswell; % this is consider a match (when rat transitions and it visits a well occupied by its partner)
            % ratinfo(i).tries(rt) = sum(candidates); % total number of trials for that rat
            rat(i).matches(rt) = sum(mysamples.match); % total number of correct choices
            rat(i).wins(rt) = sum(mysamples.Reward); % total number of matches for the pair
            rat(i).nTransitions(rt) = sum(mysamples.thiswell~=mysamples.lastwell); % total number of well transitions
        end


        % session duration and total arm transitions
        % ratinfo(i).duration=(max([ratinfo(i).ratsamples{1}.end(end) ratinfo(i).ratsamples{2}.end(end)])-...
        %     min([ratinfo(i).ratsamples{1}.start(1) ratinfo(i).ratsamples{2}.start(1)]))/60; % duration of the session

    end

    Data{j} = rat;

end

%% Calculate performance of each pair

nPairs = 3;
performance = cell(1,nPairs);

for pair = 1:nPairs
    for run = 1:numel(Data{1, pair})
        if isempty(Data{1, pair}(run).ratsamples)
            continue;  % Skip this iteration
        end
        tr = sum(Data{1,pair}(run).nTransitions);
        
        % Additional check: calculate perf only when tr < 400 (eliminate runs where one of the wells is malfunctioning)
        % if tr < 400
            Data{1,pair}(run).perf = ((Data{1,pair}(run).matches(1) / tr) / 0.5) * 100;
        % else
            % Data{1,pair}(run).perf = [];  % or any other value to indicate not applicable
        % end
    end
end

clearvars rat;

%% Segregate social W from opaque control runs and plot mean daily performance

socialW_data = cell(1, nPairs);
socialWopqCont_data = cell(1, nPairs);

for pair = 1:nPairs
    opqCont_entries = {Data{pair}.opqCont};
    % opqCont_entries = {Data{pair}.opaqueControl};
    desired_entries = {'Social W', 'Social W OPAQUE CONTROL'};
    % desired_entries = [0, 1];
    opqCont_entries_char = cellfun(@char, opqCont_entries, 'UniformOutput', false);
    filtered_entries = ismember(opqCont_entries_char, desired_entries{1});
    % filtered_entries = cellfun(@(x) any(x == desired_entries(1)), opqCont_entries);


    % Split the Data{1} struct array based on the logical array
    socialW_data{pair} = Data{pair}(filtered_entries);
    socialWopqCont_data{pair} = Data{pair}(~filtered_entries);
end

%% Get an average by date (socialW)

socialW_perf_by_date = cell(1, nPairs);
socialW_error_perf_by_date = cell(1, nPairs);

figure('Color', [1 1 1]);
hold on;

for pair = 1:nPairs
    tempData = socialW_data{1,pair}(~cellfun('isempty', {socialW_data{1,pair}.perf}));
    dates_numeric = datenum({tempData.date});
    [~, ~, subs] = unique(floor(dates_numeric));
    avg_performance = accumarray(subs, [tempData.perf]', [], @mean);
    std_performance = accumarray(subs, [tempData.perf]', [], @std);
    count = accumarray(subs, ones(size({tempData.perf})), [], @sum);
    standard_error = std_performance ./ sqrt(count);
    socialW_perf_by_date{pair}= avg_performance;
    socialW_error_perf_by_date{pair} = standard_error;
    plot(1:numel(socialW_perf_by_date{1,pair}), socialW_perf_by_date{1,pair})
    ylim([0 100])
    xlabel('Day#')
    ylabel('% Perofrmance (theoretical max)')
end

% figure('Color', [1 1 1]);
% shadedErrorBar(1:numel(socialW_perf_by_date{1,1}), socialW_perf_by_date{1,1}, socialW_error_perf_by_date{1,1});
% hold on
% plot([])
% shadedErrorBar(1:numel(socialW_perf_by_date{1,2}), socialW_perf_by_date{1,2}, socialW_error_perf_by_date{1,2}, 'Lineprops', {'b'});
% shadedErrorBar(1:numel(socialW_perf_by_date{1,3}), socialW_perf_by_date{1,3}, socialW_error_perf_by_date{1,3}, 'Lineprops', {'r'});
% xlabel('Run#')
% ylabel('% Perofrmance (theoretical max)')

%% Get an average by date (socialW opaque control)

socialWopqCont_perf_by_date = cell(1, nPairs);
socialWopqCont_error_perf_by_date = cell(1,nPairs);

figure('Color', [1 1 1]);
hold on

for pair = 1:nPairs
    tempData = socialWopqCont_data{1,pair}(~cellfun('isempty', {socialWopqCont_data{1,pair}.perf}));
    dates_numeric = datenum({tempData.date});
    [~, ~, subs] = unique(floor(dates_numeric));
    avg_performance = accumarray(subs, [tempData.perf]', [], @mean);
    std_performance = accumarray(subs, [tempData.perf]', [], @std);
    count = accumarray(subs, ones(size({tempData.perf})), [], @sum);
    standard_error = std_performance ./ sqrt(count);
    socialWopqCont_perf_by_date{pair}= avg_performance;
    socialWopqCont_error_perf_by_date{pair} = standard_error;
    plot(1:numel(socialWopqCont_perf_by_date{1,pair}), socialWopqCont_perf_by_date{1,pair})
    ylim([0 100])
    xlabel('Day#')
    ylabel('% Perofrmance (theoretical max)')
end

% Plot daily mean performance with shaded error bars
% figure('Color', [1 1 1]);
% shadedErrorBar(1:numel(socialWopqCont_perf_by_date{1,1}), socialWopqCont_perf_by_date{1,1}, socialWopqCont_error_perf_by_date{1,1});
% hold on
% plot([])
% shadedErrorBar(1:numel(socialWopqCont_perf_by_date{1,2}), socialWopqCont_perf_by_date{1,2}, socialWopqCont_error_perf_by_date{1,2}, 'Lineprops', {'b'});
% shadedErrorBar(1:numel(socialWopqCont_perf_by_date{1,3}), socialWopqCont_perf_by_date{1,3}, socialWopqCont_error_perf_by_date{1,3}, 'Lineprops', {'r'});
% xlabel('Day#')
% ylabel('% Perofrmance (theoretical max)')

%% Plot socialW and socialWopqCont performance on the same plot

pair_colors = lines(numel(socialW_perf_by_date));

figure('Color', [1 1 1]);
hold on

% Plot data from cellArray1
for i = 1:numel(socialW_perf_by_date)
    plot(1:numel(socialW_perf_by_date{i}), socialW_perf_by_date{i}, '-', 'Color', pair_colors(i, :), 'DisplayName', ['Condition 1 - Plot ' num2str(i)]);
    
end


% Plot data from socialW opaque control runs
for i = 1:numel(socialWopqCont_perf_by_date)
    plot(50:49+numel(socialWopqCont_perf_by_date{i}), socialWopqCont_perf_by_date{i}, '-', 'Color', pair_colors(i, :), 'DisplayName', ['Condition 1 - Plot ' num2str(i)]);
    
end

plot([])
hold off;

xlabel('Day#')
ylabel('% Perofrmance (theoretical max)')

%% Store performance data for all pairs across all cohorts 
% REMEMBER: You have to copy data in the empty cells manually (can be automated)

socialW_perf_WT = {};
socialW_perf_FX = {};
socialW_perf_mixed = {};

socialW_perf_WT(cellfun(@isempty, socialW_perf_WT)) = {NaN};
socialW_perf_WT = cell2mat(socialW_perf_WT);

socialW_perf_FX(cellfun(@isempty, socialW_perf_FX)) = {NaN};
socialW_perf_FX = cell2mat(socialW_perf_FX);

socialW_perf_mixed(cellfun(@isempty, socialW_perf_mixed)) = {NaN};
socialW_perf_mixed = cell2mat(socialW_perf_mixed);

%% Further, segregate data based on reward contingencies of the runs

socialW_100 = cell(1,nPairs);
socialW_50 = cell(1,nPairs);
for pair = 1:nPairs
    opqCont_entries = {socialW_data{pair}.rewardContingency};
    % opqCont_entries = {Data{pair}.opaqueControl};
    desired_entries = [100, 50];
    filtered_entries = cellfun(@(x) any(x == desired_entries(1)), opqCont_entries);
    % Split the Data{1} struct array based on the logical array
    socialW_100{pair} = socialW_data{pair}(filtered_entries);
    socialW_50{pair} = socialW_data{pair}(~filtered_entries);
end

%% Or else if you have a cut-off date for segregating the reward
% contingencies use the following
id = 2;
date_values = datetime({socialW_data{id}.date}, 'InputFormat', 'dd-MMM-yyyy');
cutoff_date = '26-Nov-2023';
cutoff_indices = date_values >= datetime(cutoff_date, 'InputFormat', 'dd-MMM-yyyy');
ashu_100 = socialW_data{id}(~cutoff_indices);
ashu_50 = socialW_data{id}(cutoff_indices);
socialW_100{id} = ashu_100;
socialW_50{id} = ashu_50;

%%

default_value = [NaN,NaN];
WT_100 = cellfun(@(x) replaceNaN(x, default_value), WT_100, 'UniformOutput', false);

%%

% Function to replace NaN with a default value
function result = replaceNaN(x, default_value)
    if any(isnan(x))
        result = default_value;
    else
        result = x;
    end
end