%% 
clear;
close all; 
clc;

%% load data
load('cohortER1_pairData_segregated_rewardCont.mat')

%% Calculate transition probability matrix and joint entropy of state occupancies

% For 100% reward contingencies


for rt = 1:numel(socialW_100)
    % temp_data = socialW_100{rt};
    % trials = cell(2, length(temp_data));

for i = 1:length(socialW_100{rt}) % iterate across 

    % Skip rows in errorIndices
    if isempty(socialW_100{rt}(i).ratsamples)
        continue;  % Skip this iteration
    end

    
        mysamples = socialW_100{rt}(i).ratsamples;
        % 
        rt1 = mysamples{1}.thiswell;
        rt2 = mysamples{2}.thiswell;
        % Calculate the transition matrix
        [transition_matrix1, states1] = calculate_transition_matrix(rt1);
        [transition_matrix2, states2] = calculate_transition_matrix(rt2);
        socialW_100{rt}(i).TPM = {transition_matrix1 transition_matrix2};

        % Calculate joint entropies of well visits
        table1 = mysamples{1};
        table2 = mysamples{2};
        t1_a1 = [];
        t1_a1 = table1.start;
        t1_a1(:, 2) = table1.end;
        t1_a1(:, 3) = table1.thiswell;
        t1_a1(:, 4) = table1.match;
        t2_a1 = [];
        t2_a1 = table2.start;
        t2_a1(:, 2) = table2.end;
        t2_a1(:, 3) = table2.hiswell;
        t2_a1(:, 4) = table2.match;
        combinedA1 = [t1_a1; t2_a1];
        sorted_A1 = sort(combinedA1);
        t1_a2 = [];
        t1_a2 = table1.start;
        t1_a2(:, 2) = table1.end;
        t1_a2(:, 3) = table1.hiswell;
        t1_a2(:, 4) = table1.match;
        t2_a2 = [];
        t2_a2 = table2.start;
        t2_a2(:, 2) = table2.end;
        t2_a2(:, 3) = table2.thiswell;
        t2_a2(:, 4) = table2.match;

        combinedA2 = [t1_a2; t2_a2];
        sorted_A2 = sort(combinedA2);

        A1A2 = [sorted_A1(:, 3) sorted_A2(:, 3)];
        nan_rows = any(isnan(A1A2), 2);
        valid_rows = ~nan_rows;
        if any(valid_rows)
            JEnt = jointEntropy(A1A2(valid_rows, :), A1A2(valid_rows, :));

            % Assign the calculated joint entropy
            socialW_100{rt}(i).JEnt = JEnt;
        else
            % If there are no valid rows, you might handle it in a specific way or leave the field empty
            socialW_100{rt}(i).JEnt = [];  % or set it to a default value or display a message
        end

        if isempty(socialW_100{rt}(i).perf)
            continue;  % Skip this iteration
        end
        normNtr = socialW_100{rt}(i).nTransitions / socialW_100{rt}(i).duration;
        socialW_100{rt}(i).normNtr = normNtr;

        normMtch = socialW_100{rt}(i).matches / socialW_100{rt}(i).duration;
        socialW_100{rt}(i).normMtch = normMtch;
   
end

end

% mydir = cd; % directory where the stateScript log files are located 
% save('Cohort AS2_ratwiseData_23-Jan-2024 17_44_54.mat', 'socialW_100')

%% Calculate transition probability matrix and joint entropy of state occupancies

% For 50% reward contingencies

for rt = 1:numel(socialW_50)

for i = 1:length(socialW_50{rt}) % iterate across 

    % Skip rows in errorIndices
    if isempty(socialW_50{rt}(i).ratsamples)
        continue;  % Skip this iteration
    end

    
        mysamples = socialW_50{rt}(i).ratsamples;
        % 
        rt1 = mysamples{1}.thiswell;
        rt2 = mysamples{2}.thiswell;
        % Calculate the transition matrix
        [transition_matrix1, states1] = calculate_transition_matrix(rt1);
        [transition_matrix2, states2] = calculate_transition_matrix(rt2);
        socialW_50{rt}(i).TPM = {transition_matrix1 transition_matrix2};

        % Calculate joint entropies of well visits
        table1 = mysamples{1};
        table2 = mysamples{2};
        t1_a1 = [];
        t1_a1 = table1.start;
        t1_a1(:, 2) = table1.end;
        t1_a1(:, 3) = table1.thiswell;
        t2_a1 = [];
        t2_a1 = table2.start;
        t2_a1(:, 2) = table2.end;
        t2_a1(:, 3) = table2.hiswell;
        combinedA1 = [t1_a1; t2_a1];
        sorted_A1 = sort(combinedA1);
        t1_a2 = [];
        t1_a2 = table1.start;
        t1_a2(:, 2) = table1.end;
        t1_a2(:, 3) = table1.hiswell;
        t2_a2 = [];
        t2_a2 = table2.start;
        t2_a2(:, 2) = table2.end;
        t2_a2(:, 3) = table2.thiswell;

        combinedA2 = [t1_a2; t2_a2];
        sorted_A2 = sort(combinedA2);

        A1A2 = [sorted_A1(:, 3) sorted_A2(:, 3)];
        nan_rows = any(isnan(A1A2), 2);
        valid_rows = ~nan_rows;
        if any(valid_rows)
            JEnt = jointEntropy(A1A2(valid_rows, :), A1A2(valid_rows, :));

            % Assign the calculated joint entropy
            socialW_50{rt}(i).JEnt = JEnt;
        else
            % If there are no valid rows, you might handle it in a specific way or leave the field empty
            socialW_50{rt}(i).JEnt = [];  % or set it to a default value or display a message
        end

        if isempty(socialW_50{rt}(i).perf)
            continue;  % Skip this iteration
        end
        normNtr = socialW_50{rt}(i).nTransitions / socialW_50{rt}(i).duration;
        socialW_50{rt}(i).normNtr = normNtr;

        normMtch = socialW_50{rt}(i).matches / socialW_50{rt}(i).duration;
        socialW_50{rt}(i).normMtch = normMtch;
        
   
end

end

% mydir = cd; % directory where the stateScript log files are located 
% save('Cohort AS2_ratwiseData_23-Jan-2024 17_44_54.mat', 'socialW_100')

%% Function to calculate transition probability matrix

function [transition_matrix, states] = calculate_transition_matrix(data)
    unique_states = unique(data);
    num_states = numel(unique_states);

    % Initialize transition matrix as a 3-by-3 zeros matrix
    transition_matrix = zeros(3);

    % Initialize transition matrix with zeros
    transition_matrix = zeros(num_states);

    % Count transitions excluding self-transitions
    for i = 1:(numel(data) - 1)
        current_state = data(i);
        next_state = data(i + 1);

        if current_state ~= next_state
            current_index = find(unique_states == current_state);
            next_index = find(unique_states == next_state);
            transition_matrix(current_index, next_index) = transition_matrix(current_index, next_index) + 1;
        end
    end

    % Normalize each row to get transition probabilities
    row_sums = sum(transition_matrix, 2);
    transition_matrix = transition_matrix ./ repmat(row_sums, 1, num_states);

    states = unique_states;
end

