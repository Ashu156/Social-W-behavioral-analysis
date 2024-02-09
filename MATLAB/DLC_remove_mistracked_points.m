%%
tic;

clear;
close all;
clc;

%% Load data and preprocess

mydir = uigetdir();
cd(mydir);
folder = dir(fullfile(mydir, '*.csv*'));

rat1Files = folder(contains({folder.name},'Rat1-AllTracking'),:);
rat2Files = folder(contains({folder.name},'Rat2-AllTracking'),:);

run_number = 2;
%%for run_number = 1:numel(rat1Files)
% 

rat1 = readtable(strcat(mydir, '\', rat1Files(run_number).name));
rat2 = readtable(strcat(mydir, '\', rat2Files(run_number).name));

session = rat1Files(run_number).name;

%% For subsequent runs

close all
clearvars -except mydir folder rat1Files rat2Files run_number ratinfo

run_number = run_number + 1;

rat1 = readtable(strcat(mydir, '\', rat1Files(run_number).name));
rat2 = readtable(strcat(mydir, '\', rat2Files(run_number).name));

session = rat1Files(run_number).name;

%% Visualize animal trajectory color coded wrt time

figure('Color', [1 1 1]);
x1 = rat1.head_1;
y1 = rat1.head_2;
t1 = (1:height(rat1))';
col1 = t1;

scatter3(x1(1:1:end), y1(1:1:end), t1(1:1:end), 30, col1(1:1:end), 'filled');
colormap('parula');  % You can change the colormap as needed
colorbar;
clim([min(col1) max(col1)])
view(90, 90)
xlabel('X-Coord')
ylabel('Y-Coord')
zlabel('Frame #')
title('Rat1 original trajectory')


figure('Color', [1 1 1]);
x2 = rat2.head_1;
y2 = rat2.head_2;
t2 = (1:height(rat2))';
col2 = t2;

scatter3(x2(1:1:end), y2(1:1:end), t2(1:1:end), 30, col2(1:1:end), 'filled');
colormap('parula');  % You can change the colormap as needed
colorbar;
clim([min(col2) max(col2)])
view(90, 90)
xlabel('X-Coord')
ylabel('Y-Coord')
zlabel('Frame #')
title('Rat2 original trajectroy')

%%

% Assuming you have tables named 'rat1' and 'rat2'
% Replace 'rat1' and 'rat2' with the actual names of your tables

tablesToProcess = {rat1, rat2};  % Add more tables as needed

columnsToRename = {'head_center_1', 'head_center_2', 'body_center_1', 'body_center_2'};

for t = 1:length(tablesToProcess)
    currentTable = tablesToProcess{t};
    
    % Create a new table with the same data
    newTable = array2table(currentTable{:,:}, 'VariableNames', currentTable.Properties.VariableNames);

    for i = 1:length(columnsToRename)
        currentColumnName = columnsToRename{i};

        % Check if the current column exists in the table
        if ismember(currentColumnName, newTable.Properties.VariableNames)
            % Replace '_center_' with '_'
            newColumnName = strrep(currentColumnName, '_center_', '_');
            % Replace 'body_' with 'bodyCenter_' and 'head_' with 'head_'
            newColumnName = strrep(newColumnName, 'body_', 'bodyCenter_');
            newColumnName = strrep(newColumnName, 'head_', 'head_');

            % Rename the column in the new table
            newTable.Properties.VariableNames{currentColumnName} = newColumnName;
        else
            disp(['Table ', num2str(t), ', Column ', currentColumnName, ' not found.']);
        end
    end
    
    % Display the renaming information
    disp(['Table ', num2str(t), ' columns renamed.']);

    % Replace the original table with the modified table
    tablesToProcess{t} = newTable;
end

rat1 = tablesToProcess{1};
rat2 = tablesToProcess{2};




%%

close all; 

% Rat 1
figure('Color', [1 1 1]);
x1 = rat1.head_1;
y1 = rat1.head_2;
t1 = (1:height(rat1))';
col1 = t1;

scatter3(x1(1:1:end), y1(1:1:end), t1(1:1:end), 30, col1(1:1:end), 'filled');
colormap('parula');  % You can change the colormap as needed
colorbar;
clim([min(col1) max(col1)])
view(90, 90)
xlabel('X-Coord')
ylabel('Y-Coord')
zlabel('Frame #')
title('Rat1 original trajectory')

% Use impoly for polygon selection
disp('Draw polygons around the points to be selected. Double-click to finish each polygon.');

% Initialize a cell array to store selected indices for each polygon
allSelectedIndices = cell(0, 1);

while true
    % Create a new polygon
    h = impoly(gca);
    
    % Get the coordinates of the selected polygon
    lassoPoints = getPosition(h);
    
    % Check if the user double-clicked to finish
    if size(lassoPoints, 1) < 3
        % If only two points are selected, break the loop
        break;
    end
    
    % Assuming your original data is in x1 and y1 for Rat 1
    selectedIndices = inpolygon(x1, y1, lassoPoints(:,1), lassoPoints(:,2));
    
    % Set the selected points to NaN for Rat 1 except for the frame column
    rat1{selectedIndices, setdiff(rat1.Properties.VariableNames, 'frame')} = NaN;
    
    % Store the selected indices for this polygon
    allSelectedIndices{end+1} = find(selectedIndices);
    
    % Plot the updated trajectory for Rat 1
    figure('Color', [1 1 1]);
    scatter3(rat1.head_1(1:1:end), rat1.head_2(1:1:end), t1(1:1:end), 30, col1(1:1:end), 'filled');
    colormap('parula');  % You can change the colormap as needed
    colorbar;
    clim([min(col1) max(col1)])
    view(90, 90)
    xlabel('X-Coord')
    ylabel('Y-Coord')
    zlabel('Frame #')
    title('Rat1 corrected trajectory')
end

% Display selected points for each polygon
disp('Selected Points for Each Polygon:');
disp(allSelectedIndices);

close all;


% Rat 2
figure('Color', [1 1 1]);
x2 = rat2.head_1;
y2 = rat2.head_2;
t1 = (1:height(rat2))';
col2 = t2;

scatter3(x2(1:1:end), y2(1:1:end), t2(1:1:end), 30, col2(1:1:end), 'filled');
colormap('parula');  % You can change the colormap as needed
colorbar;
clim([min(col1) max(col1)])
view(90, 90)
xlabel('X-Coord')
ylabel('Y-Coord')
zlabel('Frame #')
title('Rat2 original trajectory')

% Use impoly for polygon selection
disp('Draw polygons around the points to be selected. Double-click to finish each polygon.');

% Initialize a cell array to store selected indices for each polygon
allSelectedIndices = cell(0, 1);

while true
    % Create a new polygon
    h = impoly(gca);
    
    % Get the coordinates of the selected polygon
    lassoPoints = getPosition(h);
    
    % Check if the user double-clicked to finish
    if size(lassoPoints, 1) < 3
        % If only two points are selected, break the loop
        break;
    end
    
    % Assuming your original data is in x1 and y1 for Rat 1
    selectedIndices = inpolygon(x2, y2, lassoPoints(:,1), lassoPoints(:,2));
    
    % Set the selected points to NaN for Rat 1
    rat2{selectedIndices, setdiff(rat2.Properties.VariableNames, 'frame')} = NaN;
    
    % Store the selected indices for this polygon
    allSelectedIndices{end+1} = find(selectedIndices);
    
    % Plot the updated trajectory for Rat 1
    figure('Color', [1 1 1]);
    scatter3(rat2.head_1(1:1:end), rat2.head_2(1:1:end), t2(1:1:end), 30, col1(1:1:end), 'filled');
    colormap('parula');  % You can change the colormap as needed
    colorbar;
    clim([min(col1) max(col1)])
    view(90, 90)
    xlabel('X-Coord')
    ylabel('Y-Coord')
    zlabel('Frame #')
    title('Rat2 corrected trajectory')
end

% Display selected points for each polygon
disp('Selected Points for Each Polygon:');
disp(allSelectedIndices);
% head1_2head1_2
close all;

%% Calculate distances between body parts

snout_headDist1=[];
snout_headDist2=[];

snout_bodyCenterDist1=[];
snout_bodyCenterDist2=[];



for i = 1:height(rat1)
    snout_headDist1(end+1)=pdist([rat1.snout_1(i) rat1.snout_2(i); rat1.head_1(i) rat1.head_2(i)],'euclidean');
    snout_headDist2(end+1)=pdist([rat2.snout_1(i) rat2.snout_2(i); rat2.head_1(i) rat2.head_2(i)],'euclidean');

    snout_bodyCenterDist1(end+1)=pdist([rat1.snout_1(i) rat1.snout_2(i); rat1.bodyCenter_1(i) rat1.bodyCenter_2(i)],'euclidean');
    snout_bodyCenterDist2(end+1)=pdist([rat2.snout_1(i) rat2.snout_2(i); rat2.bodyCenter_1(i) rat2.bodyCenter_2(i)],'euclidean');


end


% Plotting for rat 1
figure('Color', [1 1 1])
subplot(2,3,1)
plot(rat1.head_1, rat1.head_2, 'ko-', 'MarkerSize', 0.5);
ylabel('Rat 1');
title('Snout x,y');

% Plotting for rat 1 - Snout-Head Dist.
subplot(2,3,2)
plot(snout_headDist1, 'ko-', 'MarkerSize', 0.5);
ylim([0 200]);
title('Snout-Head Dist.');

% Manually choose a y-value by clicking on the plot
[thresholdX, thresholdY] = ginput(1);

% Draw a horizontal line at the chosen y-value
line([1, length(snout_headDist1)], [thresholdY, thresholdY], 'Color', 'r');

% Find indices above the threshold
selectedIndices1 = find(snout_headDist1 > thresholdY);


subplot(2,3,3)
plot(snout_bodyCenterDist1,'ko-','MarkerSize',0.5);
ylim([0 200]);
title('Snout-Body Dist.');

% Manually choose a y-value by clicking on the plot
[thresholdX, thresholdY] = ginput(1);

% Draw a horizontal line at the chosen y-value
line([1, length(snout_bodyCenterDist1)], [thresholdY, thresholdY], 'Color', 'r');

% Find indices above the threshold
selectedIndices2 = find(snout_bodyCenterDist1 > thresholdY);




% Plotting for rat 2
subplot(2,3,4)
plot(rat2.head_1,rat2.head_2,'ro-','MarkerSize',0.5);
ylabel('Rat 2');

subplot(2,3,5)
plot(snout_headDist2,'ro-','MarkerSize',0.5);
ylim([0 200]);
xlabel('Frame No.','HorizontalAlignment','right');

% Manually choose a y-value by clicking on the plot
[thresholdX, thresholdY] = ginput(1);

% Draw a horizontal line at the chosen y-value
line([1, length(snout_headDist2)], [thresholdY, thresholdY], 'Color','k');

% Find indices above the threshold
selectedIndices3 = find(snout_headDist2 > thresholdY);

subplot(2,3,6)
plot(snout_bodyCenterDist2,'ro-','MarkerSize',0.5);
ylim([0 200]);
xlabel('Frame No.','HorizontalAlignment','right');
sgtitle(session)

% Manually choose a y-value by clicking on the plot
[thresholdX, thresholdY] = ginput(1);

% Draw a horizontal line at the chosen y-value
line([1, length(snout_bodyCenterDist2)], [thresholdY, thresholdY], 'Color', 'k');

% Find indices above the threshold
selectedIndices4 = find(snout_bodyCenterDist2 > thresholdY);



% Set the selected points to NaN and re-calculate the body part distances

rat1{selectedIndices1, setdiff(rat1.Properties.VariableNames, 'frame')} = NaN;
rat1{selectedIndices2, setdiff(rat1.Properties.VariableNames, 'frame')} = NaN;

rat2{selectedIndices3, setdiff(rat1.Properties.VariableNames, 'frame')} = NaN;
rat2{selectedIndices4, setdiff(rat1.Properties.VariableNames, 'frame')} = NaN;



snout_headDist1=[];
snout_headDist2=[];

snout_bodyCenterDist1=[];
snout_bodyCenterDist2=[];



for i = 1:height(rat1)
    snout_headDist1(end+1)=pdist([rat1.snout_1(i) rat1.snout_2(i); rat1.head_1(i) rat1.head_2(i)],'euclidean');
    snout_headDist2(end+1)=pdist([rat2.snout_1(i) rat2.snout_2(i); rat2.head_1(i) rat2.head_2(i)],'euclidean');

    snout_bodyCenterDist1(end+1)=pdist([rat1.snout_1(i) rat1.snout_2(i); rat1.bodyCenter_1(i) rat1.bodyCenter_2(i)],'euclidean');
    snout_bodyCenterDist2(end+1)=pdist([rat2.snout_1(i) rat2.snout_2(i); rat2.bodyCenter_1(i) rat2.bodyCenter_2(i)],'euclidean');


end


% Plot re-calculated body part distances

% Plotting for rat 1
figure('Color', [1 1 1])
subplot(2,3,1)
plot(rat1.head_1, rat1.head_2, 'ko-', 'MarkerSize', 0.5);
ylabel('Rat 1');
title('Snout x,y');

% Plotting for rat 1 - Snout-Head Dist.
subplot(2,3,2)
plot(snout_headDist1, 'ko-', 'MarkerSize', 0.5);
ylim([0 200]);
title('Snout-Head Dist.');


subplot(2,3,3)
plot(snout_bodyCenterDist1,'ko-','MarkerSize',0.5);
ylim([0 200]);
title('Snout-Body Dist.');


% Plotting for rat 2
subplot(2,3,4)
plot(rat2.head_1,rat2.head_2,'ro-','MarkerSize',0.5);
ylabel('Rat 2');

subplot(2,3,5)
plot(snout_headDist2,'ro-','MarkerSize',0.5);
ylim([0 200]);
xlabel('Frame No.','HorizontalAlignment','right');

subplot(2,3,6)
plot(snout_bodyCenterDist2,'ro-','MarkerSize',0.5);
ylim([0 200]);
xlabel('Frame No.','HorizontalAlignment','right');
sgtitle(session)

%%

close all; 

rat1Data = table(rat1(:,["snout_1" "snout_2"]),'VariableNames',"xy");
rat2Data = table(rat2(:,["snout_1" "snout_2"]),'VariableNames',"xy");
rat1Data = splitvars(rat1Data,"xy",'NewVariableNames',["x","y"]);
rat2Data = splitvars(rat2Data,"xy",'NewVariableNames',["x","y"]);
writetable(rat1Data,[session(1:26) '-Rat1_corrected-SnoutTracking.csv']);
writetable(rat2Data,[session(1:26) '-Rat2_corrected-SnoutTracking.csv']);
writetable(rat1,[session(1:26) '-Rat1_corrected-AllTracking.csv']);
writetable(rat2,[session(1:26) '-Rat2_corrected-AllTracking.csv']);

toc;
% end

%%

