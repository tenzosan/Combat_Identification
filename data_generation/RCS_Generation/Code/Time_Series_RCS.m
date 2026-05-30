clc;
clear all;

% -------- Generate timeseries RCS value using lookup table ------------ 
% ------------------- Aircraft Targets ----------------------------
% ------    Bomber -------------
% n = number of trajectory
n = 500;

trajectoryTable = readtable('/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/Input/No_RCS_data500.csv');
x = trajectoryTable{1:n,1:1000}; 
y = trajectoryTable{1:n,1001:2000}; 
% Calculate angles phi in degrees 
phi_data = mod(atan2d(y, x), 360);
% Load Look up table 
filename = '/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/RCS_Generation/Input/Look_Up_Table/Bomber.csv';
% Read the CSV file into a table
rcsTable = readtable(filename);
% Extract phi values and RCS values
phi_values = rcsTable{:, 1};
rcs_values = rcsTable{:, 2}; 
[row,col] = size(phi_data);
result_values = zeros(row,col);

for i = 1:row 
    [~, index] = min(abs(phi_values - phi_data(i, :))); 
    result_values(i, :) = rcs_values(index, :);
end

Bomber = result_values;

% ----------------  Civilian_Commercial_Airliner--------------------------
x = trajectoryTable{n+1:2*n,1:1000}; 
y = trajectoryTable{n+1:2*n,1001:2000}; 
% Calculate angles phi in degrees 
phi_data = mod(atan2d(y, x), 360);
% Replace actual file path
filename = '/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/RCS_Generation/Input/Look_Up_Table/Civilian_Commercial_Airliner.csv';
rcsTable = readtable(filename);
% Extract phi values and RCS values
phi_values = rcsTable{:, 1};
rcs_values = rcsTable{:, 2}; 
[row,col] = size(phi_data);
result_values = zeros(row,col);

for i = 1:row 
    [~, index] = min(abs(phi_values - phi_data(i, :))); 
    result_values(i, :) = rcs_values(index, :);
end


Civilian_Commercial_Airliner = result_values;

% -------------  Civilian Helicopter --------------------------
x = trajectoryTable{2*n+1:3*n,1:1000}; 
y = trajectoryTable{2*n+1:3*n,1001:2000}; 
% Calculate angles phi in degrees 
phi_data = mod(atan2d(y, x), 360);
% Replace actual file path
filename = '/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/RCS_Generation/Input/Look_Up_Table/Civilian_Helicopter.csv';
% Read the CSV file into a table
rcsTable = readtable(filename);
% Extract phi values and RCS values
phi_values = rcsTable{:, 1};
rcs_values = rcsTable{:, 2}; 
[row,col] = size(phi_data);
result_values = zeros(row,col);

for i = 1:row % Iterate over rows of phi_data
    [~, index] = min(abs(phi_values - phi_data(i, :))); 
    result_values(i, :) = rcs_values(index, :);
end


Civilian_Helicopter = result_values;



% ----------------  Cruise Missile -----------------
x = trajectoryTable{3*n+1:4*n,1:1000}; 
y = trajectoryTable{3*n+1:4*n,1001:2000}; 
% Calculate angles phi in degrees
phi_data = mod(atan2d(y, x), 360);
% Replace actual file path
filename = '/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/RCS_Generation/Input/Look_Up_Table/Cruise_Missile.csv';
% Read the CSV file into a table
rcsTable = readtable(filename);
% Extract phi values and RCS values
phi_values = rcsTable{:, 1};
rcs_values = rcsTable{:, 2}; 
[row,col] = size(phi_data);
result_values = zeros(row,col);

for i = 1:row % Iterate over rows of phi_data
    [~, index] = min(abs(phi_values - phi_data(i, :))); 
    result_values(i, :) = rcs_values(index, :);
end

Cruise_Missile = result_values;

% ---------------- Military Helicopter -----------------------------
x = trajectoryTable{4*n+1:5*n,1:1000}; 
y = trajectoryTable{4*n+1:5*n,1001:2000}; 
% Calculate angles phi in degrees 
phi_data = mod(atan2d(y, x), 360);
filename = '/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/RCS_Generation/Input/Look_Up_Table/Military_Helicopter.csv';
% Read the CSV file into a table
rcsTable = readtable(filename);
% Extract phi values and RCS values
phi_values = rcsTable{:, 1};
rcs_values = rcsTable{:, 2}; 
[row,col] = size(phi_data);
result_values = zeros(row,col);

for i = 1:row % Iterate over rows of phi_data
    [~, index] = min(abs(phi_values - phi_data(i, :))); 
    result_values(i, :) = rcs_values(index, :);
end

Military_Helicopter = result_values;


%------------------ Multirole Fighter  ------------------

x = trajectoryTable{5*n+1:6*n,1:1000}; 
y = trajectoryTable{5*n+1:6*n,1001:2000}; 
% Calculate angles phi in degrees 
phi_data = mod(atan2d(y, x), 360);
% Replace  actual file path
filename = '/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/RCS_Generation/Input/Look_Up_Table/Multirole_Fighter.csv';
% Read the CSV file into a table
rcsTable = readtable(filename);
% Extract phi values and RCS values
phi_values = rcsTable{:, 1};
rcs_values = rcsTable{:, 2}; 
[row,col] = size(phi_data);
result_values = zeros(row,col);

for i = 1:row 
    [~, index] = min(abs(phi_values - phi_data(i, :))); 
    result_values(i, :) = rcs_values(index, :);
end


Multirole_Fighter= result_values;


% ----------------------- Small Civilian -----------------
x = trajectoryTable{6*n+1:7*n,1:1000}; 
y = trajectoryTable{6*n+1:7*n,1001:2000}; 
% Calculate angles phi in degrees 
phi_data = mod(atan2d(y, x), 360);
% Replace actual file path
filename = '/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/RCS_Generation/Input/Look_Up_Table/Small_Civilian.csv';
% Read the CSV file into a table
rcsTable = readtable(filename);
% Extract phi values and RCS values
phi_values = rcsTable{:, 1};
rcs_values = rcsTable{:, 2}; 
[row,col] = size(phi_data);
result_values = zeros(row,col);

for i = 1:row 
    [~, index] = min(abs(phi_values - phi_data(i, :))); 
    result_values(i, :) = rcs_values(index, :);
end


Small_Civilian = result_values;

% ------------------ Small Unmanned ----------------
x = trajectoryTable{7*n+1:8*n,1:1000}; 
y = trajectoryTable{7*n+1:8*n,1001:2000}; 
% Calculate angles phi in degrees
phi_data = mod(atan2d(y, x), 360);
% Replace actual file path
filename = '/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/RCS_Generation/Input/Look_Up_Table/Small_Unmanned.csv';
% Read the CSV file into a table
rcsTable = readtable(filename);
% Extract phi values and RCS values
phi_values = rcsTable{:, 1};
rcs_values = rcsTable{:, 2}; 
[row,col] = size(phi_data);
result_values = zeros(row,col);

for i = 1:row 
    [~, index] = min(abs(phi_values - phi_data(i, :))); 
    result_values(i, :) = rcs_values(index, :);
end


Small_Unmanned = result_values;



% ------------------------ Append ----------------------
data1 = [Bomber;Civilian_Commercial_Airliner;Civilian_Helicopter;Cruise_Missile;Military_Helicopter;Multirole_Fighter;Small_Civilian;Small_Unmanned];
data1 = array2table(data1);

% Read the second table without headers
data2 = readtable('/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/Input/No_RCS_data500.csv', 'ReadVariableNames', false);

% Remove first 1000 columns from data2
data2(:, 1:2000) = [];

% Concatenate the tables column-wise
combined = [data1, data2];

% Input Raw data (Kinematics +RCS) for Machine Learning Model
writetable(combined, '/home/rasas/Desktop/Combat_Identification/Aerial_Target_Classification/8_Targets/Dempster_Classification/Input/RCS_data500.csv', 'WriteVariableNames', false);

% ---------------------------------- End -------------------------------- %












