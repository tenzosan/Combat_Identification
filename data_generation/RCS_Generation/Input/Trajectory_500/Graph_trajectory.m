data = readtable('Data500.csv');

% Extract data for x, y, and z from the table
x_1 = data(225,1:1000};     
y_1 = data{225,1001:2000};  
z_1 = data{225,2001:3000};   

% Plot the 3D data
plot3(x_1, y_1, z_1, '-r', 'LineWidth', 1.5);

% Set labels for x, y, and z axes with LaTeX formatting
xlabel('$x$ [m]', 'Interpreter', 'latex', 'FontSize', 30);
ylabel('$y$ [m]', 'Interpreter', 'latex', 'FontSize', 30);
zlabel('$z$ [m]', 'Interpreter', 'latex', 'FontSize', 30);
% Get handle to current axes
grid on;
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 18;
ax.XAxis.FontWeight = 'bold';  % Increase thickness of X-axis label
ax.YAxis.FontWeight = 'bold';  % Increase thickness of Y-axis label
ax.ZAxis.FontWeight = 'bold';
ax.XAxis.LineWidth = 2;  % Adjust the thickness for X-axis ticks
ax.YAxis.LineWidth = 2;  % Adjust the thickness for Y-axis ticks
ax.ZAxis.LineWidth = 2;
title('Military Helicopter', 'Interpreter', 'latex', 'FontSize', 20);
