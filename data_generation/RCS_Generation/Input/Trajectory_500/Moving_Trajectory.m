data = readtable('Data500.csv');


% Extract data for x, y, and z from the table
x = data{500, 1:1000};      % Extract x coordinates
y = data{500, 1001:2000};   % Extract y coordinates
z = data{500, 2001:3000};   % Extract z coordinate

% Extract data for u, v, and w from the table
u = data{500, 3001:4000};     % Extract u components
v = data{500, 4001:5000};     % Extract v components
w = data{500, 5001:6000};     % Extract w components


% Compute pitch (theta)
pitch = atan2(w, sqrt(u.^2 + v.^2));

% Compute yaw (psi)
yaw = atan2(u,v);

% Compute roll (phi) - assuming roll is zero
roll = zeros(size(yaw));
% Combine Euler angles into a single matrix
euler_angles = [x',y',z',pitch',yaw', roll'];
writematrix(euler_angles, "Missile.csv");