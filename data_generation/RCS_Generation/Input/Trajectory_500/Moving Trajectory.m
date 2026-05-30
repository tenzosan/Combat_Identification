data = readtable('Data500.csv');

% Extract data for u, v, and w from the table
u = data(2140, 3001:4000);     % Extract u components
v = data(2140, 4001:5000);     % Extract v components
w = data(2140, 5001:60000;     % Extract w components

% Transpose to make them column vectors
u = u';
v = v';
w = w';

% Compute yaw (psi)
yaw = atan2(v, u);

% Compute pitch (theta)
pitch = atan2(-w, sqrt(u.^2 + v.^2));

% Compute roll (phi) - assuming roll is zero
roll = zeros(size(yaw));

% Combine Euler angles into a single matrix
euler_angles = [data(2140,1:1000),data(2140,1001:2000),data(2140,2001:3000),roll, pitch, yaw];
writetable(euler_angles, 'Missile.csv');