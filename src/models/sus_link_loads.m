%% SUSPENSION REAR LOADS ANALYSIS (Full Kinematics, Travel, Peak Table + Combined 3g Bump g-g Loop)
clear; clc; close all;

%% INPUTS
hp_front = struct( ...
    'LowerFore',        [-387.00, 	  207.00,	 83.70], ...
    'LowerAft',         [-629.00, 	  211.25,	 89.32], ...
    'UpperFore',        [-409.00,	  314.80,	 225.65], ...
    'UpperAft',         [-614.60,	  290.30,	 218.20], ...
    'LowerOutboard',    [-496.55,	  579.09,	 127.71], ...
    'UpperOutboard',    [-511.76,	  561.69,	 271.96], ...
    'ToeInboard',       [-417.80,	  220.90,	 118.00], ...
    'ToeOutboard',      [-415.45,	  570.66,	 168.56], ...
    'PushrodLower',     [-504.68,	  553.33,	 156.93], ... 
    'PushrodUpper',     [-623.10,	  249.39,	 533.96]);  

hp_rear = struct( ...
    'LowerFore',        [-1707.078,   290.000,   104.500], ...
    'LowerAft',         [-1972.500,   229.675,   101.631], ...
    'UpperFore',        [-1767.200,   297.000,   240.157], ...
    'UpperAft',         [-1973.700,   321.431,   234.000], ...
    'LowerOutboard',    [-2010.006,   576.800,   118.948], ...
    'UpperOutboard',    [-1993.573,   552.950,   272.618], ...
    'ToeInboard',       [-2005.100,   324.184,   231.500], ...
    'ToeOutboard',      [-2065.861,   559.200,   268.732], ...
    'PushrodLower',     [-1943.75174881,   514.07314992,   118.94800000], ... 
    'PushrodUpper',     [-1850.000,   310.000,   450.000]);    

contactPatch = [-2015.95, 622.500, 0.000];

% Bellcrank & Damper Hardpoints
hp_rear.BellcrankPivot  = [-1850.000,   250.000,   450.000]; 
hp_rear.BellcrankAxis   = [-1900.000,   250.000,   450.000]; 
hp_rear.SpringBellcrank = [-1830.000,   200.000,   480.000]; 
hp_rear.SpringChassis   = [-1650.000,   200.000,   420.000]; 

links = struct( ...
    'name',    {'upper fore', 'upper aft', 'lower fore', 'lower aft', 'pushrod',      'toe link'}, ...
    'chassis', {'UpperFore',  'UpperAft',  'LowerFore',  'LowerAft',  'PushrodUpper', 'ToeInboard'}, ...
    'upright', {'UpperOutboard','UpperOutboard','LowerOutboard','LowerOutboard','PushrodLower','ToeOutboard'});

% Vehicle inputs
veh.massTotal       = 288.00;  % kg, total vehicle mass w/ driver
veh.massSprung      = 253.00;  % kg, total sprung mass
veh.massUnsprungF   = 16.80;   % kg, total front unsprung mass
veh.massUnsprungR   = 18.20;   % kg, total rear unsprung mass
veh.massSprungF     = 121.44;  % kg, front sprung mass
veh.massSprungR     = 131.56;  % kg, rear sprung mass
veh.wheelbase       = 1530;    % mm
veh.trackFront      = 1245;    % mm
veh.trackRear       = 1245;    % mm
veh.cgHeight        = 300;     % mm
veh.tireRadius      = 203.2;   % mm
veh.rollCenterF     = -25.16;  % mm
veh.rollCenterR     = 25.40;   % mm
veh.rollStiffnessF  = 0.46;    % fraction of roll stiffness at front; should be worst case scenario
veh.tlltd            = 0.46;   % total LLTD
veh.weightFrontFrac = 0.48;    % static weight distribution
veh.dfMaxAccel      = 947.70;  % N, Max Accel Downforce
veh.dfMaxCornering  = 799.15;  % N, Max Cornering Downforce
veh.aeroBalance     = 0.48;    % Assumed 50/50 split
corner = 'RL';                 

% g-g Diagram Inputs
gg.axMax = 1.50;  % g
gg.ayMax = 1.80;  % g
gg.n = 361;
g0 = 9.81;
theta = linspace(0, 360, gg.n);
fx = gg.axMax * sind(theta);
fy = gg.ayMax * cosd(theta);

% numel: Returns total number of elements in an array
nLinks = numel(links);

%% Front Link Geometry -> 6x6 Matrix
e_front = zeros(3, nLinks);
r_front = zeros(3, nLinks);

for i = 1:nLinks
    pIn     = hp_front.(links(i).chassis);
    pOut    = hp_front.(links(i).upright);
    v = pIn(:) - pOut(:);
    e_front(:, i) = v/norm(v);
    r_front(:, i) = pOut(:);
end

A_front = zeros(6, nLinks);
for i = 1:nLinks
    A_front(1:3, i) = e_front(:, i);

    % Cross product solves for moment vector (Torque = r x F)
    A_front(4:6, i) = cross(r_front(:,i), e_front(:,i));
end

%% Rear Link Geometry -> 6x6 Matrix
e_rear = zeros(3, nLinks);
r_rear = zeros(3, nLinks);
for i = 1:nLinks
    pIn     = hp_rear.(links(i).chassis);
    pOut    = hp_rear.(links(i).upright);
    v = pIn(:) - pOut(:);
    e_rear(:, i) = v/norm(v);
    r_rear(:, i) = pOut(:);
end

A_rear = zeros(6, nLinks);
for i = 1:nLinks
    A_rear(1:3, i) = e_rear(:, i);

    % Cross product solves for moment vector (Torque = r x F)
    A_rear(4:6, i) = cross(r_rear(:,i), e_rear(:,i));
end

rcp_rear = contactPatch(:);
rcp_front = [-485.902,    622.5,	0];

%% Tire loads (Nominal Maneuver)
[Fx_t, Fy_t, Fz_t] = computeTireLoads(fx, fy, veh, corner, g0, gg, 0.0);

%% Tire loads (Combined 3g Bump Dynamic Case)
[Fxr_t_bump, Fyr_t_bump, Fzr_t_bump] = computeTireLoads(fx, fy, veh, corner, g0, gg, 3.0);

%% Tire loads (Front Left - Combined 3g Bump Dynamic Case)
[Fxf_t_bump, Fyf_t_bump, Fzf_t_bump] = computeTireLoads(fx, fy, veh, 'FL', g0, gg, 3.0);

function [Fx_t, Fy_t, Fz_t] = computeTireLoads(fx, fy, veh, corner, g0, gg, bump_g)
    switch upper(corner)
        case 'FL'
            signX = -1; signY = -1; 
            m_us = veh.massUnsprungF / 2; 
            m_s_axle = veh.massSprungF;   
            track = veh.trackFront; RC = veh.rollCenterF; K_roll = veh.rollStiffnessF;
            aeroDist = veh.aeroBalance;
            lltd = veh.tlltd;
            swd = veh.weightFrontFrac;
        case 'FR'
            signX = -1; signY = +1; 
            m_us = veh.massUnsprungF / 2;
            m_s_axle = veh.massSprungF;
            track = veh.trackFront; RC = veh.rollCenterF; K_roll = veh.rollStiffnessF;
            aeroDist = veh.aeroBalance;
            lltd = veh.tlltd;
            swd = veh.weightFrontFrac;
        case 'RL'
            signX = +1; signY = -1; 
            m_us = veh.massUnsprungR / 2;
            m_s_axle = veh.massSprungR;
            track = veh.trackRear; RC = veh.rollCenterR; K_roll = 1 - veh.rollStiffnessF;
            aeroDist = 1 - veh.aeroBalance;
            lltd = 1 - veh.tlltd;
            swd = 1 - veh.weightFrontFrac;
        case 'RR'
            signX = +1; signY = +1; 
            m_us = veh.massUnsprungR / 2;
            m_s_axle = veh.massSprungR;
            track = veh.trackRear; RC = veh.rollCenterR; K_roll = 1 - veh.rollStiffnessF;
            aeroDist = 1 - veh.aeroBalance;
            lltd = 1 - veh.tlltd;
            swd = 1 - veh.weightFrontFrac;
        otherwise
            error('corner must be one of FL, FR, RL, RR');
    end
    Fz_static_mass = (m_s_axle/2 + m_us) * g0;

    % Calculates current downforce based on root-sum-square (tracing along traction circle) between Max Accel DF and Max Cornering DF
    df_current = sqrt((fx ./ gg.axMax .* veh.dfMaxAccel).^2 + (fy ./ gg.ayMax .* veh.dfMaxCornering).^2);
    Fz_aero = (df_current .* aeroDist) ./ 2; 
    
    % Add dynamic 3g bump vertical overload force
    Fz_bump_force = (m_s_axle/2 + m_us) * g0 * bump_g;
    
    Fz_static = Fz_static_mass + Fz_aero + Fz_bump_force;

    % Is this right?
    dFz_long  = signX .* veh.massTotal .* (fx .* g0) .* veh.cgHeight ./ veh.wheelbase ./ 2;   
    
    dFz_lat_us = signY .* (m_us .* 2) .* (fy .* g0) .* veh.tireRadius ./ track;
    dFz_lat_geo = signY .* m_s_axle .* (fy .* g0) .* RC ./ track;
    
    RC_cg = veh.rollCenterF .* veh.weightFrontFrac + veh.rollCenterR .* (1 - veh.weightFrontFrac);    
    
    h_arm = veh.cgHeight - RC_cg;
    M_roll = veh.massSprung .* (fy .* g0) .* h_arm;
    dFz_lat_elastic = signY .* (M_roll .* K_roll) ./ track;
    dFz_lat = (dFz_lat_us + dFz_lat_geo + dFz_lat_elastic) .* 0.5;          

    Fz_t = Fz_static + dFz_long + dFz_lat;   
   
    % Use when given an existing LLTD term
    Fz_static = (veh.massTotal .* g0 .* swd ./ 2) + Fz_aero + (veh.massTotal .* g0 .* bump_g ./ 4);
    dFz_lat = (veh.massTotal .* g0 .* fy ./ track) .* lltd ./ 2;
    Fz_t = Fz_static + dFz_lat + dFz_long;

    Fz_vehicle = (veh.massTotal * g0) + (df_current) + (bump_g * veh.massTotal * g0);
    Fx_t = (Fz_t ./ Fz_vehicle) .* fx .* g0 .* veh.massTotal;                         
    Fy_t = (Fz_t ./ Fz_vehicle) .* fy .* g0 .* veh.massTotal;                         
end


%% SOLVING LINK LOADS (Combined 3g Bump) - Fronts
F_bump_gg_front = zeros(nLinks, gg.n);

% Iterates thru all possible gg combination (360 deg. possible)
for k = 1:gg.n
    % Combined 3g Bump Case
    Ftire_bump = [Fxf_t_bump(k); Fyf_t_bump(k); Fzf_t_bump(k)];
    b_bump = zeros(6,1);
    b_bump(1:3) = -Ftire_bump;
    b_bump(4:6) = -cross(rcp_front, Ftire_bump);
    F_bump_gg_front(:,k) = A_front\b_bump;
end


%% SOLVING LINK LOADS (Nominal & Combined 3g Bump) - Rears
F_rear = zeros(nLinks, gg.n);
F_bump_gg_rear = zeros(nLinks, gg.n);

% Iterates thru all possible gg combination (360 deg. possible)
for k = 1:gg.n
    final_rcp = rcp_rear;
    if k > 0 && k < 180
        final_rcp(3) = final_rcp(3) + veh.tireRadius;
    end

    % Nominal Case
    Ftire = [Fx_t(k); Fy_t(k); Fz_t(k)];
    b = zeros(6,1);
    b(1:3) = -Ftire;
    b(4:6) = -cross(final_rcp, Ftire);
    F_rear(:,k) = A_rear\b;
    
    % Combined 3g Bump Case
    Ftire_bump = [Fxr_t_bump(k); Fyr_t_bump(k); Fzr_t_bump(k)];
    b_bump = zeros(6,1);
    b_bump(1:3) = -Ftire_bump;
    b_bump(4:6) = -cross(final_rcp, Ftire_bump);
    F_bump_gg_rear(:,k) = A_rear\b_bump;
end

%% PEAK LOAD FINDER & MANEUVER CLASSIFICATION (Nominal)
linkNames = {links.name};

% Initializing MAX links and such
T_names = cell(nLinks, 1);          % 6 x 1 matrix
T_maxTension = zeros(nLinks, 1);    % 6 x 1 matrix of only zeros
T_tensionCond = cell(nLinks, 1);
T_maxComp = zeros(nLinks, 1);
T_compCond = cell(nLinks, 1);

for i = 1:nLinks
    [maxT, idxT] = max(F_rear(i,:));
    [maxC, idxC] = min(F_rear(i,:));
    
    T_names{i} = linkNames{i};
    T_maxTension(i) = round(maxT, 1);
    T_tensionCond{i} = sprintf('Ax: %.2fg, Ay: %.2fg (%s)', fx(idxT), fy(idxT), getManeuverDesc(fx(idxT), fy(idxT)));
    
    T_maxComp(i) = round(maxC, 1);
    T_compCond{i} = sprintf('Ax: %.2fg, Ay: %.2fg (%s)', fx(idxC), fy(idxC), getManeuverDesc(fx(idxC), fy(idxC)));
end

peakTable = table(T_names(:), T_maxTension(:), T_tensionCond(:), T_maxComp(:), T_compCond(:), ...
    'VariableNames', {'Link', 'Max_Tension_N', 'Tension_Condition', 'Max_Compression_N', 'Compression_Condition'});

figTable = figure('Color', 'w', 'Name', 'Peak Suspension Link Loads Summary', 'Position', [200, 200, 950, 220]);
uitable(figTable, 'Data', table2cell(peakTable), ...
    'ColumnName', {'Link Name', 'Max Tension (N)', 'Tension Condition (g-s)', 'Max Compression (N)', 'Compression Condition (g-s)'}, ...
    'ColumnWidth', {110, 110, 280, 130, 280}, ...
    'Units', 'Normalized', 'Position', [0, 0, 1, 1]);

function desc = getManeuverDesc(ax, ay)
    if abs(ax) < 0.05 && abs(ay) < 0.05
        desc = 'Static / Straight Line';
    elseif ax > 0.05 && abs(ay) <= 0.05
        desc = 'Max Acceleration';
    elseif ax < -0.05 && abs(ay) <= 0.05
        desc = 'Max Braking';
    elseif abs(ax) <= 0.05 && ay > 0.05
        desc = 'Pure Left Cornering';
    elseif abs(ax) <= 0.05 && ay < -0.05
        desc = 'Pure Right Cornering';
    elseif ax > 0.05 && ay > 0.05
        desc = 'Accel + Left Turn';
    elseif ax > 0.05 && ay < -0.05
        desc = 'Accel + Right Turn';
    elseif ax < -0.05 && ay > 0.05
        desc = 'Brake + Left Turn';
    else
        desc = 'Brake + Right Turn';
    end
end

%% PEAK LOAD FINDER & MANEUVER CLASSIFICATION (3G Bump)
linkNames = {links.name};

% Initializing MAX links and such
T_names = cell(nLinks, 1);          % 6 x 1 matrix
T_maxTension = zeros(nLinks, 1);    % 6 x 1 matrix of only zeros
T_tensionCond = cell(nLinks, 1);
T_maxComp = zeros(nLinks, 1);
T_compCond = cell(nLinks, 1);

for i = 1:nLinks
    [maxT, idxT] = max(F_bump_gg_rear(i,:));
    [maxC, idxC] = min(F_bump_gg_rear(i,:));

    T_names{i} = linkNames{i};
    T_maxTension(i) = round(maxT, 1);
    T_tensionCond{i} = sprintf('Ax: %.2fg, Ay: %.2fg (%s)', fx(idxT), fy(idxT), getManeuverDesc(fx(idxT), fy(idxT)));

    T_maxComp(i) = round(maxC, 1);
    T_compCond{i} = sprintf('Ax: %.2fg, Ay: %.2fg (%s)', fx(idxC), fy(idxC), getManeuverDesc(fx(idxC), fy(idxC)));
end

peakTable = table(T_names(:), T_maxTension(:), T_tensionCond(:), T_maxComp(:), T_compCond(:), ...
    'VariableNames', {'Link', 'Max_Tension_N', 'Tension_Condition', 'Max_Compression_N', 'Compression_Condition'});

figTable = figure('Color', 'w', 'Name', 'Peak Suspension Link Loads Summary - 3g Bump (Rear)', 'Position', [200, 200, 950, 220]);
uitable(figTable, 'Data', table2cell(peakTable), ...
    'ColumnName', {'Link Name', 'Max Tension (N)', 'Tension Condition (g-s)', 'Max Compression (N)', 'Compression Condition (g-s)'}, ...
    'ColumnWidth', {110, 110, 280, 130, 280}, ...
    'Units', 'Normalized', 'Position', [0, 0, 1, 1]);


%% Front - PEAK LOAD FINDER & MANEUVER CLASSIFICATION (3G Bump)
linkNames = {links.name};

% Initializing MAX links and such
T_names = cell(nLinks, 1);          % 6 x 1 matrix
T_maxTension = zeros(nLinks, 1);    % 6 x 1 matrix of only zeros
T_tensionCond = cell(nLinks, 1);
T_maxComp = zeros(nLinks, 1);
T_compCond = cell(nLinks, 1);

for i = 1:nLinks
    [maxT, idxT] = max(F_bump_gg_front(i,:));
    [maxC, idxC] = min(F_bump_gg_front(i,:));

    T_names{i} = linkNames{i};
    T_maxTension(i) = round(maxT, 1);
    T_tensionCond{i} = sprintf('Ax: %.2fg, Ay: %.2fg (%s)', fx(idxT), fy(idxT), getManeuverDesc(fx(idxT), fy(idxT)));

    T_maxComp(i) = round(maxC, 1);
    T_compCond{i} = sprintf('Ax: %.2fg, Ay: %.2fg (%s)', fx(idxC), fy(idxC), getManeuverDesc(fx(idxC), fy(idxC)));
end

peakTable = table(T_names(:), T_maxTension(:), T_tensionCond(:), T_maxComp(:), T_compCond(:), ...
    'VariableNames', {'Link', 'Max_Tension_N', 'Tension_Condition', 'Max_Compression_N', 'Compression_Condition'});

figTable = figure('Color', 'w', 'Name', 'Peak Suspension Link Loads Summary - 3g Bump (Front)', 'Position', [200, 200, 950, 220]);
uitable(figTable, 'Data', table2cell(peakTable), ...
    'ColumnName', {'Link Name', 'Max Tension (N)', 'Tension Condition (g-s)', 'Max Compression (N)', 'Compression Condition (g-s)'}, ...
    'ColumnWidth', {110, 110, 280, 130, 280}, ...
    'Units', 'Normalized', 'Position', [0, 0, 1, 1]);


%% Figure 1 --- g-g diagram
figure('Color', 'w', 'Name', 'g-g diagram');
plot(fy, fx, 'LineWidth', 1.3);
grid on; box on;
xlabel('Fy (g)'); ylabel('Fx (g)');
title('g-g diagram');
axis equal;                           
xlim([-gg.ayMax*1.2, gg.ayMax*1.2]);   
ylim([-gg.axMax*1.2, gg.axMax*1.2]);   

%% Figure 2 -- Nominal suspension link loads vs g-g angle
figure('Color', 'w', 'Name', 'Nominal Suspension Link Loads');
hold on; grid on; box on;
for i=1:nLinks
    plot(theta, F_rear(i,:), 'LineWidth', 1.4);
end
xlabel('g-g angle (deg)');
ylabel('link load, N (+ tension / - compression)');
title('nominal suspension link loads (no bump)');
legend({links.name}, 'Location','best');
xlim([0 360]);

%% Figure 3 -- Combined 3g Bump Link Loads Across Entire g-g Circle
figure('Color', 'w', 'Name', 'Combined 3g Bump Link Loads Across g-g Circle');
hold on; grid on; box on;
for i = 1:nLinks
    plot(theta, F_bump_gg_rear(i,:), 'LineWidth', 1.5);
end
xlabel('g-g angle (degrees)');
ylabel('link load, N (+ tension / - compression)');
title('suspension link loads with combined 3g bump impulse across g-g circle');
legend({links.name}, 'Location', 'eastoutside');
xlim([0 360]);

%% TRAVEL SENSITIVITY (Pushrod & Lower Fore)
travel_in = -1.5 : 0.25 : 1.5;      
travel_mm = travel_in * 25.4;       
num_states = length(travel_in);

idx_LF = find(strcmp({links.name}, 'lower fore'));
idx_PR = find(strcmp({links.name}, 'pushrod'));

outboard_pts = {'LowerOutboard', 'UpperOutboard', 'ToeOutboard', 'PushrodLower'};

F_LF_travel = zeros(num_states, gg.n);
F_PR_travel = zeros(num_states, gg.n);

for j = 1:num_states
    dz = travel_mm(j);
    hp_mod = hp_rear;
    for p = 1:length(outboard_pts)
        hp_mod.(outboard_pts{p})(3) = hp_mod.(outboard_pts{p})(3) + dz;
    end
    rcp_mod = contactPatch(:) + [0; 0; dz];
    
    A_mod = zeros(6, nLinks);
    for i = 1:nLinks
        pIn  = hp_mod.(links(i).chassis);
        pOut = hp_mod.(links(i).upright);
        v = pIn(:) - pOut(:);
        e_mod = v / norm(v);
        r_mod = pOut(:);
        A_mod(1:3, i) = e_mod;
        A_mod(4:6, i) = cross(r_mod, e_mod);
    end
    
    for k = 1:gg.n
        Ftire = [Fx_t(k); Fy_t(k); Fz_t(k)];
        b = zeros(6,1);
        b(1:3) = -Ftire;
        b(4:6) = -cross(rcp_mod, Ftire);
        F_temp = A_mod \ b;
        
        F_LF_travel(j, k) = F_temp(idx_LF);
        F_PR_travel(j, k) = F_temp(idx_PR);
    end
end

colors = zeros(num_states, 3);
for i = 1:num_states
    val = travel_in(i);
    if val < 0
        intensity = 1 - (abs(val) / 1.5); 
        colors(i, :) = [0.2 + 0.3*intensity, 0.4 + 0.3*intensity, 1.0 - 0.2*intensity];
    elseif val == 0
        colors(i, :) = [0, 0, 0]; 
    else
        intensity = val / 1.5; 
        colors(i, :) = [1.0 - 0.2*intensity, 0.6 - 0.6*intensity, 0.2 - 0.2*intensity];
    end
end

leg_entries = cell(num_states, 1);
for j = 1:num_states
    leg_entries{j} = sprintf('%+.2f in. travel', travel_in(j));
end

%% Figure 4 -- Travel Sensitivity: Pushrod
figure('Color', 'w', 'Name', 'Travel Sensitivity: Pushrod');
hold on; grid on; box on;
for j = 1:num_states
    lw = 1.3; if travel_in(j) == 0, lw = 2.2; end 
    plot(theta, F_PR_travel(j,:), 'LineWidth', lw, 'Color', colors(j,:));
end
xlabel('g-g angle (degrees)');
ylabel('pushrod load (N)');
title('pushrod forces across jounce & rebound');
xlim([0 360]);
legend(leg_entries, 'Location', 'best');

%% Figure 5 -- Travel Sensitivity: Lower Fore
figure('Color', 'w', 'Name', 'Travel Sensitivity: Lower Fore');
hold on; grid on; box on;
for j = 1:num_states
    lw = 1.3; if travel_in(j) == 0, lw = 2.2; end 
    plot(theta, F_LF_travel(j,:), 'LineWidth', lw, 'Color', colors(j,:));
end
xlabel('g-g angle (degrees)');
ylabel('lower fore load (N)');
title('lower fore forces across jounce & rebound');
xlim([0 360]);
legend(leg_entries, 'Location', 'best');

%% Console Output: Combined 3g Bump Peaks Across g-g Circle
fprintf('\n--- PEAK LINK LOADS UNDER COMBINED 3G BUMP ACROSS G-G CIRCLE ---\n');
for i = 1:nLinks
    [max_t, idx_t] = max(F_bump_gg_rear(i, :));
    [max_c, idx_c] = min(F_bump_gg_rear(i, :));
    fprintf('%-12s -> Peak Tens: %7.1f N (at %3.0f deg) | Peak Comp: %7.1f N (at %3.0f deg)\n', ...
        links(i).name, max_t, theta(idx_t), max_c, theta(idx_c));
end
fprintf('---------------------------------------------------------------\n');