% Lunar Lander Master Integration Loop - Test Bench 
clear; clc;

% Dynamically add the entire repository (and all subfolders) to the MATLAB path
currentFolder = fileparts(mfilename('fullpath'));
repoRoot = fullfile(currentFolder, '..');
addpath(genpath(repoRoot));

% --- 1. Load System Parameters (Apollo 11 Specs) ---
% Pulls the physics limits from the central configuration file
params = get_sim_params(); % Initializes the lander state to match the historical Powered Descent Initiation (PDI)
dt = params.dt; 

% --- 2. Simulation Settings ---
max_steps = params.max_steps;

% THE MASTER SWITCH: 'DEAD_AI', 'MANUAL', or 'RL_AGENT'
CONTROL_MODE = 'DEAD_AI'; 

% If testing the RL agent, load the brain trained in Phase 5
if strcmp(CONTROL_MODE, 'RL_AGENT')
    load('trained_lunar_agent.mat', 'agent'); 
end

% --- 3. Initial State ---
% State Vector: [x, y, dx, dy, theta, dtheta, m_main_fuel, m_rcs_fuel]
% Scenario: Powered Descent Initiation (PDI) from lunar orbit
% Set Initial State: The exact state vector at Powered Descent Initiation (PDI)
x_current = [0; 15000; 1700; 0; pi/2; 0; 8200; 300];
u_prev = [0; 0];

% --- 4. Telemetry Logging Arrays ---
history_time   = zeros(1, max_steps);
history_y      = zeros(1, max_steps);
history_fuel   = zeros(1, max_steps);
history_veto   = zeros(1, max_steps);
history_thrust = zeros(1, max_steps);

% --- 5. THE MASTER CONTROL LOOP ---
disp(['Initiating Simulation in Mode: ', CONTROL_MODE, '...']);

for step = 1:max_steps
    current_time = step * dt;
    
    % A. The Primary AI (Nominal Control)
    % The control mode dictates the baseline behavior before the Safety Sidecar intercepts.
    switch CONTROL_MODE
        case 'DEAD_AI'
            % EXPECTED OUTCOME: Brute-force survival. 
            % Simulates a complete primary flight computer failure. 
            % Establishes the telemetry baseline to prove the Sidecar can 
            % catch a fully loaded lander in freefall at the last possible millisecond.
            u_nominal = [0; 0];
            
        case 'MANUAL'
            % EXPECTED OUTCOME: Slower freefall, sidecar intervention.
            % A basic hardcoded rule (firing 10% thrust). 
            % Since 10% thrust (4,500 N) cannot overcome lunar gravity (~20,200 N),
            % the lander will still fall, proving the Sidecar catches partial hardware failures.
            u_nominal = [params.max_main_thrust * 0.1; 0];
            
        case 'RL_AGENT'
            % EXPECTED OUTCOME: Optimal, smooth landing.
            % The trained neural network attempts to land the ship efficiently. 
            % Success is defined as landing safely without ever triggering 
            % the Sidecar's safety veto (which carries a massive reward penalty).
            
            % 1. Provide the agent with the observation state
            obs = get_ai_observation(x_current, params); 
            
            % 2. Ask the trained neural network for its raw requested action [-1, 1]
            action_cell = getAction(agent, obs);
            raw_action = cell2mat(action_cell); 
            
            % 3. Scale neural net output to physical hardware limits
            u_thrust = (raw_action(1) + 1) / 2 * params.max_main_thrust;
            u_torque = raw_action(2) * params.max_side_torque;
            
            u_nominal = [u_thrust; u_torque];
    end
    
    % B. The Action Governor (Safety Filter)
    % Intercepts the AI's command and evaluates it against reality
    [u_actual, VetoTriggered, h_alt, h_fuel] = safety_sidecar_filter(x_current, u_nominal, params);
    
    % C. The Physics Engine (Environment Step)
    % Calculates continuous state derivatives using the final, filtered action
    dxdt = lunar_lander_dynamics(x_current, u_actual, params);
    
    % Discrete Euler Integration to step physical time forward
    x_next = x_current + dxdt * dt;
    
    % D. Terminal Condition Check (Physics Boundary)
    % The simulation ends if the spacecraft hits the ground
    IsDone = (x_next(2) <= 0);
    % E. Log Data for Post-Flight Telemetry
    history_time(step)   = current_time;
    history_y(step)      = x_current(2);
    history_fuel(step)   = x_current(7);
    history_veto(step)   = VetoTriggered;
    history_thrust(step) = u_actual(1);
    
    % F. Update State for the Next Microsecond
    x_current = x_next;
    u_prev = u_actual;
    
    % Terminal Condition Check
    if IsDone
        % Trim the empty pre-allocated zeros from the logs
        history_time   = history_time(1:step);
        history_y      = history_y(1:step);
        history_fuel   = history_fuel(1:step);
        history_veto   = history_veto(1:step);
        history_thrust = history_thrust(1:step);
        fprintf('Simulation terminated at t = %.2f seconds.\n', current_time);
        break;
    end
end

% --- 6. TELEMETRY VISUALIZATION & LOGGING ---
% Plot data
fig = figure('Name', sprintf('Flight Telemetry: %s', CONTROL_MODE), 'Position', [100, 100, 1000, 800]);

% Plot 1: Altitude over Time
ax1 = subplot(3,1,1); 
plot(history_time, history_y, 'b-', 'LineWidth', 2);
hold on;
yline(1.5, 'r--', 'Safety Buffer (1.5m)');
title(sprintf('Lander Altitude - %s', CONTROL_MODE), 'Interpreter', 'none');
ylabel('Meters');
grid on;

% Plot 2: Engine Thrust & Veto Triggers
ax2 = subplot(3,1,2); % 
plot(history_time, history_thrust * 1e-3, 'k.', 'DisplayName','Commanded Thrust');
hold on;
% Highlight areas where the Sidecar took control
veto_indices = find(history_veto == 1);
if ~isempty(veto_indices)
    plot(history_time(veto_indices), history_thrust(veto_indices) * 1e-3, 'r.','DisplayName', 'Sidecar Override');
end
title(sprintf('Main Engine Thrust - %s (Red dots = Sidecar Override)', CONTROL_MODE), 'Interpreter', 'none');
ylabel('Thrust (kN)');
grid on; legend('location', 'best');

% Plot 3: Fuel Depletion
ax3 = subplot(3,1,3); 
plot(history_time, history_fuel, 'g-', 'LineWidth', 2);
title(sprintf('Fuel Mass Remaining - %s', CONTROL_MODE), 'Interpreter', 'none');
xlabel('Time (Seconds)');
ylabel('Kilograms');
grid on;

linkaxes([ax1, ax2, ax3], 'x');

% --- 7. AUTOMATED FILE SAVING ---
% Get the absolute path of the directory where this script is located
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));

% Define the target directory for our test artifacts inside the repo
log_dir = fullfile(script_dir, 'Flight_Logs');

% Create the directory if it doesn't exist yet
if ~exist(log_dir, 'dir')
    mkdir(log_dir);
end

% Generate a clean filename based on the mode and save it
% Using exportgraphics for a clean, high-res image export
filename = fullfile(log_dir, sprintf('telemetry_%s.png', CONTROL_MODE));
exportgraphics(fig, filename, 'Resolution', 300);

fprintf('Telemetry saved successfully to: %s\\n', filename);