function main_simulation(CONTROL_MODE, agent_mat_file, USE_SIDECAR)
% MAIN_SIMULATION Lunar Lander Master Integration Loop - Test Bench 
%
% Inputs:
%   CONTROL_MODE   - String: 'UNPOWERED_ORBIT', 'HARDCODED_PILOT', or 'RL_AGENT' (Default: 'HARDCODED_PILOT')
%   agent_mat_file - String: Path to the .mat file containing the trained agent (Default: 'trained_lunar_agent.mat')
%   USE_SIDECAR    - Boolean: Enable or disable Safety Sidecar override (Default: true)

    if nargin < 1
        CONTROL_MODE = 'HARDCODED_PILOT';
    end
    if nargin < 2
        agent_mat_file = 'trained_lunar_agent.mat';
    end
    if nargin < 3
        USE_SIDECAR = true;
    end

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
    
    % If testing the RL agent, load the specified brain
    if strcmp(CONTROL_MODE, 'RL_AGENT')
        fprintf('Loading Agent from: %s\n', agent_mat_file);
        load(agent_mat_file, 'agent'); 
    end

% --- 3. Initial State ---
% State Vector: [x, y, dx, dy, theta, dtheta, m_main_fuel, m_rcs_fuel]
% Scenario: Powered Descent Initiation (PDI) from lunar orbit
% Set Initial State: The exact state vector at Powered Descent Initiation (PDI)
x_current = [0; 15000; 1700; 0; pi/2; 0; 8200; 300];
u_prev = [0; 0];

% --- 4. Telemetry Logging Arrays ---
history_time   = zeros(1, max_steps);
history_x      = zeros(1, max_steps);
history_y      = zeros(1, max_steps);
history_dy     = zeros(1, max_steps);
history_theta  = zeros(1, max_steps);
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
        case 'UNPOWERED_ORBIT'
            % EXPECTED OUTCOME: Spacecraft remains in a super-orbital trajectory. 
            % Proves the fidelity of the physics engine (centrifugal force cancels gravity).
            u_nominal = [0; 0];
            
        case 'HARDCODED_PILOT'
            % EXPECTED OUTCOME: Successful landing, but highly inefficient fuel usage.
            % A fully functional PD-controller autopilot. It perfectly calculates thrust vectors 
            % but relies on rigid math rather than Neural Network optimization.
            
            m_total = params.dry_mass + x_current(7) + x_current(8);
            
            % 1. Target Velocities
            target_dx = 0; % Always aim to cancel horizontal velocity
            
            % Target descent rate based on altitude: v = -sqrt(2 * a * s)
            % Fall moderately fast at high altitudes, slow down to -1 m/s near the ground
            target_dy = -max(sqrt(2 * 0.3 * max(x_current(2), 0.1)), 1.0); 
            
            % 2. Velocity Errors
            error_dx = target_dx - x_current(3);
            error_dy = target_dy - x_current(4);
            
            % 3. Desired Thrust Vector
            % Scale gains based on mass (F = m*a) to achieve ~0.5 to 1.0 m/s^2 correction per 1 m/s error
            Kp_x = m_total * 0.8; 
            Kp_y = m_total * 1.5; 
            desired_thrust_x = error_dx * Kp_x;
            desired_thrust_y = (error_dy * Kp_y) + (params.gravity * m_total); % Feed-forward gravity
            
            % 4. Target Attitude and Thrust Magnitude
            T_mag = sqrt(desired_thrust_x^2 + desired_thrust_y^2);
            theta_target = atan2(-desired_thrust_x, desired_thrust_y);
            
            % 5. Attitude Controller (PD)
            error_theta = theta_target - x_current(5);
            % Wrap angle to [-pi, pi]
            error_theta = atan2(sin(error_theta), cos(error_theta)); 
            
            % Massively increase damping to prevent phase-space oscillations
            Kp_theta = 50000;
            Kd_theta = 100000;
            u_torque = (error_theta * Kp_theta) - (x_current(6) * Kd_theta);
            u_torque = max(min(u_torque, params.max_side_torque), -params.max_side_torque);
            
            % 6. Engine Controller
            % Only fire the main engine if we are pointed within 15 degrees (~0.25 rad) of the target
            if abs(error_theta) < 0.25
                u_thrust = max(min(T_mag, params.max_main_thrust), 0);
            else
                u_thrust = 0; % Wait until rotation completes
            end
            
            u_nominal = [u_thrust; u_torque];
            
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
    % Intercepts the AI's command and evaluates it against reality if enabled
    if USE_SIDECAR
        [u_actual, VetoTriggered, h_alt, h_fuel] = safety_sidecar_filter(x_current, u_nominal, params);
    else
        u_actual = u_nominal;
        VetoTriggered = false;
    end
    
    % C. The Physics Engine (Environment Step)
    % Calculates continuous state derivatives using the final, filtered action
    dxdt = lunar_lander_dynamics(x_current, u_actual, params);
    
    % Discrete Euler Integration to step physical time forward
    x_next = x_current + dxdt * dt;
    % Wrap angle theta to [-pi, pi] so it never accumulates indefinitely
    x_next(5) = atan2(sin(x_next(5)), cos(x_next(5)));
    
    % D. Terminal Condition Check (Physics Boundary)
    % The simulation ends if the spacecraft hits the ground
    IsDone = (x_next(2) <= 0);
    % E. Log Data for Post-Flight Telemetry
    history_time(step)   = current_time;
    history_x(step)      = x_current(1);
    history_y(step)      = x_current(2);
    history_dy(step)     = x_current(4);
    history_theta(step)  = x_current(5);
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
        history_x      = history_x(1:step);
        history_y      = history_y(1:step);
        history_dy     = history_dy(1:step);
        history_theta  = history_theta(1:step);
        history_fuel   = history_fuel(1:step);
        history_veto   = history_veto(1:step);
        history_thrust = history_thrust(1:step);
        fprintf('Simulation terminated at t = %.2f seconds.\n', current_time);
        break;
    end
end

% --- 6. TELEMETRY VISUALIZATION & LOGGING ---
% Plot data
if USE_SIDECAR
    mode_label = sprintf('%s (With Sidecar)', CONTROL_MODE);
    file_label = sprintf('telemetry_%s_With_Sidecar.png', CONTROL_MODE);
else
    mode_label = sprintf('%s (Without Sidecar)', CONTROL_MODE);
    file_label = sprintf('telemetry_%s_Without_Sidecar.png', CONTROL_MODE);
end

fig = figure('Name', sprintf('Flight Telemetry: %s', mode_label), 'Position', [100, 100, 1000, 800]);

% Plot 1: Altitude over Time
ax1 = subplot(3,1,1); 
plot(history_time, history_y, 'b-', 'LineWidth', 2);
hold on;
yline(1.5, 'r--', 'Safety Buffer (1.5m)');
title(sprintf('Lander Altitude - %s', mode_label), 'Interpreter', 'none');
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
title(sprintf('Main Engine Thrust - %s (Red dots = Sidecar Override)', mode_label), 'Interpreter', 'none');
ylabel('Thrust (kN)');
grid on; legend('location', 'best');

% Plot 3: Fuel Depletion
ax3 = subplot(3,1,3); 
plot(history_time, history_fuel, 'g-', 'LineWidth', 2);
title(sprintf('Fuel Mass Remaining - %s', mode_label), 'Interpreter', 'none');
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
filename = fullfile(log_dir, file_label);
exportgraphics(fig, filename, 'Resolution', 300);

fprintf('Telemetry saved successfully to: %s\\n', filename);

% --- 8. VIDEO VISUALIZATION ---
disp('Launching Advanced Visualizer...');
animate_lunar_lander(history_time, history_x, history_y, history_dy, history_theta, history_thrust, history_fuel, history_veto, params);

end
