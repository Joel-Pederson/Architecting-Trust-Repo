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
    max_steps = params.max_sim_steps;
    
    % If testing the RL agent, load the specified brain
    if strcmp(CONTROL_MODE, 'RL_AGENT')
        fprintf('Loading Agent from: %s\n', agent_mat_file);
        load(agent_mat_file, 'agent');

        % Evaluate the greedy policy. SAC and PPO carry stochastic actors, so getAction
        % would otherwise SAMPLE from the policy distribution and this demonstration run
        % would show a deliberately noisy pilot rather than the trained one.
        if isprop(agent, 'UseExplorationPolicy')
            agent.UseExplorationPolicy = false;
        end
    end

% --- 3. Initial State ---
% State Vector: [x, y, dx, dy, theta, dtheta, m_main_fuel, m_rcs_fuel]
% Scenario: High Altitude Terminal Descent (Final Approach)
% Set Initial State: LEM at 2,500m altitude descending at -25 m/s with 20 m/s horizontal drift
x_current = [0; 2500; 20; -25; 0; 0; 8200; 300];
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
            % A fully functional PD-controller autopilot. It perfectly calculates thrust
            % vectors but relies on rigid math rather than Neural Network optimization.
            % Lives in core/scripted_pilot.m so the trade study and the environment
            % sanity checks can use the exact same controller.
            u_nominal = scripted_pilot(x_current, params);

        case 'RL_AGENT'
            % EXPECTED OUTCOME: Optimal, smooth landing.
            % The trained neural network attempts to land the ship efficiently. 
            % Success is defined as landing safely without ever triggering 
            % the Sidecar's safety veto (which carries a massive reward penalty).
            
            % 1. Provide the agent with the observation state
            obs = get_ai_observation(x_current, params);

            % 2. Ask the trained neural network for its raw requested action [-1, 1]
            %    The cell wrapper is required: getAction expects one cell per observation
            %    channel, and this call site was passing a bare array.
            action_cell = getAction(agent, {obs});
            raw_action = cell2mat(action_cell);

            % 3. Scale neural net output to physical hardware limits.
            %    THROUGH THE SHARED MAP. This branch previously carried its own copy of
            %    the retired raw-throttle map, u = (a+1)/2 * T_max, while the training
            %    environment had moved to gravity-compensated thrust. An agent evaluated
            %    here was therefore flying a different plant than it trained on: at full
            %    tanks, action 0 commanded 22.5 kN against the environment's 20.3 kN
            %    hover. Every telemetry_RL_AGENT*.png in Flight_Logs predates this fix.
            u_nominal = action_to_command(raw_action, x_current, params);
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

    % Clamp tanks at empty. Euler stepping can otherwise drive fuel mass slightly
    % negative, which quietly corrupts every mass and inertia term downstream of it.
    x_next(7) = max(0, x_next(7));
    x_next(8) = max(0, x_next(8));
    
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
