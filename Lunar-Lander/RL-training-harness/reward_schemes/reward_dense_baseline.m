function [Reward, IsDone] = reward_dense_baseline(x, x_prev, u_actual, u_prev, VetoTriggered, params, weights)
% REWARD_DENSE_BASELINE Calculates the continuous reinforcement learning score.
%
% Reward Philosophy (Dense Baseline):
%   - Dense Rewards (Calculated every step): 
%       * Potential-Based Shaping: Rewards the agent for getting closer to the pad, slowing down, and staying upright.
%       * Fuel Penalty:       -0.3 for main engine, -0.03 for side engines.
%
%   - Sparse Rewards (Calculated at termination): 
%       * Success:            +100 points (Soft touchdown)
%       * Crash:              -100 points (Hard impact or excessive tilt)
%       * Out of Bounds (OOB):-100 points (Flew outside the flight box)

    % Unpack state variables
    x_pos = x(1);
    y_pos = x(2);
    dx = x(3);
    dy = x(4);
    theta = x(5);
    dtheta = x(6);
    
    % Initialize flags
    IsDone = false;
    Reward = 0;
    
    % Extract dynamic hardware limits from the central struct
    max_T   = params.max_main_thrust;
    max_Tau = params.max_side_torque;
    
    % Remove bottleneck: Weights are now passed in directly from the environment
    
    % --- 1. SPATIAL NORMALIZATION ---
    % Normalize coordinates against expected max boundaries so penalties stay fractional
    norm_x = x_pos / 500000;
    norm_y = y_pos / 15000;
    norm_dx = dx / 2000;
    norm_dy = dy / 150;
    
    norm_x_prev = x_prev(1) / 500000;
    norm_y_prev = x_prev(2) / 15000;
    norm_dx_prev = x_prev(3) / 2000;
    norm_dy_prev = x_prev(4) / 150;
    theta_prev = x_prev(5);
    
    % --- 2. POTENTIAL-BASED REWARD SHAPING (Gym Standard) ---
    % Potential is high when the agent is close to the pad, moving slowly, and upright.
    shaping_prev = -100 * sqrt(norm_x_prev^2 + norm_y_prev^2) - 100 * sqrt(norm_dx_prev^2 + norm_dy_prev^2) - 100 * abs(theta_prev);
    shaping = -100 * sqrt(norm_x^2 + norm_y^2) - 100 * sqrt(norm_dx^2 + norm_dy^2) - 100 * abs(theta);
    
    % The reward is the difference in potential (guarantees a net +100 reward for descending successfully)
    shaping_reward = shaping - shaping_prev;
    
    % --- 3. FUEL PENALTIES (Gym Standard) ---
    norm_T_main = u_actual(1) / max_T;
    norm_T_side = abs(u_actual(2)) / max_Tau;
    
    fuel_penalty = -0.3 * norm_T_main - 0.03 * norm_T_side;
    
    % Sum the continuous rewards
    Reward = shaping_reward + fuel_penalty;
    
    % --- 3. THE SIDECAR PENALTY ---
    % Reduced from -50 to -5. A long 80s burn now costs -20,000 points.
    % This hurts, but it is mathematically better than dying.
    % This teaches the AI that relying on the safety net is worse than braking itself.
    if VetoTriggered
        Reward = Reward + weights.sidecar_veto; 
    end
    
    % --- 4. TERMINAL CONDITIONS ---
    % Check ground contact and assess crash vs successful landing
    if y_pos <= 0
        IsDone = true;
        % Impact tolerances: mark crash if any exceed safe limits
        if abs(dy) > 1.0 || abs(dx) > 0.5 || abs(theta) > 0.1
            % Soft Crash Penalty: Penalize based on impact speed so the AI learns a gradient to slow down!
            impact_speed = sqrt(dx^2 + dy^2);
            crash_severity = min(1.0, impact_speed / 50.0); % Cap at 1.0 (50 m/s)
            
            % Base penalty on speed
            Reward = Reward + (weights.crash * crash_severity); 
            
            % Additional flat penalty for landing sideways or spinning (forces upright landings)
            if abs(theta) > 0.1 || abs(dtheta) > 0.1
                Reward = Reward + (weights.crash * 0.5); 
            end
        else
            Reward = Reward + weights.success; % SUCCESS 
        end
    % Ceiling set to 20,000m. Lateral boundaries expanded to 500,000m to allow for orbital velocity braking.
    % Out-of-bounds terminal case (excess lateral/vertical displacement)
    elseif abs(x_pos) > 500000 || y_pos > 20000
        IsDone = true;
        Reward = Reward + weights.oob; % OOB (Catastrophic penalty to prevent the Sideways Missile exploit)
    end
end
