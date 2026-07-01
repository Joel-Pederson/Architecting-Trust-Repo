function [Reward, IsDone] = calculate_reward(x, u_actual, u_prev, VetoTriggered, params)
    % Unpack state variables
    x_pos = x(1);
    y_pos = x(2);
    dx = x(3);
    dy = x(4);
    theta = x(5);
    
    % Initialize flags
    IsDone = false;
    Reward = 0;
    
    % Extract dynamic hardware limits from the central struct
    max_T   = params.max_main_thrust;
    max_Tau = params.max_side_torque;
    
    % --- 1. SPATIAL NORMALIZATION ---
    % Normalize coordinates against expected max boundaries so penalties stay fractional
    norm_x = x_pos / 5000;
    norm_y = y_pos / 15000;
    
    % --- 2. CONTINUOUS PENALTIES ---
    % Distance penalty (Scaled to be max -0.1 per step)
    dist_penalty = -0.1 * sqrt(norm_x^2 + norm_y^2);
    
    % Tilt penalty
    tilt_penalty = -0.1 * abs(theta);
    
    % Thrust penalties (Normalized to max -0.5 per step)
    % This prevents the massive numeric difference between Newtons and Newton-meters 
    % from skewing the AI's learning priorities.
    norm_T_main = u_actual(1) / max_T;
    norm_T_side = u_actual(2) / max_Tau;
    fuel_penalty = -0.5 * (norm_T_main^2 + norm_T_side^2); 
    
    % Smoothness penalty normalized against the same bounds
    norm_u_actual = [norm_T_main; norm_T_side];
    norm_u_prev   = [u_prev(1) / max_T; u_prev(2) / max_Tau];
    smoothness_penalty = -0.1 * sum((norm_u_actual - norm_u_prev).^2);
    
    % Penalize agent for being above 80 meters - creating a mathematical gravitational pull. 
    % The only way the AI can stop bleeding points is to descend into that 80-meter safe box and land.
    % Creates a gentle, mathematically stable pull toward the 80m box
    beyond_x = max(0, abs(x_pos) - 80) / 5000;
    beyond_y = max(0, y_pos - 80) / 15000;
    beyond_bounds_penalty = -0.1 * (beyond_x + beyond_y);
    
    % Sum the continuous rewards
    Reward = dist_penalty + tilt_penalty + fuel_penalty + smoothness_penalty + beyond_bounds_penalty;
    
    % --- 3. THE SIDECAR PENALTY ---
    % Reduced from -50 to -5. A long 80s burn now costs -20,000 points.
    % This hurts, but it is mathematically better than dying.
    % This teaches the AI that relying on the safety net is worse than braking itself.
    if VetoTriggered
        Reward = Reward - 5; 
    end
    
    % --- 4. TERMINAL CONDITIONS ---
    % Check ground contact and assess crash vs successful landing
    if y_pos <= 0
        IsDone = true;
        % Impact tolerances: mark crash if any exceed safe limits
        if abs(dy) > 1.0 || abs(dx) > 0.5 || abs(theta) > 0.1
            Reward = Reward - 50000; % CRASH (Catastrophic penalty)
        else
            Reward = Reward + 10000; % SUCCESS 
        end
    % Ceiling set to 20,000m. Lateral boundaries expanded to 5,000m.
    % Out-of-bounds terminal case (excess lateral/vertical displacement)
    elseif abs(x_pos) > 5000 || y_pos > 20000
        IsDone = true;
        Reward = Reward - 50000; % OOB (Catastrophic penalty to prevent the Sideways Missile exploit)
    end
end