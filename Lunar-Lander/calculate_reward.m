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
    
    % --- 1. Continuous Penalties ---
    dist_penalty = -0.1 * sqrt(x_pos^2 + y_pos^2);
    tilt_penalty = -0.1 * abs(theta);
    
    % Normalize controls mathematically to a percentage (0.0 to 1.0)
    % This prevents the massive numeric difference between Newtons and Newton-meters 
    % from skewing the AI's learning priorities.
    norm_T_main = u_actual(1) / max_T;
    norm_T_side = u_actual(2) / max_Tau;
    
    fuel_penalty = -0.5 * (norm_T_main^2 + norm_T_side^2); 
    
    % Smoothness penalty normalized against the same bounds
    norm_u_actual = [norm_T_main; norm_T_side];
    norm_u_prev   = [u_prev(1) / max_T; u_prev(2) / max_Tau];
    smoothness_penalty = -0.1 * sum((norm_u_actual - norm_u_prev).^2);
    
    oob_penalty = -0.01 * (max(0, abs(x_pos) - 80)^2 + max(0, y_pos - 80)^2);
    
    Reward = dist_penalty + tilt_penalty + fuel_penalty + smoothness_penalty + oob_penalty;
    
    % --- 2. The Sidecar Penalty ---
    % If the sidecar had to save the agent, slap it with a heavy penalty.
    % This teaches the AI that relying on the safety net is worse than braking itself.
    if VetoTriggered
        Reward = Reward - 50; 
    end
    
    % --- 3. Terminal Conditions ---
    % Check ground contact and assess crash vs successful landing
    if y_pos <= 0
        IsDone = true;
        % Impact tolerances: mark crash if any exceed safe limits
        if abs(dy) > 1.0 || abs(dx) > 0.5 || abs(theta) > 0.1
            Reward = Reward - 1000; % CRASH
        else
            Reward = Reward + 1000; % SUCCESS
        end
    % Ceiling set to 20,000m. Lateral boundaries expanded to 5,000m.
    % Out-of-bounds terminal case (excess lateral/vertical displacement)
    elseif abs(x_pos) > 5000 || y_pos > 20000
        % Ceiling set to 20,000m. Lateral boundaries expanded to 5,000m.
        IsDone = true;
        Reward = Reward - 500; 
    end
end