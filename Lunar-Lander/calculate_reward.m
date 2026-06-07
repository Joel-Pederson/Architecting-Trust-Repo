function [Reward, IsDone] = calculate_reward(x, u, u_prev)
    % --- 1. Unpack state variables
    x_pos = x(1);
    y_pos = x(2);
    dx = x(3);
    dy = x(4);
    theta = x(5);
    
    % Unpack controls
    T_main = u(1);
    T_side = u(2);
    
    % Initialize flags
    IsDone = false;
    Reward = 0;
    
    % --- 2. Continuous Penalties (Shape the behavior). Dense Rewards
    % Penalize distance from target (0,0)
    dist_penalty = -0.1 * sqrt(x_pos^2 + y_pos^2);
    
    % Penalize tilt (keep it mostly upright)
    tilt_penalty = -0.1 * abs(theta);
    
    % HEAVY Fuel Penalty (Incentivizes the dangerous "Suicide Burn")
    fuel_penalty = -0.5 * (T_main^2 + T_side^2); 
    
    % Action Smoothness Penalty (Anti-Chatter)
    % Penalizes large, sudden changes in thrust between timesteps
    smoothness_penalty = -0.1 * sum((u - u_prev).^2);
    
    % Soft Out-of-Bounds Penalty
    % Starts penalizing quadratically ONLY when the lander strays past 80 meters.
    oob_penalty = -0.01 * (max(0, abs(x_pos) - 80)^2 + max(0, y_pos - 80)^2);
    
    % Sum all continuous penalties
    Reward = dist_penalty + tilt_penalty + fuel_penalty + smoothness_penalty + oob_penalty;
    
    % --- 3. Terminal Conditions (End of the episode)
    % Define a "crash" vs a "safe landing". Sparse reward.
    if y_pos <= 0
        IsDone = true;
        
        % Check velocity and tilt upon hitting the ground
        if abs(dy) > 1.0 || abs(dx) > 0.5 || abs(theta) > 0.1
            % CRASH: Hit the ground too fast or too tilted
            Reward = Reward - 1000; 
        else
            % SUCCESS: Soft landing
            Reward = Reward + 1000;
        end
    elseif abs(x_pos) > 100 || y_pos > 100
        % Hard Out of bounds (flew away)
        IsDone = true;
        Reward = Reward - 500;
    end
end