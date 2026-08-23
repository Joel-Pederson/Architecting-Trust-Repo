function [Reward, IsDone] = reward_sparse_only(x, x_prev, ~, ~, ~, weights)
% REWARD_SPARSE_ONLY Calculates the reinforcement learning score using ONLY sparse rewards.
%
% Reward Philosophy (Sparse Only):
%   - Dense Rewards (Calculated every step): 
%       * NONE. The AI receives 0 points per step.
%
%   - Sparse Rewards (Calculated at termination): 
%       * Success:            +10,000 points (Soft touchdown)
%       * Crash:              -50,000 points (Hard impact or excessive tilt)
%       * Out of Bounds (OOB):-50,000 points (Flew outside the 500km x 20km flight box)

    % Unpack state variables
    x_pos = x(1);
    y_pos = x(2);
    dx = x(3);
    dy = x(4);
    theta = x(5);
    
    % Initialize flags
    IsDone = false;
    Reward = 0; % No dense rewards during flight!
    
    % Weights passed in directly
    
    % --- TERMINAL CONDITIONS ---
    % Check ground contact and assess crash vs successful landing
    if y_pos <= 0
        IsDone = true;
        % Impact tolerances: mark crash if any exceed safe limits
        if abs(dy) > 1.0 || abs(dx) > 0.5 || abs(theta) > 0.1
            Reward = weights.crash; % CRASH 
        else
            Reward = weights.success; % SUCCESS 
        end
    % Out-of-bounds terminal case
    elseif abs(x_pos) > 500000 || y_pos > 20000
        IsDone = true;
        Reward = weights.oob; % OOB
    end
end
