function [Reward, IsDone, Outcome] = reward_sparse_only(x, ~, ~, ~, ~, params, weights)
% REWARD_SPARSE_ONLY Calculates the reinforcement learning score using ONLY sparse rewards.
%
% Kept as the ablation arm: it isolates how much of the agent's performance comes from
% reward shaping versus the environment itself.
%
% Reward Philosophy (Sparse Only):
%   - Dense Rewards: NONE. The AI receives 0 points per step.
%   - Sparse Rewards (at termination):
%       * Success: weights.success
%       * Crash:   weights.crash
%       * OOB:     weights.oob
%
% NOTE ON SIGNATURE: this takes seven arguments to match reward_dense_baseline exactly.
% It previously declared six while LunarLanderEnv called it with seven, so `weights`
% silently received the params struct and every terminal step died on an undefined
% field. Both schemes must stay signature-compatible; the environment dispatches to
% them through the same call.

    y_pos = x(2);
    x_pos = x(1);
    dx    = x(3);
    dy    = x(4);
    theta = x(5);

    IsDone = false;
    Outcome = 'flying';
    Reward = 0; % No dense rewards during flight

    % --- TERMINAL CONDITIONS ---
    if y_pos <= params.touchdown_alt
        IsDone = true;
        is_hard = abs(dy) > params.max_touchdown_dy || ...
                  abs(dx) > params.max_touchdown_dx || ...
                  abs(theta) > params.max_touchdown_tilt;
        if is_hard
            Outcome = 'crashed';
            Reward = weights.crash;
        else
            Outcome = 'landed';
            Reward = weights.success;
        end
    elseif abs(x_pos) > params.max_abs_x || y_pos > params.max_alt
        IsDone = true;
        Outcome = 'oob';
        Reward = weights.oob;
    end

    Reward = Reward / weights.reward_scale;
end
