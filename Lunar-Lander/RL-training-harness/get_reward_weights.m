function weights = get_reward_weights()
    % get_reward_weights: Single source of truth for RL training rewards
    weights.crash = -50000;
    weights.success = 10000;
    weights.oob = -50000;
    weights.sidecar_veto = -5;
end