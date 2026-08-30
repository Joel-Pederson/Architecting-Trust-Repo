function weights = get_reward_weights()
    % get_reward_weights: Single source of truth for RL training rewards
    weights.crash = -500;
    weights.success = 500;
    weights.oob = -500;
    weights.sidecar_veto = -10;
end