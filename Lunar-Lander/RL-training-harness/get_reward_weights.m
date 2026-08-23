function weights = get_reward_weights()
    % get_reward_weights: Single source of truth for RL training rewards
    weights.crash = -100;
    weights.success = 100;
    weights.oob = -100;
    weights.sidecar_veto = -10;
end