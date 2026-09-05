function [ok, report] = check_policy_diversity(agent, params, min_spread)
% CHECK_POLICY_DIVERSITY Detects a collapsed (state-independent) deterministic policy.
%
% A saturated actor is the failure mode that is easiest to mistake for "still learning".
% It trains without error, produces a plausible reward curve, and evaluates to a clean
% 100% of a single outcome - but the policy is a constant. DDPG at 1200 episodes on this
% harness commanded thrust between +0.9894 and +0.9928 across a state sweep spanning
% 5 m to 2500 m of altitude and -0.5 to -40 m/s of descent rate: a spread of 0.0034 over
% an action range of 2.0. It had stopped being a function of the state at all.
%
% Catching that automatically matters because the alternative is noticing by hand, and
% the symptom (0% landings) looks identical to an agent that simply needs more training.
%
% Inputs:
%   agent      - trained RL agent
%   params     - get_sim_params
%   min_spread - (Optional) minimum required action range across the probe states.
%                Default 0.015. CALIBRATED, not guessed: the collapsed DDPG agent scored
%                0.0034 and a freshly initialised TD3 scores about 0.049, so the two are
%                14x apart and the threshold sits between them with margin on both sides.
%                An earlier default of 0.05 sat on the WRONG side and flagged untrained
%                agents - a health gate that fires at episode zero is worthless.
%
% Outputs:
%   ok     - false if the policy looks collapsed
%   report - struct with .spread (per action dim), .actions, .states, .reason

    if nargin < 3 || isempty(min_spread), min_spread = 0.015; end

    % States chosen to span the situations a lander must distinguish between: high and
    % low, fast and slow, drifting and centred. A policy that responds to none of these
    % is not a controller.
    probe = [   0    50   -2    0    0;
                0    50   -8    0    0;
                0    50   -1    2  0.1;
                0   500  -10    0    0;
                0   500  -25    0    0;
                0  2500  -25    0    0;
              100  2500  -40   10 -0.1;
                0    10   -1    0    0;
                0    10   -5    0    0;
                0     5 -0.5    0    0];

    restore = [];
    if isprop(agent, 'UseExplorationPolicy')
        restore = agent.UseExplorationPolicy;
        agent.UseExplorationPolicy = false;
    end
    cleanup = onCleanup(@() restore_flag(agent, restore));

    n = size(probe, 1);
    actions = zeros(n, 2);
    for i = 1:n
        x = [probe(i,1); probe(i,2); probe(i,4); probe(i,3); probe(i,5); 0; ...
             params.max_main_fuel; params.max_rcs_fuel];
        a = getAction(agent, {get_ai_observation(x, params)});
        if iscell(a), a = a{1}; end
        actions(i,:) = double(reshape(a, 1, []));
    end

    spread = max(actions, [], 1) - min(actions, [], 1);

    % Being PINNED AT A BOUND is the specific pathology, and it separates a collapsed
    % trained policy from a merely untrained one: the failed DDPG agent sat at +0.989
    % thrust everywhere, whereas a fresh network hovers near zero. Reported separately so
    % the two look different in the diagnostics rather than both reading "constant".
    saturated = abs(mean(actions, 1)) > 0.9;

    report = struct();
    report.spread    = spread;
    report.actions   = actions;
    report.states    = probe;
    report.saturated = saturated;
    ok = all(spread >= min_spread);

    reasons = {};
    if ~ok
        reasons{end+1} = sprintf(['policy is effectively constant: action spread ' ...
            '[%.4f %.4f] across a state sweep from 5 m to 2500 m (need >= %.3f)'], ...
            spread(1), spread(2), min_spread);
    end
    if any(saturated & (spread < min_spread))
        reasons{end+1} = sprintf(['and pinned at a control bound (mean action ' ...
            '[%+.3f %+.3f]) - this is actor saturation, not undertraining'], ...
            mean(actions(:,1)), mean(actions(:,2)));
    end
    report.reason = strjoin(reasons, ' ');
end


function restore_flag(agent, value)
    if ~isempty(value) && isprop(agent, 'UseExplorationPolicy')
        agent.UseExplorationPolicy = value;
    end
end
