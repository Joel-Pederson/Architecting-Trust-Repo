function ep = rollout_episode(env, agent, max_steps)
% ROLLOUT_EPISODE Runs one deterministic episode and returns full physical telemetry.
%
% This drives the environment directly rather than going through sim(), for two reasons:
%   1. It returns the TRUE physics state in real units, so nothing downstream has to
%      invert the observation normalizers. Getting those constants out of sync is what
%      made the visualizer render altitudes 6.7x too large and lateral position 500x
%      too large.
%   2. It exposes the per-episode safety metrics (outcome, sidecar engagement count,
%      time under guardian authority) that the A/B study needs.
%
% IMPORTANT: exploration is disabled before rolling out. SAC and PPO carry stochastic
% actors, so getAction SAMPLES from the policy distribution unless UseExplorationPolicy is
% false. Leaving it on would measure a deliberately noisy policy and would penalise
% exactly the two architectures whose exploration is built into the policy itself,
% making the trade study unfair in a way that is very hard to see in the results.
%
% Inputs:
%   env       - LunarLanderEnv instance
%   agent     - trained RL agent
%   max_steps - (Optional) episode cap in agent steps. Defaults to params.max_agent_steps.
%
% Outputs:
%   ep - struct with fields:
%        .t, .states (8xN), .controls (2xN), .veto (1xN logical)
%        .reward, .steps, .outcome, .veto_count, .veto_steps
%        .touchdown_dx, .touchdown_dy, .touchdown_theta

    if nargin < 3 || isempty(max_steps)
        max_steps = env.params.max_agent_steps;
    end

    % Evaluate the greedy policy. Restored afterwards so a caller that reuses the agent
    % for further training does not silently lose its exploration.
    restore_exploration = false;
    if isprop(agent, 'UseExplorationPolicy')
        restore_exploration = agent.UseExplorationPolicy;
        agent.UseExplorationPolicy = false;
    end
    cleanup = onCleanup(@() restore_flag(agent, restore_exploration));

    obs = reset(env);

    states   = zeros(8, max_steps + 1);
    controls = zeros(2, max_steps + 1);
    veto     = false(1, max_steps + 1);

    states(:, 1) = env.State;
    total_reward = 0;
    n = 1;

    for i = 1:max_steps
        action = getAction(agent, {obs});
        if iscell(action)
            action = action{1};
        end
        action = reshape(double(action), [], 1);

        [obs, r, done, logs] = step(env, action);

        n = i + 1;
        states(:, n)   = logs.State;
        controls(:, n) = logs.Control;
        veto(n)        = logs.VetoActive;
        total_reward   = total_reward + r;

        if done
            break;
        end
    end

    % Trim the preallocated buffers to the length actually flown
    states   = states(:, 1:n);
    controls = controls(:, 1:n);
    veto     = veto(1:n);

    ep = struct();
    ep.t        = (0:n - 1) * env.params.agent_dt;
    ep.states   = states;
    ep.controls = controls;
    ep.veto     = veto;
    ep.reward   = total_reward;
    ep.steps    = n - 1;

    % An episode that hit the step cap without terminating is a timeout, not a landing.
    if isempty(env.Outcome) || strcmp(env.Outcome, 'flying')
        ep.outcome = 'timeout';
    else
        ep.outcome = env.Outcome;
    end

    ep.phase      = env.Phase;
    ep.veto_count = env.VetoCount;
    ep.veto_steps = env.VetoSteps;

    % Terminal condition at the moment the episode ended, for landing-quality analysis
    ep.touchdown_dx    = states(3, end);
    ep.touchdown_dy    = states(4, end);
    ep.touchdown_theta = states(5, end);
end


function restore_flag(agent, value)
% Restores the agent's exploration setting once the rollout finishes or errors.
    if isprop(agent, 'UseExplorationPolicy')
        agent.UseExplorationPolicy = value;
    end
end
