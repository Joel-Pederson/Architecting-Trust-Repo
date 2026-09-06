function m = evaluate_policy(env, agent, n_episodes)
% EVALUATE_POLICY Aggregates safety and performance metrics over deterministic rollouts.
%
% Shared by run_algorithm_trade (Stage 1) and run_ab_study (Stage 2) so both stages
% measure identically - a trade study whose two halves compute "landing rate" slightly
% differently is worse than no trade study.
%
% Exploration is disabled inside rollout_episode, so stochastic policies (SAC, PPO) are
% evaluated on their mean action rather than a sample.
%
% Inputs:
%   env        - LunarLanderEnv, already configured with the desired guardian mode
%   agent      - trained RL agent
%   n_episodes - number of rollouts
%
% Outputs:
%   m - struct of aggregate metrics. Outcome rates sum to 1 across
%       landed / crashed / oob / stalled / timeout.

    % Evaluate on the UNIFORM curriculum mix regardless of what the agent trained on.
    % Training uses a P1-weighted diet to make the success region reachable; scoring on
    % that same easy diet would inflate the landing rate and make the guardian A/B look
    % better than it is. Train on one distribution, report on another.
    if isfield(env.params, 'eval_curriculum_weights')
        env.CurriculumWeights = env.params.eval_curriculum_weights;
    end

    outcomes      = cell(n_episodes, 1);
    rewards       = zeros(n_episodes, 1);
    veto_counts   = zeros(n_episodes, 1);
    veto_steps    = zeros(n_episodes, 1);
    impact_speeds = zeros(n_episodes, 1);
    steps         = zeros(n_episodes, 1);
    phases        = zeros(n_episodes, 1);

    for k = 1:n_episodes
        ep = rollout_episode(env, agent);
        outcomes{k}      = ep.outcome;
        rewards(k)       = ep.reward;
        veto_counts(k)   = ep.veto_count;
        veto_steps(k)    = ep.veto_steps;
        impact_speeds(k) = sqrt(ep.touchdown_dx^2 + ep.touchdown_dy^2);
        steps(k)         = ep.steps;
        phases(k)        = ep.phase;
    end

    m = struct();
    m.n                 = n_episodes;
    m.landing_rate      = mean(strcmp(outcomes, 'landed'));
    m.crash_rate        = mean(strcmp(outcomes, 'crashed'));
    m.oob_rate          = mean(strcmp(outcomes, 'oob'));
    m.stall_rate        = mean(strcmp(outcomes, 'stalled'));
    m.timeout_rate      = mean(strcmp(outcomes, 'timeout'));
    m.mean_reward       = mean(rewards);
    m.std_reward        = std(rewards);
    m.mean_veto_count   = mean(veto_counts);
    m.mean_veto_steps   = mean(veto_steps);
    % Impact speed is only meaningful for episodes that actually reached the ground.
    % Averaging it over timeouts folds in "velocity at the step cap" for landers still
    % airborne, which produced nonsense like a 132 m/s mean impact on a 50 m scenario.
    grounded = strcmp(outcomes, 'landed') | strcmp(outcomes, 'crashed');
    if any(grounded)
        m.mean_impact_speed = mean(impact_speeds(grounded));
    else
        m.mean_impact_speed = NaN;
    end
    m.n_grounded        = nnz(grounded);
    m.mean_steps        = mean(steps);
    m.outcomes          = outcomes;
    m.rewards           = rewards;
    m.phases            = phases;

    % Per-phase breakdown. A headline landing rate averaged over a mix of a 50 m hover
    % and a 2500 m terminal descent hides which regime actually works, which is exactly
    % the claim the paper needs to be precise about.
    landed = strcmp(outcomes, 'landed');
    % Every CONFIGURED phase, not a hardcoded three. The powered descent was invisible
    % to every architecture comparison until this read the phase count instead.
    for ph = 1:numel(env.params.phase_max_steps)
        sel = (phases == ph);
        sel_g = sel & grounded;
        if any(sel)
            m.landing_rate_by_phase(ph) = mean(landed(sel));
            if any(sel_g)
                m.impact_by_phase(ph)   = mean(impact_speeds(sel_g));
            else
                m.impact_by_phase(ph)   = NaN;
            end
            m.n_by_phase(ph)            = nnz(sel);
        else
            m.landing_rate_by_phase(ph) = NaN;
            m.impact_by_phase(ph)       = NaN;
            m.n_by_phase(ph)            = 0;
        end
    end
end
