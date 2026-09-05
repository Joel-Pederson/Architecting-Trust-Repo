function results = analyze_validation_agent(agent_type, n_episodes)
% ANALYZE_VALIDATION_AGENT Resolve whether a policy can land at all, and under what.
%
% A zero greedy landing rate has two very different explanations, and the training reward
% curve cannot distinguish them:
%
%   (a) The policy genuinely cannot land. Nothing in its behaviour reaches the touchdown
%       box, so the task or the budget is the problem.
%   (b) The policy CAN land while exploring but has not consolidated it into the greedy
%       action. PPO's run showed 108 positive-reward episodes and five above +4.5 with a
%       0%% greedy landing rate, which is exactly this signature.
%
% Case (b) is an exploitation/consolidation problem and calls for off-policy replay or
% longer training. Case (a) is not. Running the saved agent both ways separates them.
%
% Also breaks results out per curriculum phase, because a policy that lands the 50 m
% scenario but not the 2500 m descent is a partial success worth reporting, not a failure.
%
% Inputs:
%   agent_type - 'ppo' | 'sac' | 'td3' | 'ddpg'; loads validation_agent_<type>.mat
%   n_episodes - rollouts per configuration (default 100)
%
% Outputs:
%   results - struct with greedy and exploratory metrics, and the per-phase breakdown

    if nargin < 2 || isempty(n_episodes), n_episodes = 100; end

    currentFolder = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(currentFolder, '..');
    addpath(genpath(repoRoot));

    f = fullfile(repoRoot, sprintf('validation_agent_%s.mat', lower(agent_type)));
    if ~isfile(f)
        error('analyzeValidationAgent:NoAgent', ...
            ['No saved agent at %s. Only runs from after the agent-persistence fix have ' ...
             'one; earlier validation runs did not save their policy.'], f);
    end
    data = load(f, 'agent');
    agent = data.agent;

    results = struct();
    for mode = {'greedy', 'exploratory'}
        label = mode{1};
        rng(42);   % identical scenarios for both modes

        env = LunarLanderEnv('DenseBaseline', 'on');
        if isprop(agent, 'UseExplorationPolicy')
            agent.UseExplorationPolicy = strcmp(label, 'exploratory');
        end

        % evaluate_policy forces greedy internally, so drive rollouts directly here to
        % keep exploration under this function's control.
        outcomes = cell(n_episodes,1); phases = zeros(n_episodes,1);
        impacts  = zeros(n_episodes,1); rewards = zeros(n_episodes,1);
        env.CurriculumWeights = env.params.eval_curriculum_weights;

        for k = 1:n_episodes
            ep = rollout_with_mode(env, agent, strcmp(label,'exploratory'));
            outcomes{k} = ep.outcome; phases(k) = ep.phase;
            impacts(k)  = sqrt(ep.touchdown_dx^2 + ep.touchdown_dy^2);
            rewards(k)  = ep.reward;
        end

        landed = strcmp(outcomes, 'landed');
        m = struct('landing_rate', mean(landed), 'mean_reward', mean(rewards));
        for ph = 1:3
            sel = phases == ph;
            if any(sel)
                m.landing_by_phase(ph) = mean(landed(sel));
                m.n_by_phase(ph) = nnz(sel);
            else
                m.landing_by_phase(ph) = NaN; m.n_by_phase(ph) = 0;
            end
        end
        m.best_reward = max(rewards);
        results.(label) = m;

        fprintf('%-12s landing %5.1f%%   mean reward %+7.2f   best %+7.2f   [P1 %.0f%% P2 %.0f%% P3 %.0f%%]\n', ...
            label, 100*m.landing_rate, m.mean_reward, m.best_reward, ...
            100*m.landing_by_phase(1), 100*m.landing_by_phase(2), 100*m.landing_by_phase(3));
    end

    fprintf('\n');
    if results.exploratory.landing_rate > 0 && results.greedy.landing_rate == 0
        fprintf(['DIAGNOSIS: the policy CAN land while exploring but not greedily.\n' ...
                 'This is a consolidation problem - the successes exist but are not being\n' ...
                 'exploited. Off-policy replay or a longer budget is the lever, not reward design.\n']);
    elseif results.greedy.landing_rate > 0
        fprintf('DIAGNOSIS: the greedy policy lands. The task is solved at this budget.\n');
    else
        fprintf(['DIAGNOSIS: no landings in either mode. The policy is not reaching the\n' ...
                 'touchdown box at all, so budget or task difficulty is the constraint -\n' ...
                 'not exploitation of rare successes.\n']);
    end
end


function ep = rollout_with_mode(env, agent, explore)
% Minimal rollout that respects the caller's exploration setting.
    if isprop(agent, 'UseExplorationPolicy')
        agent.UseExplorationPolicy = explore;
    end
    obs = reset(env);
    ep = struct('outcome','timeout','phase',env.Phase,'reward',0, ...
                'touchdown_dx',0,'touchdown_dy',0);
    for i = 1:env.params.max_agent_steps
        a = getAction(agent, {obs});
        if iscell(a), a = a{1}; end
        [obs, r, done] = step(env, reshape(double(a), [], 1));
        ep.reward = ep.reward + r;
        if done, break; end
    end
    if ~isempty(env.Outcome) && ~strcmp(env.Outcome,'flying')
        ep.outcome = env.Outcome;
    end
    ep.phase = env.Phase;
    ep.touchdown_dx = env.State(3);
    ep.touchdown_dy = env.State(4);
end
