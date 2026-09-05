function results = run_algorithm_trade(opts)
% RUN_ALGORITHM_TRADE Stage 1 of the experiment: which RL architecture can fly this?
%
% Trains DDPG, TD3, SAC and PPO on identical environments, rewards, networks and sample
% times, with the Developmental Guardian ON. Only the learning rule differs, so the
% ranking is attributable to the algorithm.
%
% This doubles as the smoke test for the whole harness. A short budget here surfaces
% configuration faults - flat reward, episodes ending instantly, NaN rewards - before
% committing hours to the full guardian A/B study in Stage 2.
%
% If opts.auto_continue is true, the winning architecture is handed straight to
% run_ab_study at full budget, but ONLY if it passes health_check. Auto-continuing on a
% broken configuration is how you lose an afternoon.
%
% Inputs (optional struct):
%   opts.agent_types    - default {'ddpg','td3','sac','ppo'}
%   opts.max_episodes   - smoke budget per architecture (default 300)
%   opts.eval_episodes  - evaluation rollouts per architecture (default 30)
%   opts.reward_scheme  - default 'DenseBaseline'
%   opts.guardian_mode  - default 'on'
%   opts.auto_continue  - default true; run Stage 2 on the winner
%   opts.full_episodes  - Stage 2 budget per arm (default 5000)
%   opts.seed           - default 42
%
% Outputs:
%   results - struct with .summary table, .per_agent metrics, .winner, .stage2

    if nargin < 1, opts = struct(); end
    if ~isfield(opts, 'agent_types'),   opts.agent_types   = {'ddpg','td3','sac','ppo'}; end
    if ~isfield(opts, 'max_episodes'),  opts.max_episodes  = 300;             end
    if ~isfield(opts, 'eval_episodes'), opts.eval_episodes = 30;              end
    if ~isfield(opts, 'reward_scheme'), opts.reward_scheme = 'DenseBaseline'; end
    if ~isfield(opts, 'guardian_mode'), opts.guardian_mode = 'on';            end
    if ~isfield(opts, 'auto_continue'), opts.auto_continue = true;            end
    if ~isfield(opts, 'full_episodes'), opts.full_episodes = 5000;            end
    if ~isfield(opts, 'seed'),          opts.seed          = 42;              end
    if ~isfield(opts, 'use_parallel'),  opts.use_parallel  = true;            end

    currentFolder = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(currentFolder, '..');
    addpath(genpath(repoRoot));

    per_agent = struct();
    rows = {};

    for i = 1:numel(opts.agent_types)
        atype = opts.agent_types{i};
        fprintf('\n=================================================\n');
        fprintf(' TRADE STUDY: %s  (%d episodes, guardian %s)\n', ...
            upper(atype), opts.max_episodes, upper(opts.guardian_mode));
        fprintf('=================================================\n');

        rec = struct('agent_type', atype);

        try
            env = LunarLanderEnv(opts.reward_scheme, opts.guardian_mode);
            obsInfo = getObservationInfo(env);
            actInfo = getActionInfo(env);

            agent = build_agent(atype, obsInfo, actInfo, env.params.agent_dt);

            trainOpts = get_training_options(atype, opts.max_episodes);
            trainOpts.Plots = 'none';
            trainOpts.Verbose = true;
            trainOpts.UseParallel = opts.use_parallel;
            % Never stop a smoke run early on the reward criterion - we want the full
            % learning curve from every architecture so the trends are comparable.
            trainOpts.StopTrainingCriteria = 'EpisodeCount';
            trainOpts.StopTrainingValue = opts.max_episodes;

            tstart = tic;
            s = train(agent, env, trainOpts);
            rec.train_seconds = toc(tstart);

            rec.train_stats = s;
            rec.agent = agent;
            rec.reward_curve = s.EpisodeReward;

            % Deterministic evaluation on a fixed scenario set, identical for every arch
            rng(opts.seed);
            eval_env = LunarLanderEnv(opts.reward_scheme, opts.guardian_mode);
            rec.metrics = evaluate_policy(eval_env, agent, opts.eval_episodes);

            rec.health = health_check(rec.reward_curve, rec.metrics, agent, env.params);
            rec.failed = false;
            rec.error = '';

            save(fullfile(repoRoot, sprintf('trade_agent_%s.mat', atype)), 'agent');

        catch ME
            % One architecture failing to build must not abort the whole study. This is
            % also where a MATLAB version incompatibility in a builder will surface.
            fprintf(2, 'ARCHITECTURE %s FAILED: %s\n', upper(atype), ME.message);
            rec.failed = true;
            rec.error = ME.message;
            rec.metrics = empty_metrics();
            rec.health = struct('passed', false, 'reasons', {{ME.message}});
            rec.reward_curve = [];
            rec.train_seconds = NaN;
        end

        per_agent.(atype) = rec;

        m = rec.metrics;
        rows(end+1, :) = {atype, rec.failed, m.landing_rate, m.crash_rate, ...
                          m.oob_rate, m.stall_rate, m.timeout_rate, ...
                          m.mean_reward, m.mean_steps, m.mean_veto_count, ...
                          rec.health.passed}; %#ok<AGROW>
    end

    summary = cell2table(rows, 'VariableNames', ...
        {'Agent', 'Failed', 'LandingRate', 'CrashRate', 'OobRate', 'StallRate', ...
         'TimeoutRate', 'MeanReward', 'MeanSteps', 'MeanVetoCount', 'Healthy'});

    disp(' ');
    disp('=== STAGE 1: ALGORITHM TRADE STUDY ===');
    disp(summary);

    % --- PICK THE WINNER ---
    % Landing rate first, mean reward as the tiebreak. Reward alone is a poor ranking
    % here: potential-based shaping pays out for descending at all, so an architecture
    % that reliably descends and reliably crashes can outscore one that lands sometimes.
    winner = '';
    best = [-inf -inf];
    for i = 1:numel(opts.agent_types)
        atype = opts.agent_types{i};
        rec = per_agent.(atype);
        if rec.failed || ~rec.health.passed
            continue;
        end
        score = [rec.metrics.landing_rate, rec.metrics.mean_reward];
        if score(1) > best(1) || (score(1) == best(1) && score(2) > best(2))
            best = score;
            winner = atype;
        end
    end

    results = struct('summary', summary, 'per_agent', per_agent, ...
                     'winner', winner, 'opts', opts, 'stage2', []);

    if isempty(winner)
        fprintf(2, '\nNO ARCHITECTURE PASSED THE HEALTH CHECK.\n');
        print_diagnostics(per_agent, opts.agent_types);
        fprintf(2, 'Stopping before Stage 2. Fix the harness before spending the compute.\n');
        save(fullfile(repoRoot, 'algorithm_trade_results.mat'), 'results');
        return;
    end

    fprintf('\nWINNER: %s  (landing rate %.1f%%, mean reward %.2f)\n', ...
        upper(winner), 100 * best(1), best(2));

    save(fullfile(repoRoot, 'algorithm_trade_results.mat'), 'results');

    % --- STAGE 2 ---
    if opts.auto_continue
        fprintf('\nHealth check passed. Continuing into Stage 2 guardian A/B on %s.\n', ...
            upper(winner));
        ab_opts = struct();
        ab_opts.agent_type    = winner;
        ab_opts.max_episodes  = opts.full_episodes;
        ab_opts.eval_episodes = 100;
        ab_opts.reward_scheme = opts.reward_scheme;
        ab_opts.seed          = opts.seed;
        ab_opts.use_parallel  = opts.use_parallel;

        results.stage2 = run_ab_study(ab_opts);
        save(fullfile(repoRoot, 'algorithm_trade_results.mat'), 'results');
    else
        fprintf('\nauto_continue is off. Run run_ab_study with agent_type=''%s'' when ready.\n', winner);
    end
end


function h = health_check(reward_curve, m, agent, params)
% Gate before committing to the full study. Every condition here corresponds to a
% specific failure the original harness exhibited, so a red flag names the regression.

    reasons = {};

    if isempty(reward_curve)
        reasons{end+1} = 'no reward curve produced';
    else
        n = numel(reward_curve);
        if any(~isfinite(reward_curve))
            reasons{end+1} = 'reward curve contains NaN or Inf';
        end
        % Reward must be trending up. The original symptom was a flat curve.
        if n >= 20
            k = max(1, floor(n / 4));
            early = mean(reward_curve(1:k));
            late  = mean(reward_curve(end-k+1:end));
            if late <= early
                reasons{end+1} = sprintf('reward not improving (first quarter %.2f, last quarter %.2f)', early, late);
            end
        end
    end

    % Episodes must last meaningfully longer than the old few-second collapse. At 10 Hz,
    % 100 agent steps is 10 seconds of flight.
    if m.mean_steps < 100
        reasons{end+1} = sprintf('episodes too short (mean %.0f agent steps, expected >100)', m.mean_steps);
    end

    % Something other than crashing must be happening at least occasionally.
    if m.landing_rate == 0 && m.stall_rate == 0
        reasons{end+1} = 'every evaluation episode ended in a crash, OOB or timeout';
    end

    % The guardian should be an occasional barrier, not a continuous co-pilot. One
    % engagement per agent step means the envelope has collapsed again.
    if m.mean_veto_count > 0.5 * max(m.mean_steps, 1)
        reasons{end+1} = sprintf('guardian engaging almost every step (%.0f engagements over %.0f steps)', ...
            m.mean_veto_count, m.mean_steps);
    end

    % The policy must still be a FUNCTION OF THE STATE. A saturated actor trains without
    % error, produces a plausible reward curve, and evaluates to a clean 100% of one
    % outcome - so none of the checks above catch it. DDPG at 1200 episodes commanded
    % thrust between +0.9894 and +0.9928 across a state sweep from 5 m to 2500 m and was
    % only caught by hand. It is not caught by hand again.
    if nargin >= 4 && ~isempty(agent)
        [diverse, div_report] = check_policy_diversity(agent, params);
        if ~diverse
            reasons{end+1} = div_report.reason;
        end
    end

    h = struct('passed', isempty(reasons), 'reasons', {reasons});
end


function print_diagnostics(per_agent, agent_types)
    fprintf(2, '\n--- DIAGNOSTICS ---\n');
    for i = 1:numel(agent_types)
        atype = agent_types{i};
        rec = per_agent.(atype);
        fprintf(2, '%s:\n', upper(atype));
        if rec.failed
            fprintf(2, '  build/train error: %s\n', rec.error);
        end
        for j = 1:numel(rec.health.reasons)
            fprintf(2, '  - %s\n', rec.health.reasons{j});
        end
    end
end


function m = empty_metrics()
    m = struct('n', 0, 'landing_rate', NaN, 'crash_rate', NaN, 'oob_rate', NaN, ...
        'stall_rate', NaN, 'timeout_rate', NaN, 'mean_reward', NaN, 'std_reward', NaN, ...
        'mean_veto_count', NaN, 'mean_veto_steps', NaN, 'mean_impact_speed', NaN, ...
        'mean_steps', NaN, 'outcomes', {{}}, 'rewards', []);
end


% evaluate_policy lives in its own file so Stage 1 and Stage 2 measure identically.
