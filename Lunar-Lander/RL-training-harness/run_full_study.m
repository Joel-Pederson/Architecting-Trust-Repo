function results = run_full_study(opts)
% RUN_FULL_STUDY Top-level experiment driver: validate, then commit the full budget.
%
% Two phases, deliberately sequenced:
%
%   PHASE A - VALIDATION. Trains one architecture (PPO by default, the Stage 1 winner)
%   at a moderate budget and asks a single question: does the agent ever land? Stage 1
%   produced ZERO landings in 1200 episodes, and a random-policy probe produced zero in
%   360 more, meaning the +5 touchdown reward had never once entered the replay buffer.
%   The curriculum was reweighted toward the reachable Phase 1 scenario to fix that. If
%   that fix did not work, no amount of extra compute will help, and this phase stops
%   before the expensive part rather than after it.
%
%   PHASE B - FULL GUARDIAN A/B across every architecture. Each gets run_ab_study's 2x2
%   (trained with/without the Developmental Guardian, evaluated with/without the
%   Operational Sidecar). Results are saved incrementally after each architecture so a
%   failure twelve hours in does not lose the work already done.
%
% Inputs (optional struct):
%   opts.validate_agent    - default 'ppo'
%   opts.validate_episodes - default 1500
%   opts.eval_episodes     - default 50 for validation, 100 for the A/B
%   opts.agent_types       - default {'ddpg','td3','sac','ppo'}
%   opts.full_episodes     - default 5000 per arm
%   opts.skip_validation   - default false; set true to force straight into Phase B
%   opts.seed              - default 42
%
% Outputs:
%   results - struct with .validation, .ab (per architecture), .completed

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'validate_agent'),    opts.validate_agent    = 'ppo';   end
    if ~isfield(opts,'validate_episodes'), opts.validate_episodes = 1500;    end
    if ~isfield(opts,'eval_episodes'),     opts.eval_episodes     = 50;      end
    if ~isfield(opts,'agent_types'),       opts.agent_types = {'ddpg','td3','sac','ppo'}; end
    if ~isfield(opts,'full_episodes'),     opts.full_episodes     = 5000;    end
    if ~isfield(opts,'skip_validation'),   opts.skip_validation   = false;   end
    if ~isfield(opts,'seed'),              opts.seed              = 42;      end

    currentFolder = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(currentFolder, '..');
    addpath(genpath(repoRoot));

    params = get_sim_params();
    results = struct('validation', [], 'ab', struct(), 'completed', {{}}, 'opts', opts);
    outfile = fullfile(repoRoot, 'full_study_results.mat');

    fprintf('\n########## CURRICULUM ##########\n');
    fprintf('Training mix   : P1 %.0f%%  P2 %.0f%%  P3 %.0f%%\n', 100*params.curriculum_weights);
    fprintf('Evaluation mix : P1 %.0f%%  P2 %.0f%%  P3 %.0f%%  (uniform, deliberately)\n', ...
        100*params.eval_curriculum_weights);

    % ---------- PHASE A ----------
    if ~opts.skip_validation
        fprintf('\n########## PHASE A: VALIDATION (%s, %d episodes) ##########\n', ...
            upper(opts.validate_agent), opts.validate_episodes);

        env = LunarLanderEnv('DenseBaseline', 'on');
        agent = build_agent(opts.validate_agent, getObservationInfo(env), ...
            getActionInfo(env), env.params.agent_dt);

        trainOpts = get_training_options(opts.validate_agent, opts.validate_episodes);
        trainOpts.Plots = 'none';
        trainOpts.Verbose = true;
        trainOpts.StopTrainingCriteria = 'EpisodeCount';
        trainOpts.StopTrainingValue = opts.validate_episodes;

        s = train(agent, env, trainOpts);

        rng(opts.seed);
        m = evaluate_policy(LunarLanderEnv('DenseBaseline','on'), agent, opts.eval_episodes);

        results.validation = struct('agent_type', opts.validate_agent, ...
            'metrics', m, 'reward_curve', s.EpisodeReward);
        save(outfile, 'results');

        % Persist the validation agent itself. Without it, a run that ends with an
        % interesting policy leaves nothing to interrogate afterwards - the reward curve
        % alone cannot tell you whether rare high-scoring episodes were real landings or
        % just excellent descents, and re-running costs another full budget to find out.
        save(fullfile(repoRoot, sprintf('validation_agent_%s.mat', ...
            lower(opts.validate_agent))), 'agent');

        report_metrics(opts.validate_agent, m, s.EpisodeReward);

        if m.landing_rate <= 0
            fprintf(2, ['\nVALIDATION FAILED: still zero landings after %d episodes.\n' ...
                'The curriculum reweighting did not make the success region reachable.\n' ...
                'Stopping before the full study - more compute will not fix this.\n'], ...
                opts.validate_episodes);
            return;
        end

        fprintf('\nVALIDATION PASSED: landing rate %.1f%%. Committing to the full study.\n', ...
            100*m.landing_rate);
    end

    % ---------- PHASE B ----------
    fprintf('\n########## PHASE B: GUARDIAN A/B, ALL ARCHITECTURES ##########\n');
    for i = 1:numel(opts.agent_types)
        atype = opts.agent_types{i};
        fprintf('\n########## A/B STUDY: %s ##########\n', upper(atype));
        try
            ab = run_ab_study(struct('agent_type', atype, ...
                'max_episodes',  opts.full_episodes, ...
                'eval_episodes', 100, ...
                'seed',          opts.seed));
            results.ab.(atype) = ab;
            results.completed{end+1} = atype;
        catch ME
            fprintf(2, 'A/B STUDY FAILED FOR %s: %s\n', upper(atype), ME.message);
            results.ab.(atype) = struct('failed', true, 'error', ME.message);
        end
        % Save after every architecture: a failure late in a multi-hour run must not
        % discard the architectures that already finished.
        save(outfile, 'results');
        fprintf('Progress saved: %s\n', outfile);
    end

    fprintf('\n########## FULL STUDY COMPLETE ##########\n');
    fprintf('Completed: %s\n', strjoin(results.completed, ', '));
end


function report_metrics(name, m, curve)
    fprintf('\n--- %s validation result ---\n', upper(name));
    fprintf('landing %.1f%%  crash %.1f%%  oob %.1f%%  stall %.1f%%  timeout %.1f%%\n', ...
        100*m.landing_rate, 100*m.crash_rate, 100*m.oob_rate, 100*m.stall_rate, 100*m.timeout_rate);
    fprintf('mean reward %.2f   mean impact %.2f m/s   mean vetoes %.1f\n', ...
        m.mean_reward, m.mean_impact_speed, m.mean_veto_count);
    fprintf('landing rate by phase: P1 %.0f%% (n=%d)  P2 %.0f%% (n=%d)  P3 %.0f%% (n=%d)\n', ...
        100*m.landing_rate_by_phase(1), m.n_by_phase(1), ...
        100*m.landing_rate_by_phase(2), m.n_by_phase(2), ...
        100*m.landing_rate_by_phase(3), m.n_by_phase(3));
    fprintf('mean impact by phase : P1 %.2f  P2 %.2f  P3 %.2f  (m/s)\n', ...
        m.impact_by_phase(1), m.impact_by_phase(2), m.impact_by_phase(3));

    % How close is the agent to the touchdown box? With zero landings this is the only
    % way to distinguish "converging, needs more episodes" from "not learning at all",
    % which is exactly the distinction that decides whether more compute is worth
    % spending. A ratio of 1.0 means arriving exactly at the limit.
    p = get_sim_params();
    fprintf('impact / touchdown limit : P1 %.2fx  P2 %.2fx  P3 %.2fx  (1.0 = landing)\n', ...
        m.impact_by_phase(1)/p.max_touchdown_dy, ...
        m.impact_by_phase(2)/p.max_touchdown_dy, ...
        m.impact_by_phase(3)/p.max_touchdown_dy);
    n = numel(curve); k = max(1, floor(n/4));
    fprintf('reward trend: first quarter %.2f -> last quarter %.2f\n', ...
        mean(curve(1:k)), mean(curve(end-k+1:end)));
end
