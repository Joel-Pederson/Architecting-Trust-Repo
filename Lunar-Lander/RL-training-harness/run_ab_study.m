function results = run_ab_study(opts)
% RUN_AB_STUDY Trains and evaluates the Safety Sidecar A/B experiment.
%
% This is the experiment that turns the Architecting Trust framework from an assertion
% into a measurement. It trains two agents that differ in exactly one respect - whether
% the Developmental Guardian filtered their commands during training - and then
% evaluates each of them both with and without the Operational Sidecar attached.
%
% The resulting 2x2 answers four distinct questions:
%
%                        | Eval guardian OFF        | Eval guardian ON
%   ----------------------+--------------------------+---------------------------------
%   Trained guardian OFF | Baseline. What an        | Retrofit. How many crashes does
%                        | unprotected RL agent     | the Operational Sidecar prevent
%                        | actually does.           | on a policy that never saw it?
%   ----------------------+--------------------------+---------------------------------
%   Trained guardian ON  | Crutch test. Did the     | Full framework. The configuration
%                        | agent learn to fly, or   | the paper proposes: safety across
%                        | to lean on the barrier?  | the whole lifecycle.
%
% The "crutch test" cell is the one worth watching. If a guardian-trained agent collapses
% when the guardian is removed, the Developmental Guardian is masking the reward signal
% rather than shaping it, and that is a finding about the architecture - not a bug.
%
% Inputs (all optional, passed as a struct):
%   opts.max_episodes  - training episodes per arm (default 5000)
%   opts.eval_episodes - evaluation episodes per cell (default 100)
%   opts.agent_type    - default 'ddpg'
%   opts.reward_scheme - default 'DenseBaseline'
%   opts.seed          - RNG seed for reproducible evaluation (default 42)
%
% Outputs:
%   results - struct with .agents, .cells (2x2 metrics), and .summary table

    if nargin < 1, opts = struct(); end
    if ~isfield(opts, 'max_episodes'),  opts.max_episodes  = 5000;            end
    if ~isfield(opts, 'eval_episodes'), opts.eval_episodes = 100;             end
    if ~isfield(opts, 'agent_type'),    opts.agent_type    = 'ddpg';          end
    if ~isfield(opts, 'reward_scheme'), opts.reward_scheme = 'DenseBaseline'; end
    if ~isfield(opts, 'seed'),          opts.seed          = 42;              end
    if ~isfield(opts, 'use_parallel'),  opts.use_parallel  = true;            end

    currentFolder = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(currentFolder, '..');
    addpath(genpath(repoRoot));

    train_modes = {'off', 'on'};
    eval_modes  = {'off', 'on'};

    agents = struct();
    stats  = struct();

    % --- 1. TRAIN BOTH ARMS ---
    for i = 1:numel(train_modes)
        mode = train_modes{i};
        fprintf('\n=== TRAINING ARM: Developmental Guardian %s ===\n', upper(mode));

        env = LunarLanderEnv(opts.reward_scheme, mode);
        obsInfo = getObservationInfo(env);
        actInfo = getActionInfo(env);

        agent = build_agent(opts.agent_type, obsInfo, actInfo, env.params.agent_dt);

        trainOpts = get_training_options(opts.agent_type, opts.max_episodes);
        trainOpts.Plots = 'none';
        trainOpts.Verbose = true;
        trainOpts.UseParallel = opts.use_parallel;

        s = train(agent, env, trainOpts);

        agents.(['guardian_' mode]) = agent;
        stats.(['guardian_' mode])  = s;

        save(fullfile(repoRoot, ...
            sprintf('ab_agent_%s_guardian_%s.mat', lower(opts.agent_type), mode)), 'agent');
    end

    % --- 2. EVALUATE THE 2x2 ---
    cells = struct();
    rows = {};

    for i = 1:numel(train_modes)
        for j = 1:numel(eval_modes)
            tm = train_modes{i};
            em = eval_modes{j};

            fprintf('\n--- EVALUATING: trained=%s  evaluated=%s ---\n', tm, em);

            % Same seed for every cell so all four face an identical set of initial
            % conditions. Without this, differences in landing rate are partly just
            % differences in which scenarios each cell happened to draw.
            rng(opts.seed);

            eval_env = LunarLanderEnv(opts.reward_scheme, em);
            m = evaluate_policy(eval_env, agents.(['guardian_' tm]), opts.eval_episodes);

            key = sprintf('train_%s_eval_%s', tm, em);
            cells.(key) = m;

            % Every outcome is reported. Listing only landed/crashed/timeout leaves
            % out-of-bounds and stalled episodes invisible, so the rates silently fail
            % to sum to 1 and a reader cannot tell what happened to the missing runs.
            rows(end+1, :) = {tm, em, m.landing_rate, m.crash_rate, m.oob_rate, ...
                              m.stall_rate, m.timeout_rate, m.mean_reward, ...
                              m.mean_veto_count, m.mean_impact_speed}; %#ok<AGROW>
        end
    end

    summary = cell2table(rows, 'VariableNames', ...
        {'TrainGuardian', 'EvalGuardian', 'LandingRate', 'CrashRate', 'OobRate', ...
         'StallRate', 'TimeoutRate', 'MeanReward', 'MeanVetoCount', 'MeanImpactSpeed'});

    disp(' ');
    disp('=== SAFETY SIDECAR A/B STUDY ===');
    disp(summary);

    results = struct('agents', agents, 'train_stats', stats, 'cells', cells, ...
                     'summary', summary, 'opts', opts);

    outfile = fullfile(repoRoot, sprintf('ab_study_results_%s.mat', lower(opts.agent_type)));
    save(outfile, 'results');
    fprintf('Saved: %s\n', outfile);
end


% evaluate_policy lives in its own file so Stage 1 and Stage 2 measure identically.
