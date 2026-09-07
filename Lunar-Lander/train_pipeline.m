function results = train_pipeline(opts)
% TRAIN_PIPELINE Rebuilds the flight agent from nothing, end to end.
%
%   train_pipeline()                              % all four stages, ~2 h
%   train_pipeline(struct('stages', 3:4))         % resume at DAgger
%   train_pipeline(struct('n_candidates', 4))     % cheaper, noisier selection
%
% Agent and dataset .mat files are gitignored, so a fresh clone has none of them. This is
% the script that regenerates them. Every stage writes its artefact to the Lunar-Lander
% folder and can be resumed independently via `opts.stages`.
%
% --- THE FOUR STAGES ---
%   1 DEMONSTRATE  generate_demonstrations           -> demonstrations.mat
%     Fly the classical guidance law (core/scripted_pilot + core/braking_guidance) across
%     all four curriculum phases and record every (state, action) pair it produced.
%
%   2 CLONE        pretrain_actor_supervised x N     -> cloned_agent_4phase.mat
%     Fit N independently initialised TD3 actors and keep the one that FLIES best.
%     Selection is on closed-loop landing rate, never validation loss.
%
%   3 DAGGER       dagger_refine                     -> dagger_corpus.mat
%     Roll out the CLONE, label the states it reaches with what the EXPERT would do there.
%
%   4 SELECT       two-stage screen and verify       -> cloned_agent_4phase.mat
%     Re-clone from the aggregated corpus, screen wide and cheap, then RE-MEASURE a
%     shortlist on fresh seeds. Taking the max of one noisy screen is the winner's curse.
%
% README.md, "Why the pipeline has this shape", carries the measurement behind each of
% those choices.
%
% Stage 2 overwrites cloned_agent_4phase.mat with a seed clone; stage 4 overwrites it
% again with the finished agent. Running stages 3:4 alone therefore requires a stage-2
% clone to already be on disk.
%
% --- RUNTIME ---
% Measured on the machine this was developed on (Apple silicon, MATLAB R2025a, one
% process, no GPU): stage 1 ~20 min, stage 2 ~20 min, stages 3-4 ~75 min. About two hours
% end to end. Stage 3 dominates because each DAgger round flies complete powered descents.
%
% Inputs (optional struct):
%   opts.stages        - stages to run (default 1:4)
%   opts.n_candidates  - clones trained per selection stage (default 8)
%   opts.epochs        - supervised epochs per clone (default 25)
%   opts.dagger_rounds - DAgger iterations (default 6)
%   opts.eps_per_phase - DAgger rollouts per phase per round (default [2 2 3 6])
%   opts.screen_n      - episodes per phase when screening candidates (default 8)
%   opts.verify_n      - episodes per phase when verifying the shortlist (default 25)
%   opts.shortlist     - candidates carried from screen to verify (default 3)
%   opts.seed          - base RNG seed for candidate initialisation (default 2000)
%
% Outputs:
%   results - struct with .demos_summary, .screen, .verify, .final_rates as available
%
% See also GENERATE_DEMONSTRATIONS, PRETRAIN_ACTOR_SUPERVISED, DAGGER_REFINE,
%          EVALUATE_FINAL_AGENT, RUN_TRAINED_AGENT.

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'stages'),        opts.stages        = 1:4;        end
    if ~isfield(opts,'n_candidates'),  opts.n_candidates  = 8;          end
    if ~isfield(opts,'epochs'),        opts.epochs        = 25;         end
    if ~isfield(opts,'dagger_rounds'), opts.dagger_rounds = 6;          end
    if ~isfield(opts,'eps_per_phase'), opts.eps_per_phase = [2 2 3 6];  end
    if ~isfield(opts,'screen_n'),      opts.screen_n      = 8;          end
    if ~isfield(opts,'verify_n'),      opts.verify_n      = 25;         end
    if ~isfield(opts,'shortlist'),     opts.shortlist     = 3;          end
    if ~isfield(opts,'seed'),          opts.seed          = 2000;       end

    here = fileparts(mfilename('fullpath'));
    addpath(genpath(here));
    p = get_sim_params();

    demos_file = fullfile(here, 'demonstrations.mat');
    corpus_file = fullfile(here, 'dagger_corpus.mat');
    agent_file = fullfile(here, 'cloned_agent_4phase.mat');

    results = struct();
    t_all = tic;

    % --- STAGE 1: DEMONSTRATE ---
    if ismember(1, opts.stages)
        banner('STAGE 1/4  DEMONSTRATE - recording the classical pilot');
        % generate_demonstrations writes demonstrations.mat itself.
        demos = generate_demonstrations();
        results.demos_summary = summarise(demos);
    end

    % --- STAGE 2: CLONE ---
    if ismember(2, opts.stages)
        banner('STAGE 2/4  CLONE - fitting candidate actors to the demonstrations');
        demos = require_mat(demos_file, 'demos', ...
            'Run stage 1 first: train_pipeline(struct(''stages'', 1)).');
        best = select_clone(demos, p, opts, opts.seed);
        agent = best.agent;
        save(agent_file, 'agent');
        fprintf('\nSaved seed clone -> %s\n', agent_file);
        results.seed_clone = rmfield(best, 'agent');
    end

    % --- STAGE 3: DAGGER ---
    if ismember(3, opts.stages)
        banner('STAGE 3/4  DAGGER - labelling the states the clone actually visits');
        demos = require_mat(demos_file, 'demos', ...
            'Run stage 1 first: train_pipeline(struct(''stages'', 1)).');
        seed = require_mat(agent_file, 'agent', ...
            'Run stage 2 first: train_pipeline(struct(''stages'', 2)).');

        [~, history, aggregated] = dagger_refine(seed, demos, p, struct( ...
            'iterations',    opts.dagger_rounds, ...
            'eps_per_phase', opts.eps_per_phase, ...
            'epochs',        opts.epochs));

        % Save the CORPUS, not the final round's weights. The aggregated dataset is what
        % DAgger durably produces; any single round's network is one noisy draw from it,
        % which is why stage 4 re-selects rather than trusting the last fit.
        demos = aggregated;
        save(corpus_file, 'demos', '-v7.3');
        fprintf('\nSaved aggregated corpus -> %s\n', corpus_file);
        results.dagger_history = history;
    end

    % --- STAGE 4: SELECT ---
    if ismember(4, opts.stages)
        banner('STAGE 4/4  SELECT - screen wide, then verify the shortlist');
        corpus = require_mat(corpus_file, 'demos', ...
            'Run stage 3 first: train_pipeline(struct(''stages'', 3)).');
        best = select_clone(corpus, p, opts, opts.seed + 1000, true);
        agent = best.agent;
        save(agent_file, 'agent');
        fprintf('\nSaved final agent -> %s\n', agent_file);
        results.screen = best.screen;
        results.verify = best.verify;
        results.final_rates = best.rates;
    end

    fprintf('\nPipeline finished in %.1f min.\n', toc(t_all)/60);
    fprintf('Verify with evaluate_final_agent, then watch it with run_trained_agent(''orbit'').\n');
end


function best = select_clone(demos, p, opts, base_seed, do_verify)
% Train opts.n_candidates clones, screen them cheaply, optionally re-measure a shortlist.
    if nargin < 5, do_verify = false; end

    cands  = cell(1, opts.n_candidates);
    screen = zeros(opts.n_candidates, numel(p.phase_max_steps));

    fprintf('%8s %10s %8s %8s %8s %8s %9s\n', ...
        'cand', 'RMSEval', 'P1', 'P2', 'P3', 'P4', 'mean');
    for k = 1:opts.n_candidates
        % Each candidate gets its own seed so the spread below reflects initialisation,
        % not a shared random stream.
        rng(base_seed + k);
        env = LunarLanderEnv('DenseBaseline', 'on');
        a = build_agent('td3', getObservationInfo(env), getActionInfo(env), p.agent_dt);
        [a, info] = pretrain_actor_supervised(a, demos, p, ...
            struct('max_epochs', opts.epochs, 'verbose', false));
        cands{k} = a;
        screen(k,:) = phase_landing_rates(a, p, ...
                          struct('n_episodes', opts.screen_n, 'seed', 111)).rate;
        fprintf('%8d %10.4f %7.0f%% %7.0f%% %7.0f%% %7.0f%% %8.0f%%\n', k, info.rmse_val, ...
            100*screen(k,1), 100*screen(k,2), 100*screen(k,3), 100*screen(k,4), ...
            100*mean(screen(k,:)));
    end

    if ~do_verify
        [~, k] = max(mean(screen, 2));
        best = struct('agent', cands{k}, 'index', k, 'screen', screen, ...
                      'verify', [], 'rates', screen(k,:));
        fprintf('\nBEST candidate %d (screened at n=%d only)\n', k, opts.screen_n);
        return;
    end

    % Re-measure the shortlist on a DIFFERENT seed. Selecting on the screening numbers
    % themselves would pick whichever candidate got the friendliest eight initial
    % conditions, not the best policy.
    [~, order] = sort(mean(screen, 2), 'descend');
    short = order(1:min(opts.shortlist, numel(order)))';
    fprintf('\n  shortlist: candidates %s -> re-measuring at n=%d\n', ...
        mat2str(short), opts.verify_n);

    verify = nan(size(screen));
    best_mean = -1; best_k = short(1);
    for k = short
        verify(k,:) = phase_landing_rates(cands{k}, p, ...
                          struct('n_episodes', opts.verify_n, 'seed', 777)).rate;
        fprintf('  verify %d: P1 %3.0f%% P2 %3.0f%% P3 %3.0f%% P4 %3.0f%% | mean %3.0f%%\n', ...
            k, 100*verify(k,1), 100*verify(k,2), 100*verify(k,3), 100*verify(k,4), ...
            100*mean(verify(k,:)));
        if mean(verify(k,:)) > best_mean
            best_mean = mean(verify(k,:));
            best_k = k;
        end
    end

    fprintf('\nSELECTED candidate %d (verified n=%d): mean %.0f%%\n', ...
        best_k, opts.verify_n, 100*best_mean);
    best = struct('agent', cands{best_k}, 'index', best_k, 'screen', screen, ...
                  'verify', verify, 'rates', verify(best_k,:));
end



function s = summarise(demos)
    n  = arrayfun(@(e) size(e.actions,2), demos.episodes);
    ph = [demos.episodes.phase];
    s = struct('episodes', numel(n), 'transitions', sum(n));
    fprintf('\n  dataset: %d episodes, %d transitions\n', s.episodes, s.transitions);
    for k = 1:max(ph)
        fprintf('    P%d: %3d episodes, %7d transitions (%4.1f%%)\n', ...
            k, nnz(ph==k), sum(n(ph==k)), 100*sum(n(ph==k))/sum(n));
    end
end


function v = require_mat(file, var, remedy)
    if ~isfile(file)
        error('trainPipeline:MissingArtefact', ...
            'Required file not found: %s\n%s', file, remedy);
    end
    loaded = load(file, var);
    v = loaded.(var);
end


function banner(text)
    fprintf('\n%s\n%s\n%s\n', repmat('=', 1, 78), text, repmat('=', 1, 78));
end
