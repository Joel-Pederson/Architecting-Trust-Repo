function demos = generate_demonstrations(opts)
% GENERATE_DEMONSTRATIONS Records expert trajectories from the classical pilot.
%
% Produces the dataset that seeds an off-policy agent's replay buffer, so learning does
% not depend on undirected exploration stumbling into the success region.
%
% --- WHY THIS EXISTS ---
% Four architectures at 1200 episodes each, plus two longer runs, produced ZERO landings
% under greedy evaluation. The reward landscape was verified correct on five separate
% properties, and the task is demonstrably solvable: scripted_pilot lands 100% / 100% /
% 100% of every curriculum phase through this same [-1,1] action interface. The gap is
% exploration - a landing requires a coordinated descent, lateral null and square-up, and
% random action sequences never produce one. Demonstrations put successful transitions
% into the buffer directly.
%
% --- RECORDING CONTRACT ---
% Two decisions here are load-bearing:
%
%   RAW STATES, not normalised observations. get_ai_observation hardcodes 1000/3000/100/
%   8200/300 as literals that no single source owns. A cached normalised dataset would be
%   silently invalidated by any edit to those constants, and nothing would fail loudly.
%   Store the 8x1 physics state and normalise at load time instead.
%
%   CLIPPED actions, not raw quotients. command_to_action clips to [-1,1], and the
%   clipped value is what the environment actually received. Cloning the unclipped
%   quotient would train the network toward commands the plant cannot execute, so the
%   targets would not match the trajectory that produced them.
%
% Inputs (optional struct):
%   opts.n_per_phase   - episodes attempted per curriculum phase. Scalar, or one entry
%                        per phase. Default [40 40 40 12]: a Phase 4 powered descent is
%                        ~8100 agent steps against ~450 for Phase 1, so equal episode
%                        counts would let it dominate the dataset by an order of
%                        magnitude and pull the fit away from the terminal regime, which
%                        is where the touchdown limits are actually decided.
%   opts.guardian      - 'off' (default) or 'on'. Off by default so the dataset captures
%                        the pilot's own behaviour rather than guardian-assisted
%                        trajectories, keeping it neutral for the guardian A/B.
%   opts.noise_frac    - fraction of episodes flown with action noise (default 0.5)
%   opts.noise_sigma   - std of that noise (default 0.08)
%   opts.seed          - default 202
%   opts.outfile       - default <repo>/demonstrations.mat; '' to skip saving
%
% Outputs:
%   demos - struct with .episodes (struct array), .meta, and summary counts

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'n_per_phase'), opts.n_per_phase = [40 40 40 12]; end
    if ~isfield(opts,'guardian'),    opts.guardian    = 'off'; end
    if ~isfield(opts,'noise_frac'),  opts.noise_frac  = 0.5;   end
    if ~isfield(opts,'noise_sigma'), opts.noise_sigma = 0.08;  end
    if ~isfield(opts,'seed'),        opts.seed        = 202;   end

    here = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(here, '..');
    addpath(genpath(repoRoot));
    if ~isfield(opts,'outfile')
        opts.outfile = fullfile(repoRoot, 'demonstrations.mat');
    end

    p = get_sim_params();
    rng(opts.seed);

    episodes = struct('states', {}, 'actions', {}, 'rewards', {}, ...
                      'outcome', {}, 'phase', {}, 'sigma', {});
    attempted = 0; kept = 0;

    n_phases = numel(p.phase_max_steps);
    per_phase = opts.n_per_phase;
    if isscalar(per_phase), per_phase = repmat(per_phase, 1, n_phases); end

    for phase = 1:n_phases
        env = LunarLanderEnv('DenseBaseline', opts.guardian);
        env.CurriculumWeights = double((1:n_phases) == phase);

        for ep = 1:per_phase(phase)
            % Noise on a fraction of episodes. A purely on-policy expert dataset lies on
            % a measure-zero manifold: the critic sees no state the expert would not have
            % visited, so it cannot learn what makes those actions good. Perturbed
            % episodes supply the off-manifold contrast.
            if rand() < opts.noise_frac
                sigma = opts.noise_sigma;
            else
                sigma = 0;
            end

            attempted = attempted + 1;
            e = fly_one(env, p, sigma);

            % Only successful trajectories. A crashed demonstration teaches the actor to
            % reproduce a crash, and the buffer already fills with failures during online
            % fine-tuning.
            if strcmp(e.outcome, 'landed')
                e.phase = phase;
                e.sigma = sigma;
                episodes(end+1) = e; %#ok<AGROW>
                kept = kept + 1;
            end
        end
        fprintf('  phase %d: %d/%d landed\n', phase, ...
            nnz([episodes.phase] == phase), per_phase(phase));
    end

    n_trans = sum(arrayfun(@(e) size(e.actions, 2), episodes));

    demos = struct();
    demos.episodes    = episodes;
    demos.n_episodes  = kept;
    demos.n_attempted = attempted;
    demos.n_transitions = n_trans;
    demos.meta = struct('guardian', opts.guardian, 'seed', opts.seed, ...
                        'noise_frac', opts.noise_frac, 'noise_sigma', opts.noise_sigma, ...
                        'harness_version', p.harness_version, ...
                        'generated', datetime('now'));

    fprintf('\nkept %d/%d episodes, %d transitions (guardian %s)\n', ...
        kept, attempted, n_trans, opts.guardian);

    if ~isempty(opts.outfile)
        save(opts.outfile, 'demos', '-v7.3');
        fprintf('Saved: %s\n', opts.outfile);
    end
end


function e = fly_one(env, p, sigma)
% One pilot episode. states is 8x(N+1) so every action has both its state and next state.
    reset(env);
    % Loop to the LONGEST phase budget, not the global default. The environment ends the
    % episode at its own per-phase cap, so this only has to be long enough not to truncate
    % first - and a Phase 4 powered descent takes ~8150 agent steps against the global
    % default of 3000, which silently cut every one of them short of touchdown.
    max_n = max(p.phase_max_steps);
    states  = zeros(8, max_n + 1);
    actions = zeros(2, max_n);
    rewards = zeros(1, max_n);

    states(:,1) = env.State;
    n = 0;
    for i = 1:max_n
        x = env.State;
        % braking_guidance is a SUPERSET of scripted_pilot: it flies the powered descent
        % and hands off to scripted_pilot itself once inside that controller's envelope,
        % so one call covers every phase and there is no dispatch to keep in sync.
        u = braking_guidance(x, p);
        a = command_to_action(u, x, p);
        if sigma > 0
            a = max(-1, min(a + sigma * randn(2,1), 1));
        end

        [~, r, done] = step(env, a);

        n = i;
        actions(:,i)  = a;
        rewards(i)    = r;
        states(:,i+1) = env.State;
        if done, break; end
    end

    e = struct();
    e.states  = states(:, 1:n+1);
    e.actions = actions(:, 1:n);
    e.rewards = rewards(1:n);
    e.outcome = env.Outcome;
    e.phase   = env.Phase;
    e.sigma   = sigma;
end
