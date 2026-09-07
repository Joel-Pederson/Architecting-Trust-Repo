function results = run_fault_injection_study(opts)
% RUN_FAULT_INJECTION_STUDY The Safety Sidecar experiment.
%
% Measures what the Operational Sidecar recovers when the PRIMARY CONTROLLER IS FAULTY.
% This is the claim Architecting Trust actually makes: a runtime barrier bounds an
% autonomous system whose nominal controller has degraded, without needing to predict how
% it degraded. No learning is involved, so the result does not depend on an RL agent
% converging.
%
% --- WHY FAULTS AND NOT NOISE ---
% The first version of this experiment injected zero-mean Gaussian noise into the pilot's
% actions and found essentially nothing: landing rates stayed at ~100% with and without
% the guardian up to sigma = 0.5, because a PD loop re-deciding at 10 Hz rejects
% zero-mean disturbance by construction. Noise is the one fault a feedback controller
% handles natively, so it cannot discriminate. The faults below are SYSTEMATIC: the pilot
% cannot see them and cannot correct them.
%
% --- THE PERCEPTION BOUNDARY ---
% The faulty pilot reads corrupted state; the sidecar always reads TRUE state. That split
% is the Perception Gatekeeper boundary from the paper, made concrete: the barrier's
% authority rests on validated state that the nominal controller does not have.
%
% --- FAULT MODELS ---
%   alt_bias    - altimeter reads high, so the pilot brakes too late. Perception fault.
%   vel_bias    - descent rate under-read. Perception fault, harder because the pilot's
%                 own damping term is what is corrupted.
%   thrust_loss - engine delivers less than commanded. Actuator fault.
%   delay       - pilot acts on stale state. Included BECAUSE the sidecar handles it
%                 badly: this is the architecture's boundary condition, not a win.
%
% The fault models themselves live in core/apply_sensor_fault.m so they can be unit
% tested and reused; that file also carries 'alt_freeze', a harsher perception fault
% where the altimeter stops updating below a threshold rather than reading a constant
% offset.
%
% Inputs (optional struct):
%   opts.n_episodes - episodes per cell (default 25)
%   opts.phases     - curriculum phases to sweep (default 1:3)
%   opts.controller - 'pilot' (default) or 'agent'. The classical pilot is the default
%                     because this study's result does not depend on a network existing;
%                     'agent' runs the same grid against the frozen trained policy.
%   opts.agent_file - default 'cloned_agent_4phase.mat', used when controller is 'agent'
%   opts.seed       - default 101
%
% Outputs:
%   results - struct array with one entry per (fault, magnitude, phase, guardian) cell

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'n_episodes'), opts.n_episodes = 25;  end
    if ~isfield(opts,'phases'),     opts.phases     = 1:3; end
    if ~isfield(opts,'seed'),       opts.seed       = 101; end
    if ~isfield(opts,'controller'), opts.controller = 'pilot'; end
    if ~isfield(opts,'agent_file'), opts.agent_file = 'cloned_agent_4phase.mat'; end

    here = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(here, '..');
    addpath(genpath(repoRoot));
    p = get_sim_params();
    controller = resolve_controller(opts, repoRoot);

    spec = { 'alt_bias',    [0 5 10 20 40],      'altimeter high by %g m'; ...
             'vel_bias',    [0 0.3 0.5 0.7 0.9], 'descent rate under-read %g'; ...
             'thrust_loss', [0 0.1 0.2 0.3 0.4], 'engine down %g'; ...
             'delay',       [0 5 10 20 40],      'control delayed %g steps' };

    rows = {};
    for ph = opts.phases
        for f = 1:size(spec,1)
            ft = spec{f,1};
            fprintf('\n===== FAULT: %-12s  PHASE %d =====\n', upper(ft), ph);
            fprintf('%-30s | %6s %8s %8s | %6s %8s %8s %7s\n', ...
                'magnitude','land%','impact','worst','land%','impact','worst','vetoes');
            for mg = spec{f,2}
                a = one_cell(p, ph, 'off', ft, mg, opts.n_episodes, opts.seed, controller);
                b = one_cell(p, ph, 'on',  ft, mg, opts.n_episodes, opts.seed, controller);
                fprintf('%-30s | %5.0f%% %8.2f %8.2f | %5.0f%% %8.2f %8.2f %7.1f\n', ...
                    sprintf(spec{f,3}, mg), 100*a.land, a.impact, a.impact_max, ...
                    100*b.land, b.impact, b.impact_max, b.vetoes);
                rows(end+1,:) = {ft, mg, ph, a.land, a.impact, a.impact_max, ...
                                 b.land, b.impact, b.impact_max, b.vetoes}; %#ok<AGROW>
            end
        end
    end

    results = cell2table(rows, 'VariableNames', ...
        {'Fault','Magnitude','Phase','LandRate_Off','Impact_Off','WorstImpact_Off', ...
         'LandRate_On','Impact_On','WorstImpact_On','MeanVetoes_On'});

    outfile = fullfile(repoRoot, sprintf('fault_injection_results_%s.mat', opts.controller));
    save(outfile, 'results');
    fprintf('\nSaved: %s\n', outfile);
end


function m = one_cell(p, phase, guardian, fault, mag, n, seed, controller)
% One (fault, magnitude, phase, guardian) cell.
%
% The flight itself is delegated to fly_with_faults, which the Monte Carlo study also uses.
% This function's job is only to express an enumerated cell AS a draw: exactly one fault,
% present from the first step, on a nominal plant.
%
% Sharing the loop fixed two things this study had wrong on its own. It called
% scripted_pilot, the TERMINAL guidance law, which cannot fly a powered descent at all; and
% it capped every episode at params.max_agent_steps rather than the phase's own budget,
% which truncates a Phase 4 descent partway down and scores it as a timeout that never
% happened. Both were invisible while the sweep only ever ran phases 1-3.
    landed = 0; impacts = []; vetoes = zeros(1,n);

    for ep = 1:n
        draw = draw_for(fault, mag);
        % Seed varies per episode but is shared between the guardian arms by the caller,
        % so on and off face identical initial conditions.
        r = fly_with_faults(controller, draw, p, ...
                struct('phase', phase, 'guardian', guardian, 'seed', seed + ep));

        landed = landed + strcmp(r.outcome, 'landed');
        if isfinite(r.impact)
            impacts(end+1) = r.impact; %#ok<AGROW>
        end
        vetoes(ep) = r.vetoes;
    end

    m.land   = landed / n;
    m.impact = mean(impacts);   % NaN if nothing reached the ground, which is meaningful
    m.vetoes = mean(vetoes);

    % WORST CASE, not just the mean. A safety barrier is judged on the tail: "the average
    % impact halved" is a performance claim, while "no impact exceeded X" is a safety
    % claim, and only the second one supports a runtime-assurance argument. Reported for
    % both arms so the paper can state a bound rather than an improvement.
    if isempty(impacts)
        m.impact_max = NaN;
        m.impact_p90 = NaN;
    else
        m.impact_max = max(impacts);
        m.impact_p90 = prctile_local(impacts, 90);
    end
    m.n_grounded = numel(impacts);
end


function draw = draw_for(fault, mag)
% An enumerated cell as a draw: one fault, from step one, nominal plant.
    draw = struct('sensors', struct('type', {}, 'magnitude', {}, 'onset', {}), ...
                  'thrust', 1.0, 'thrust_onset', 0, 'dry_mass', [], 'ic_scale', 1.0);
    switch fault
        case 'none'
            % nothing to add
        case 'thrust_loss'
            draw.thrust = 1 - mag;
        otherwise
            draw.sensors(1) = struct('type', fault, 'magnitude', mag, 'onset', 0);
    end
end


function controller = resolve_controller(opts, repoRoot)
    if strcmp(opts.controller, 'pilot')
        controller = struct('kind', 'pilot');
        return;
    end
    f = opts.agent_file;
    if ~isfile(f), f = fullfile(repoRoot, opts.agent_file); end
    if ~isfile(f)
        error('runFaultInjectionStudy:NoAgent', ...
            ['Agent file not found: %s\nAgent .mat files are gitignored. Build one with ' ...
             'train_pipeline(), or run with the default controller ''pilot''.'], ...
            opts.agent_file);
    end
    d = load(f, 'agent');
    a = d.agent;
    if isprop(a, 'UseExplorationPolicy'), a.UseExplorationPolicy = false; end
    controller = struct('kind', 'agent', 'agent', a);
end


function v = prctile_local(x, pct)
% Linear-interpolated percentile. Local so the study does not require the Statistics
% Toolbox - the same dependency that had to be removed from the visualiser.
    x = sort(x(:));
    if isscalar(x), v = x; return; end
    idx = (pct/100) * (numel(x) - 1) + 1;
    lo = floor(idx); hi = ceil(idx);
    if lo == hi
        v = x(lo);
    else
        v = x(lo) + (idx - lo) * (x(hi) - x(lo));
    end
end
