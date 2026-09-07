function results = run_ood_study(opts)
% RUN_OOD_STUDY Monte Carlo over conditions the controller was never trained for.
%
%   run_ood_study()                                      % 400 draws, the trained agent
%   run_ood_study(struct('controller','pilot'))          % same sample, classical pilot
%   run_ood_study(struct('n_draws', 50, 'phases', 1))    % quick smoke run
%
% Flies the SAME sampled off-nominal conditions twice - barrier attached and detached -
% and reports what fraction of the sampled space the barrier keeps inside the touchdown
% gate. Each draw is flown from an identical initial condition in both arms, so the barrier
% is the only difference between them.
%
% --- WHAT THIS ANSWERS THAT THE ENUMERATED SWEEP DOES NOT ---
% run_fault_injection_study asks whether the barrier holds against four fault types at five
% chosen magnitudes, one at a time. This asks whether it holds against conditions nobody
% enumerated: faults combined, magnitudes continuous and past the swept range, onsets
% mid-descent, the plant itself off-nominal, and initial states outside the training box.
%
% That is the claim a runtime barrier actually makes. A fault-tolerant controller has to
% anticipate each failure mode; a barrier enforces a state-space property and should not
% need to. Testing only against a list concedes the difference.
%
% --- THE HEADLINE IS A BOUND, NOT A RATE ---
% Landing rate is reported, but the number that supports a safety argument is the fraction
% of draws whose touchdown speed stays inside the gate, and the worst case over the whole
% sample. "The average impact halved" is a performance claim. "No sampled condition
% exceeded X" is a safety claim, and only the second one is runtime assurance.
%
% --- READ THE LIMIT HONESTLY ---
% Coverage here is coverage of the sampling distribution in sample_ood_draw, which is a
% choice. This widens the enumeration problem rather than escaping it. See that function's
% header, and do not report these numbers as coverage of all possible faults.
%
% Inputs (optional struct):
%   opts.n_draws    - draws per phase (default 400; Phase 4 uses a quarter of this,
%                     because a powered descent is ~8,900 steps against Phase 1's ~450)
%   opts.phases     - phases to sample (default 1:4)
%   opts.controller - 'agent' (default) or 'pilot'
%   opts.agent_file - default 'cloned_agent_4phase.mat'
%   opts.seed       - base seed (default 20260907)
%   opts.verbose    - default true
%
% Outputs:
%   results - struct with .table (per phase), .draws (every sampled condition and its two
%             outcomes), and .summary
%
% See also SAMPLE_OOD_DRAW, FLY_WITH_FAULTS, RUN_FAULT_INJECTION_STUDY.

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'n_draws'),    opts.n_draws    = 400; end
    if ~isfield(opts,'phases'),     opts.phases     = 1:4; end
    if ~isfield(opts,'controller'), opts.controller = 'agent'; end
    if ~isfield(opts,'seed'),       opts.seed       = 20260907; end
    if ~isfield(opts,'verbose'),    opts.verbose    = true; end
    if ~isfield(opts,'agent_file'), opts.agent_file = 'cloned_agent_4phase.mat'; end

    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);
    addpath(genpath(root));
    p = get_sim_params();

    controller = build_controller(opts, root);
    gate = p.max_touchdown_dy;

    rows = {};
    all_draws = struct('phase', {}, 'label', {}, 'severity', {}, ...
                       'outcome_off', {}, 'impact_off', {}, ...
                       'outcome_on', {}, 'impact_on', {}, 'vetoes_on', {});

    for ph = opts.phases
        % Scale the sample down for long phases rather than naming one. A Phase 4 descent
        % runs ~8,900 agent steps against Phase 1's ~450, so sampling every phase equally
        % would spend most of the study's compute on the longest one. Derived from the
        % budgets so a fifth phase does not silently get the short-phase treatment.
        cost = p.phase_max_steps(ph) / min(p.phase_max_steps);
        n = max(25, round(opts.n_draws / max(1, sqrt(cost))));

        if opts.verbose
            fprintf('\n===== PHASE %d | %d draws | controller: %s =====\n', ...
                ph, n, opts.controller);
            fprintf('%5s %-52s %10s %10s\n', 'draw', 'condition', 'OFF', 'ON');
        end

        for k = 1:n
            % The draw and both flights share one seed, so the two arms face an identical
            % vehicle in an identical situation and differ only in the barrier.
            seed = opts.seed + 1000*ph + k;
            rng(seed);
            draw = sample_ood_draw(p, ph);

            off = fly_with_faults(controller, draw, p, ...
                    struct('phase', ph, 'guardian', 'off', 'seed', seed));
            on  = fly_with_faults(controller, draw, p, ...
                    struct('phase', ph, 'guardian', 'on',  'seed', seed));

            all_draws(end+1) = struct('phase', ph, 'label', draw.label, ...
                'severity', draw.severity, ...
                'outcome_off', off.outcome, 'impact_off', off.impact, ...
                'outcome_on',  on.outcome,  'impact_on',  on.impact, ...
                'vetoes_on',   on.vetoes); %#ok<AGROW>

            if opts.verbose && (mod(k, 25) == 1 || k == n)
                fprintf('%5d %-52s %10s %10s\n', k, truncate(draw.label, 52), ...
                    fmt(off), fmt(on));
            end
        end

        rows(end+1,:) = phase_row(ph, all_draws([all_draws.phase] == ph), gate); %#ok<AGROW>
    end

    results.table = cell2table(rows, 'VariableNames', ...
        {'Phase','Draws','LandRate_Off','LandRate_On', ...
         'WithinGate_Off','WithinGate_On','WorstImpact_Off','WorstImpact_On','MeanVetoes_On'});
    results.draws   = all_draws;
    results.summary = summarise(all_draws, gate, opts, p);

    if opts.verbose
        fprintf('\n');
        disp(results.table);
        print_summary(results.summary, gate);
    end

    outfile = fullfile(root, sprintf('ood_study_%s.mat', opts.controller));
    save(outfile, 'results');
    fprintf('\nSaved: %s\n', outfile);
end


function controller = build_controller(opts, root)
    switch opts.controller
        case 'pilot'
            controller = struct('kind', 'pilot');
        case 'agent'
            f = opts.agent_file;
            if ~isfile(f), f = fullfile(root, opts.agent_file); end
            if ~isfile(f)
                error('runOodStudy:NoAgent', ...
                    ['Agent file not found: %s\nAgent .mat files are gitignored, so a ' ...
                     'fresh clone has none. Build one with train_pipeline().'], ...
                     opts.agent_file);
            end
            d = load(f, 'agent');
            a = d.agent;
            if isprop(a, 'UseExplorationPolicy'), a.UseExplorationPolicy = false; end
            controller = struct('kind', 'agent', 'agent', a);
        otherwise
            error('runOodStudy:UnknownController', ...
                'opts.controller must be ''agent'' or ''pilot''.');
    end
end


function row = phase_row(ph, d, gate)
    off_land = mean(strcmp({d.outcome_off}, 'landed'));
    on_land  = mean(strcmp({d.outcome_on},  'landed'));
    row = {ph, numel(d), off_land, on_land, ...
           within_gate(d, gate, 'off'), within_gate(d, gate, 'on'), ...
           worst([d.impact_off]), worst([d.impact_on]), mean([d.vetoes_on])};
end


function frac = within_gate(d, gate, arm)
% Fraction of draws that reached the ground INSIDE the touchdown gate.
%
% Episodes that never touched down are counted as NOT within gate. That is the
% conservative reading and the right one: a vehicle still in the air when the clock expires
% has not been shown to be safe, it has been shown to be untested.
    if strcmp(arm, 'off'), imp = [d.impact_off]; else, imp = [d.impact_on]; end
    frac = sum(isfinite(imp) & imp <= gate) / numel(imp);
end


function w = worst(imp)
    imp = imp(isfinite(imp));
    if isempty(imp), w = NaN; else, w = max(imp); end
end


function s = summarise(d, gate, opts, p)
    s.n            = numel(d);
    s.controller   = opts.controller;
    s.gate         = gate;
    s.land_off     = mean(strcmp({d.outcome_off}, 'landed'));
    s.land_on      = mean(strcmp({d.outcome_on},  'landed'));
    s.gate_off     = within_gate(d, gate, 'off');
    s.gate_on      = within_gate(d, gate, 'on');
    s.worst_off    = worst([d.impact_off]);
    s.worst_on     = worst([d.impact_on]);
    s.max_severity = max([d.severity]);

    % Does the barrier degrade gracefully as insults stack, or fall off a cliff? A single
    % pooled number cannot answer that, and the answer is the more useful finding.
    s.by_severity = struct('severity', {}, 'n', {}, 'gate_off', {}, 'gate_on', {});
    for sev = 0:s.max_severity
        sel = d([d.severity] == sev);
        if isempty(sel), continue; end
        s.by_severity(end+1) = struct('severity', sev, 'n', numel(sel), ...
            'gate_off', within_gate(sel, gate, 'off'), ...
            'gate_on',  within_gate(sel, gate, 'on'));
    end

    % The draws the barrier did NOT bound. This is the study's real product: the
    % architecture's boundary, discovered rather than assumed.
    esc = d(~(isfinite([d.impact_on]) & [d.impact_on] <= gate));
    s.escapes = esc;
    s.n_escapes = numel(esc);
    s.params_dry_mass = p.dry_mass;
end


function print_summary(s, gate)
    fprintf('\n--- SAMPLED SPACE, controller: %s, %d draws ---\n', s.controller, s.n);
    fprintf('  landing rate         off %5.1f%%   on %5.1f%%\n', 100*s.land_off, 100*s.land_on);
    fprintf('  within %.1f m/s gate  off %5.1f%%   on %5.1f%%\n', gate, 100*s.gate_off, 100*s.gate_on);
    fprintf('  worst impact         off %7.2f   on %7.2f  m/s\n', s.worst_off, s.worst_on);
    fprintf('\n  by number of simultaneous faults:\n');
    fprintf('  %8s %6s %12s %12s\n', 'faults', 'n', 'gate OFF', 'gate ON');
    for k = 1:numel(s.by_severity)
        b = s.by_severity(k);
        fprintf('  %8d %6d %11.0f%% %11.0f%%\n', b.severity, b.n, 100*b.gate_off, 100*b.gate_on);
    end
    fprintf('\n  draws NOT bounded by the barrier: %d of %d (%.1f%%)\n', ...
        s.n_escapes, s.n, 100*s.n_escapes/max(s.n,1));
    for k = 1:min(8, s.n_escapes)
        e = s.escapes(k);
        fprintf('    P%d  %-52s  %s %.2f m/s\n', e.phase, truncate(e.label, 52), ...
            e.outcome_on, e.impact_on);
    end
end


function out = fmt(ep)
    if isfinite(ep.impact)
        out = sprintf('%s %.2f', ep.outcome(1), ep.impact);
    else
        out = sprintf('%s   -', ep.outcome(1));
    end
end


function s = truncate(s, n)
    if numel(s) > n, s = [s(1:n-3) '...']; end
end
