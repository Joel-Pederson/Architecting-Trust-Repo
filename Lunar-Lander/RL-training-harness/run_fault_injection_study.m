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
% Inputs (optional struct):
%   opts.n_episodes - episodes per cell (default 25)
%   opts.phases     - curriculum phases to sweep (default 1:3)
%   opts.seed       - default 101
%
% Outputs:
%   results - struct array with one entry per (fault, magnitude, phase, guardian) cell

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'n_episodes'), opts.n_episodes = 25;  end
    if ~isfield(opts,'phases'),     opts.phases     = 1:3; end
    if ~isfield(opts,'seed'),       opts.seed       = 101; end

    here = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(here, '..');
    addpath(genpath(repoRoot));
    p = get_sim_params();

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
                a = one_cell(p, ph, 'off', ft, mg, opts.n_episodes, opts.seed);
                b = one_cell(p, ph, 'on',  ft, mg, opts.n_episodes, opts.seed);
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

    outfile = fullfile(repoRoot, 'fault_injection_results.mat');
    save(outfile, 'results');
    fprintf('\nSaved: %s\n', outfile);
end


function m = one_cell(p, phase, guardian, fault, mag, n, seed)
    env = LunarLanderEnv('DenseBaseline', guardian);
    env.CurriculumWeights = double((1:3) == phase);
    rng(seed);   % identical initial conditions for the on/off pair
    landed = 0; impacts = []; vetoes = zeros(1,n);
    for ep = 1:n
        reset(env);
        hist = {};
        for i = 1:p.max_agent_steps
            s_true = env.State;
            s_seen = corrupt(s_true, fault, mag, hist);
            if strcmp(fault,'delay'), hist{end+1} = s_true; end %#ok<AGROW>
            u = scripted_pilot(s_seen, p);
            if strcmp(fault,'thrust_loss'), u(1) = u(1) * (1 - mag); end
            % Invert the environment's action map. Mass comes from TRUE state: a
            % fuel-gauge fault is a different experiment.
            a = command_to_action(u, s_true, p);
            [~,~,done] = step(env, a);
            if done, break; end
        end
        landed = landed + strcmp(env.Outcome,'landed');
        if any(strcmp(env.Outcome, {'landed','crashed'}))
            impacts(end+1) = sqrt(env.State(3)^2 + env.State(4)^2); %#ok<AGROW>
        end
        vetoes(ep) = env.VetoCount;
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


function s = corrupt(s_true, fault, mag, hist)
    s = s_true;
    switch fault
        case 'alt_bias', s(2) = s_true(2) + mag;
        case 'vel_bias', s(4) = s_true(4) * (1 - mag);
        case 'delay'
            if ~isempty(hist)
                s = hist{max(1, numel(hist) - round(mag))};
            end
        case 'thrust_loss'   % applied to the command, not the state
        otherwise
            error('runFaultInjectionStudy:UnknownFault', 'Unknown fault: %s', fault);
    end
end
