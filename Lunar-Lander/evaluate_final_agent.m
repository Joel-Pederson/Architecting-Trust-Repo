function T = evaluate_final_agent(opts)
% EVALUATE_FINAL_AGENT The headline result table: every phase, guardian on and off.
%
%   evaluate_final_agent()                              % 30 episodes per cell
%   evaluate_final_agent(struct('n_episodes', 10))      % quicker, noisier
%
% Flies the trained agent through all four curriculum phases twice - once with the Safety
% Sidecar attached and once with it detached - and reports landing rate, mean touchdown
% impact and WORST touchdown impact for each cell.
%
% --- WHY THE WORST CASE AND NOT JUST THE MEAN ---
% A safety argument is about the tail. A mean impact of 0.2 m/s says nothing if one
% episode in thirty arrives at 4 m/s, so the maximum is reported alongside and is the
% number to read against the touchdown limit.
%
% --- WHY BOTH GUARDIAN ARMS ---
% Two distinct claims, and they need opposite evidence:
%   NON-INTRUSIVENESS  a competent controller should be unaffected by the barrier. Equal
%                      landing rates with it on and off is the result that shows this.
%   NECESSITY          the barrier has to matter when the controller is WRONG. This table
%                      cannot show that, by construction - the agent here is healthy. See
%                      demo_agent_rescue and run_fault_injection_study for that half.
%
% --- REFERENCE RESULT ---
% The agent produced by train_pipeline scored, at n=30 per cell:
%
%   phase | GUARDIAN ON              | GUARDIAN OFF
%         |  land%  impact  worst    |  land%  impact  worst
%       1 |   100%    0.24    0.26   |   100%    0.26    0.35
%       2 |   100%    0.21    0.21   |   100%    0.30    0.38
%       3 |   100%    0.19    0.20   |   100%    0.18    0.20
%       4 |   100%    0.46    0.50   |   100%    0.54    0.60
%
% 240 episodes, zero failures, worst impact 0.60 m/s against a 1.0 m/s limit - including
% 60 complete powered descents from 15.2 km and 1697 m/s across 550 km of downrange.
% Retraining from scratch will not reproduce those digits exactly: candidate selection
% draws different initialisations. The landing rates should reproduce; the impacts drift.
%
% Inputs (optional struct):
%   opts.n_episodes - episodes per cell (default 30)
%   opts.agent_file - default 'cloned_agent_4phase.mat'
%   opts.seed       - default 777, so both arms face identical initial conditions
%
% Outputs:
%   T - table, one row per phase
%
% See also TRAIN_PIPELINE, RUN_TRAINED_AGENT, RUN_FAULT_INJECTION_STUDY.

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'n_episodes'), opts.n_episodes = 30;  end
    if ~isfield(opts,'seed'),       opts.seed       = 777; end
    if ~isfield(opts,'agent_file'), opts.agent_file = 'cloned_agent_4phase.mat'; end

    here = fileparts(mfilename('fullpath'));
    addpath(genpath(here));
    p = get_sim_params();

    resolved = opts.agent_file;
    if ~isfile(resolved), resolved = fullfile(here, opts.agent_file); end
    if ~isfile(resolved)
        error('evaluateFinalAgent:NoAgent', ...
            ['Agent file not found: %s\n' ...
             'Agent .mat files are gitignored, so a fresh clone has none. ' ...
             'Build one with train_pipeline().'], opts.agent_file);
    end
    loaded = load(resolved, 'agent');
    agent = loaded.agent;
    if isprop(agent, 'UseExplorationPolicy'), agent.UseExplorationPolicy = false; end

    n_phases = numel(p.phase_max_steps);
    fprintf('\n%s\n%d episodes per phase, guardian ON and OFF\n\n', resolved, opts.n_episodes);
    fprintf('%6s | %-26s | %-26s\n', 'phase', 'GUARDIAN ON', 'GUARDIAN OFF');
    fprintf('%6s | %7s %8s %8s | %7s %8s %8s\n', ...
        '', 'land%', 'impact', 'worst', 'land%', 'impact', 'worst');

    % Same seed in both arms, so the guardian is the ONLY difference between them.
    base = struct('n_episodes', opts.n_episodes, 'seed', opts.seed);
    on  = phase_landing_rates(agent, p, setfield(base, 'guardian', 'on'));  %#ok<SFLD>
    off = phase_landing_rates(agent, p, setfield(base, 'guardian', 'off')); %#ok<SFLD>

    rows = zeros(n_phases, 6);
    for ph = 1:n_phases
        rows(ph,:) = [on.rate(ph), on.impact(ph), on.worst(ph), ...
                      off.rate(ph), off.impact(ph), off.worst(ph)];
        fprintf('%6d | %6.0f%% %8.2f %8.2f | %6.0f%% %8.2f %8.2f\n', ph, ...
            100*on.rate(ph), on.impact(ph), on.worst(ph), ...
            100*off.rate(ph), off.impact(ph), off.worst(ph));
    end
    fprintf('%6s | %6.0f%%                  | %6.0f%%\n', 'mean', ...
        100*mean(rows(:,1)), 100*mean(rows(:,4)));
    fprintf('\ntouchdown limits: |dy| <= %.1f m/s, |dx| <= %.1f m/s, |theta| <= %.2f rad\n', ...
        p.max_touchdown_dy, p.max_touchdown_dx, p.max_touchdown_tilt);

    T = array2table(rows, 'VariableNames', ...
        {'LandRate_On','Impact_On','WorstImpact_On', ...
         'LandRate_Off','Impact_Off','WorstImpact_Off'});
    T.Phase = (1:n_phases)';
    T = movevars(T, 'Phase', 'Before', 1);
end
