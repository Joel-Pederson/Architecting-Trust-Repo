function [idx, label] = select_phase(env, scenario)
% SELECT_PHASE Restricts an environment to one curriculum phase, by name or by index.
%
%   select_phase(env, 'orbit')      % powered descent only
%   select_phase(env, 3)            % terminal descent only
%   [idx, label] = select_phase(env, 'approach')
%
% --- WHY THIS EXISTS ---
% Nine call sites had each written their own version of the same one-hot:
%
%     env.CurriculumWeights = double((1:numel(p.phase_max_steps)) == phase);
%
% and two of them had hardcoded the phase count instead of reading it:
%
%     env.CurriculumWeights = double((1:3) == phase);          % <- stale
%
% That is not a style problem. When the powered descent was added as a fourth phase,
% every hardcoded copy silently stopped being able to select it - and one of them was
% run_fault_injection_study, which is the script the paper's central result comes from.
% Asking for phase 4 there produced an all-zero weight vector, which the environment then
% normalised by its own sum. The phase count now has exactly one source, params, and is
% read rather than written down.
%
% Accepts everything core/phase_from_name accepts, so scenario names work here too and
% callers do not have to resolve them first.
%
% Inputs:
%   env      - LunarLanderEnv (handle object; mutated in place)
%   scenario - 'touchdown' | 'approach' | 'terminal' | 'orbit', or an index 1-4
%
% Outputs:
%   idx   - the resolved phase index
%   label - canonical human-readable description of that phase
%
% See also PHASE_FROM_NAME, LUNARLANDERENV, PHASE_LANDING_RATES.

    [idx, label] = phase_from_name(scenario);

    n_phases = numel(env.params.phase_max_steps);
    if idx > n_phases
        error('selectPhase:PhaseNotConfigured', ...
            ['Scenario "%s" resolves to phase %d, but params.phase_max_steps defines ' ...
             'only %d phases.'], char(string(scenario)), idx, n_phases);
    end

    env.CurriculumWeights = double((1:n_phases) == idx);
end
