function [idx, label] = select_phase(env, scenario)
% SELECT_PHASE Restricts an environment to one curriculum phase, by name or by index.
%
%   select_phase(env, 'orbit')      % powered descent only
%   select_phase(env, 3)            % terminal descent only
%   [idx, label] = select_phase(env, 'approach')
%
% Nine call sites each wrote this one-hot themselves and two hardcoded `(1:3)`, so adding
% the powered descent silently made it unselectable in run_fault_injection_study - the
% script the paper's central result comes from. The phase count now has one source and is
% read from params, not written down. Accepts anything phase_from_name accepts.
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
