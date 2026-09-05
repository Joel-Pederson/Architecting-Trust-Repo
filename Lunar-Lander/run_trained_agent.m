function run_trained_agent(use_sidecar, agent_file)
% RUN_TRAINED_AGENT Visual playback of the final trained agent.
%
% Single entry point for watching the agent fly, with or without the Operational Sidecar
% attached. Wraps main_simulation so the agent file and mode string do not have to be
% remembered.
%
%   run_trained_agent()        % sidecar ON  (default)
%   run_trained_agent(false)   % sidecar OFF - the unprotected baseline
%   run_trained_agent(true, 'my_other_agent.mat')
%
% Writes telemetry to core/Flight_Logs/telemetry_RL_AGENT_*.png and opens the animated
% visualiser.
%
% --- WHAT THIS AGENT IS ---
% A neural policy CLONED from the classical guidance law in core/scripted_pilot.m, not an
% agent that discovered the task by exploration. That distinction is deliberate and is
% documented here so results are not misread.
%
% Four RL architectures (DDPG, TD3, SAC, PPO) at 1200 episodes each produced ZERO landings
% from scratch, in three distinct failure modes: DDPG saturated its actor high (max climb,
% 62% timeouts), TD3 saturated low (engine off, 100% crash), SAC's critic diverged upward.
% The task is not the problem - scripted_pilot lands 100/100/98.3% of the curriculum
% through this same action interface. Undirected exploration simply never reaches a
% success region that requires a coordinated descent, lateral null and square-up.
%
% Behaviour cloning closes that gap. See generate_demonstrations.m and
% pretrain_actor_supervised.m.
%
% --- MEASURED PERFORMANCE (cloned_agent_best.mat, 40 greedy episodes, guardian ON) ---
%
%   landing 80.0%   crash 0.0%   timeout 10.0%
%   mean impact 0.40 m/s (touchdown limit 1.0)   mean reward +5.26
%   by phase: P1 45%   P2 100%   P3 86%
%
% Every failure is a timeout, not a crash: the agent flies a competent descent and then
% declines to commit to touchdown. That is the signature of imitation-learning
% distribution shift - the terminal manoeuvre is a small fraction of the training samples,
% so small accumulated errors leave the agent slightly off the expert's state distribution
% exactly where precision matters most.
%
% --- MODEL SELECTION NOTE ---
% This agent was chosen from 8 candidates by CLOSED-LOOP LANDING RATE, not by validation
% loss. Across those candidates the correlation between validation RMSE and landing rate
% was -0.021, i.e. none: an epoch sweep showed RMSE falling monotonically from 0.184 to
% 0.107 while landing rate bounced 43 / 10 / 57 / 47 / 3 / 20 percent. Selecting on
% regression error would have picked a policy that hovers.
%
% Inputs:
%   use_sidecar - (Optional) logical, default true
%   agent_file  - (Optional) path to a .mat containing a variable `agent`,
%                 default 'cloned_agent_best.mat'

    if nargin < 1 || isempty(use_sidecar), use_sidecar = true; end
    if nargin < 2 || isempty(agent_file),  agent_file  = 'cloned_agent_best.mat'; end

    here = fileparts(mfilename('fullpath'));
    addpath(genpath(here));

    resolved = agent_file;
    if ~isfile(resolved)
        resolved = fullfile(here, agent_file);
    end
    if ~isfile(resolved)
        error('runTrainedAgent:NoAgent', ...
            ['Agent file not found: %s\n' ...
             'Agent .mat files are gitignored, so a fresh clone of this repo will not ' ...
             'have one. Regenerate with:\n' ...
             '    demos = generate_demonstrations();\n' ...
             '    %% then clone an actor and save it as `agent` (see ' ...
             'pretrain_actor_supervised.m)'], agent_file);
    end

    fprintf('Playing back %s with sidecar %s\n', agent_file, string_onoff(use_sidecar));
    main_simulation('RL_AGENT', resolved, use_sidecar);
end


function s = string_onoff(tf)
    if tf, s = 'ON'; else, s = 'OFF'; end
end
