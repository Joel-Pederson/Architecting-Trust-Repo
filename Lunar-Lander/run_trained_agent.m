function ep = run_trained_agent(use_sidecar, opts)
% RUN_TRAINED_AGENT Visual playback of the final trained agent.
%
%   run_trained_agent()                             % find a landing, sidecar ON
%   run_trained_agent(false)                        % find a landing, sidecar OFF
%   run_trained_agent(true, struct('show','any'))   % show the next episode, whatever it does
%   run_trained_agent(true, struct('phase',3))      % restrict to the 2500 m descent
%
% Rolls episodes until it finds one matching `show`, then animates that one and reports
% how many attempts it took. The attempt count is printed deliberately: this agent lands
% about 80% of the time, and hiding the failures would misrepresent it.
%
% --- WHY THIS DOES NOT CALL main_simulation ---
% main_simulation runs to params.max_sim_steps, which is 900 s. The agent was TRAINED
% under params.max_agent_steps, which is 300 s. Playing it back with three times the
% clock it ever saw is not the same experiment: a run that would have been scored a
% timeout at 300 s instead spends 600 more seconds hovering, which looks like a hang
% rather than the failure mode it is. Rolling the episode here keeps playback and
% training on the same footing.
%
% --- WHAT THIS AGENT IS ---
% A neural policy CLONED from the classical guidance law in core/scripted_pilot.m, not an
% agent that discovered the task by exploration. Four RL architectures (DDPG, TD3, SAC,
% PPO) at 1200 episodes each produced ZERO landings from scratch, in three distinct
% failure modes. The task is not the problem - scripted_pilot lands 100/100/98.3% through
% this same action interface. Undirected exploration simply never reaches a success region
% that requires a coordinated descent, lateral null and square-up.
%
% --- MEASURED PERFORMANCE (40 greedy episodes, guardian ON) ---
%
%   landing 80.0%   crash 0.0%   timeout 10.0%
%   mean impact 0.40 m/s (touchdown limit 1.0)   mean reward +5.26
%   by phase: P1 45%   P2 100%   P3 86%
%
% Every failure is a TIMEOUT, not a crash: the agent flies a competent descent and then
% declines to commit to touchdown. That is imitation-learning distribution shift - the
% terminal manoeuvre is a small fraction of the training samples, so accumulated errors
% leave the agent slightly off the expert's state distribution exactly where precision
% matters. It is also a benign failure mode, which is worth noting for the safety case.
%
% --- MODEL SELECTION ---
% Chosen from 8 candidates by CLOSED-LOOP LANDING RATE, not validation loss. Across those
% candidates the correlation between validation RMSE and landing rate was -0.021. An epoch
% sweep had RMSE falling monotonically 0.184 -> 0.107 while landing rate bounced
% 43/10/57/47/3/20 percent, so selecting on regression error picks a policy that hovers.
%
% Inputs:
%   use_sidecar - (Optional) logical, default true
%   opts        - (Optional) struct:
%                   .show       'landing' (default) | 'any'
%                   .phase      [] for the mixed curriculum (default), or 1 | 2 | 3
%                   .max_tries  default 25
%                   .agent_file default 'cloned_agent_best.mat'
%                   .animate    default true
%
% Outputs:
%   ep - the rolled episode (telemetry, outcome, veto counts)

    if nargin < 1 || isempty(use_sidecar), use_sidecar = true; end
    if nargin < 2, opts = struct(); end
    if ~isfield(opts,'show'),       opts.show       = 'landing'; end
    if ~isfield(opts,'phase'),      opts.phase      = [];        end
    if ~isfield(opts,'max_tries'),  opts.max_tries  = 25;        end
    if ~isfield(opts,'animate'),    opts.animate    = true;      end
    if ~isfield(opts,'agent_file'), opts.agent_file = 'cloned_agent_best.mat'; end

    here = fileparts(mfilename('fullpath'));
    addpath(genpath(here));

    resolved = opts.agent_file;
    if ~isfile(resolved), resolved = fullfile(here, opts.agent_file); end
    if ~isfile(resolved)
        error('runTrainedAgent:NoAgent', ...
            ['Agent file not found: %s\n' ...
             'Agent .mat files are gitignored, so a fresh clone will not have one. ' ...
             'Regenerate with generate_demonstrations() then pretrain_actor_supervised().'], ...
            opts.agent_file);
    end

    data = load(resolved, 'agent');
    agent = data.agent;
    p = get_sim_params();

    if use_sidecar, gm = 'on'; else, gm = 'off'; end
    env = LunarLanderEnv('DenseBaseline', gm);
    if ~isempty(opts.phase)
        env.CurriculumWeights = double((1:3) == opts.phase);
    end

    fprintf('\nAgent: %s   sidecar %s\n', opts.agent_file, upper(gm));

    % --- Search for an episode matching `show` ---
    outcomes = {};
    ep = [];
    for k = 1:opts.max_tries
        candidate = rollout_episode(env, agent);
        outcomes{end+1} = candidate.outcome; %#ok<AGROW>
        if strcmp(opts.show, 'any') || strcmp(candidate.outcome, 'landed')
            ep = candidate;
            fprintf('Found a %s on attempt %d of %d.\n', candidate.outcome, k, opts.max_tries);
            break;
        end
    end

    if isempty(ep)
        % Report honestly rather than animating nothing.
        fprintf(2, ['No landing in %d attempts (outcomes: %s).\n' ...
                    'This agent lands ~80%% of the time, so this is unlucky or the ' ...
                    'agent file has changed. Try run_trained_agent(%d, ' ...
                    'struct(''show'',''any'')) to watch a failure instead.\n'], ...
                opts.max_tries, strjoin(unique(outcomes), ', '), use_sidecar);
        return;
    end

    fprintf('  outcome %s | touchdown dy %+.2f m/s, dx %+.2f m/s, theta %+.3f rad\n', ...
        ep.outcome, ep.touchdown_dy, ep.touchdown_dx, ep.touchdown_theta);
    fprintf('  duration %.1f s | sidecar engagements %d\n', ep.t(end), ep.veto_count);
    if k > 1
        fprintf('  (%d earlier attempt(s) this run: %s)\n', k-1, ...
            strjoin(outcomes(1:k-1), ', '));
    end

    if opts.animate
        animate_lunar_lander(ep.t, ep.states(1,:), ep.states(2,:), ep.states(4,:), ...
            ep.states(5,:), ep.controls(1,:), ep.states(7,:), ep.veto, p);
    end
end
