function ep = run_trained_agent(scenario, opts)
% RUN_TRAINED_AGENT Visual playback of the final trained agent.
%
%   run_trained_agent()                     % any scenario, sidecar on
%   run_trained_agent('orbit')              % the full powered descent from orbit
%   run_trained_agent('terminal')           % 2.5 km terminal descent
%   run_trained_agent('approach')           % 500 m glide slope
%   run_trained_agent('touchdown')          % 50 m final touchdown
%
%   run_trained_agent('orbit', struct('sidecar','off'))   % same, barrier detached
%   run_trained_agent('orbit', struct('show','any'))      % next episode, pass or fail
%
% Scenario names are resolved by core/phase_from_name.
%
% Rolls episodes until it finds one matching `show`, then animates that one and reports
% how many attempts it took. That count is printed deliberately: the measured agent lands
% on the first attempt every time, so anything else is a regression the search would
% otherwise hide.
%
% --- WHY THIS DOES NOT CALL main_simulation ---
% main_simulation runs to params.max_sim_steps, which is 900 s. The agent was TRAINED
% under params.max_agent_steps, which is 300 s. Playing it back with three times the
% clock it ever saw is not the same experiment: a run that would have been scored a
% timeout at 300 s instead spends 600 more seconds hovering, which looks like a hang
% rather than the failure mode it is. Rolling the episode here keeps playback and
% training on the same footing.
%
% The agent is a neural policy CLONED from the classical guidance law, not one that
% discovered the task by exploration, and it lands 100% of all four phases at n=30 with
% the barrier attached or detached. README.md, "Why the pipeline has this shape", explains
% why cloning rather than RL and why selection is on landing rate rather than loss.
%
% Inputs:
%   scenario - (Optional) 'touchdown' | 'approach' | 'terminal' | 'orbit'
%   opts     - (Optional) struct:
%                   .sidecar    'on' (default) | 'off'
%                   .show       'landing' (default) | 'any'
%                   .phase      [] for the mixed curriculum (default), or 1 | 2 | 3
%                   .max_tries  default 25
%                   .agent_file default 'cloned_agent_4phase.mat'
%                   .animate    default true
%
% Outputs:
%   ep - the rolled episode (telemetry, outcome, veto counts)

    % Scenario names, not numbers. run_trained_agent(true, struct('phase',4)) meant
    % nothing to read; run_trained_agent('orbit') says what it does.
    if nargin < 1, scenario = []; end
    if nargin < 2, opts = struct(); end

    % First argument may be a scenario name, or the old logical sidecar flag.
    use_sidecar = true;
    if islogical(scenario) || (isnumeric(scenario) && isscalar(scenario) && ismember(scenario,[0 1]) && ~isempty(scenario))
        use_sidecar = logical(scenario);
        scenario = [];
    end
    if ~isfield(opts,'sidecar'), opts.sidecar = use_sidecar; end
    if ischar(opts.sidecar) || isstring(opts.sidecar)
        opts.sidecar = any(strcmpi(char(opts.sidecar), {'on','true','yes'}));
    end
    use_sidecar = logical(opts.sidecar);

    if ~isempty(scenario)
        opts.phase = phase_from_name(scenario);
    end

    % Remaining defaults. These were lost when the argument handling was rewritten for
    % scenario names, which broke every call - the kind of thing a smoke test catches
    % immediately and a lint pass does not.
    if ~isfield(opts,'show'),       opts.show       = 'landing'; end
    if ~isfield(opts,'phase'),      opts.phase      = [];        end
    if ~isfield(opts,'max_tries'),  opts.max_tries  = 25;        end
    if ~isfield(opts,'animate'),    opts.animate    = true;      end
    if ~isfield(opts,'agent_file'), opts.agent_file = 'cloned_agent_4phase.mat'; end

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
        select_phase(env, opts.phase);
    end

    fprintf('\nAgent: %s   sidecar %s\n', opts.agent_file, upper(gm));

    % --- Search for an episode matching `show` ---
    outcomes = {};
    ep = [];
    for k = 1:opts.max_tries
        % Roll to the LONGEST phase budget. The default is params.max_agent_steps
        % (3000), which truncates a Phase 4 powered descent at ~8900 steps and reports
        % it as a timeout that never happened.
        candidate = rollout_episode(env, agent, max(p.phase_max_steps));
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
                    'The evaluated agent lands 100%% of the time at n=30 per phase, so ' ...
                    'this means the agent file has changed or was retrained. Add ' ...
                    'struct(''show'',''any'') to watch a failure instead.\n'], ...
                opts.max_tries, strjoin(unique(outcomes), ', '));
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
