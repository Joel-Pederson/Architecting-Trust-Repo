function results = demo_agent_rescue(scenario, opts)
% DEMO_AGENT_RESCUE The sidecar rescuing the TRAINED NEURAL AGENT from a blind sensor.
%
%   demo_agent_rescue()                  % 'terminal' scenario, 12 m altimeter fault
%   demo_agent_rescue('orbit')           % the same fault on the descent from orbit
%   demo_agent_rescue('terminal', struct('magnitude', 25))     % harsher fault
%   demo_agent_rescue('orbit',    struct('animate', false))    % numbers only
%
% Scenario names are resolved by core/phase_from_name:
%   'touchdown' | 'approach' | 'terminal' | 'orbit'
%
% --- WHY THIS DEMO EXISTS ---
% Running the trained agent with the sidecar on and off shows nothing: it lands 100% of
% the time either way, and the barrier stays INACTIVE the whole flight. That is the
% non-intrusiveness result - a competent controller never approaches the barrier - but it
% makes for a demonstration in which nothing happens.
%
% The barrier only earns its place when the controller is WRONG. So this corrupts the
% altimeter the agent reads while leaving the sidecar on true state, and flies the same
% scenario twice. That split is the Perception Gatekeeper boundary from the paper, applied
% to a neural policy rather than to the classical pilot.
%
% The agent cannot detect the fault: every observation it receives is self-consistent, and
% nothing in its training distribution looked different.
%
% Inputs:
%   alt_bias - metres the altimeter over-reads (default 12)
%   phase    - curriculum phase 1-4 (default 3)
%   animate  - show the visualiser (default true)
%
% Outputs:
%   results - 1x2 struct array (guardian off, then on)

    if nargin < 1 || isempty(scenario), scenario = 'terminal'; end
    if nargin < 2, opts = struct(); end
    if ~isfield(opts,'magnitude'), opts.magnitude = 12;   end
    if ~isfield(opts,'animate'),   opts.animate   = true; end

    [phase, phase_label] = phase_from_name(scenario);
    alt_bias = opts.magnitude;
    animate  = opts.animate;

    here = fileparts(mfilename('fullpath'));
    addpath(genpath(here));
    p = get_sim_params();

    f = fullfile(here, 'cloned_agent_4phase.mat');
    if ~isfile(f)
        error('demoAgentRescue:NoAgent', 'Agent not found: %s', f);
    end
    d = load(f, 'agent'); agent = d.agent;
    if isprop(agent, 'UseExplorationPolicy'), agent.UseExplorationPolicy = false; end

    fprintf('\n=== SIDECAR RESCUE: TRAINED AGENT, BLIND ALTIMETER ===\n');
    fprintf('Scenario: %s\n', phase_label);
    fprintf('The agent''s altimeter reads %g m HIGH; it cannot detect this.\n', alt_bias);
    fprintf('The sidecar reads TRUE state. Same scenario and same policy in both runs.\n\n');

    results = struct('guardian', {}, 'outcome', {}, 'impact', {}, 'vetoes', {});
    modes = {'off', 'on'};

    for i = 1:2
        ep = fly(agent, p, modes{i}, alt_bias, phase);
        fprintf('  guardian %-3s : %-8s  impact %7.2f m/s  (dy %+7.2f, dx %+6.2f)  vetoes %4d\n', ...
            upper(modes{i}), ep.outcome, ep.impact, ep.dy, ep.dx, ep.vetoes);
        results(i) = struct('guardian', modes{i}, 'outcome', ep.outcome, ...
                            'impact', ep.impact, 'vetoes', ep.vetoes);
        if animate
            fprintf('    launching visualiser (guardian %s)...\n', upper(modes{i}));
            animate_lunar_lander(ep.t, ep.states(1,:), ep.states(2,:), ep.states(4,:), ...
                ep.states(5,:), ep.controls(1,:), ep.states(7,:), ep.veto, p);
        end
    end

    fprintf('\nTouchdown limits: |dy| <= %.1f, |dx| <= %.1f m/s\n', ...
        p.max_touchdown_dy, p.max_touchdown_dx);
    fprintf('Statistics over 30 fault cells: run_fault_injection_study\n\n');
end


function ep = fly(agent, p, guardian, alt_bias, phase)
% Same seed both runs, so the only difference is the barrier.
    env = LunarLanderEnv('DenseBaseline', guardian);
    env.CurriculumWeights = double((1:numel(p.phase_max_steps)) == phase);
    rng(101);
    reset(env);

    cap = p.phase_max_steps(phase);
    states   = zeros(8, cap + 1);
    controls = zeros(2, cap + 1);
    veto     = false(1, cap + 1);
    states(:,1) = env.State;

    n = 1;
    for i = 1:cap
        s_true = env.State;

        % The AGENT sees a corrupted altitude. The sidecar, inside the environment, sees
        % the truth - that asymmetry is the entire point.
        s_seen = apply_sensor_fault(s_true, 'alt_bias', alt_bias);
        a = getAction(agent, {get_ai_observation(s_seen, p)});
        if iscell(a), a = a{1}; end

        [~, ~, done, logs] = step(env, reshape(double(a), [], 1));
        n = i + 1;
        states(:,n)   = logs.State;
        controls(:,n) = logs.Control;
        veto(n)       = logs.VetoActive;
        if done, break; end
    end

    ep.states   = states(:, 1:n);
    ep.controls = controls(:, 1:n);
    ep.veto     = veto(1:n);
    ep.t        = (0:n-1) * p.agent_dt;
    ep.outcome  = env.Outcome;
    ep.vetoes   = env.VetoCount;
    ep.dy = env.State(4);
    ep.dx = env.State(3);
    ep.impact = hypot(env.State(3), env.State(4));
end
