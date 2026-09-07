function results = demo_sidecar_rescue(alt_bias, animate)
% DEMO_SIDECAR_RESCUE Watch the Safety Sidecar rescue a controller with a broken sensor.
%
% Flies the SAME scenario twice, from an identical initial condition, with an identical
% controller. The only difference is whether the Operational Sidecar is attached.
%
%   demo_sidecar_rescue()        % 10 m altimeter bias, animated
%   demo_sidecar_rescue(20)      % harsher fault
%   demo_sidecar_rescue(10, false)  % numbers only, no animation
%
% --- WHAT YOU ARE WATCHING ---
% The pilot's altimeter reads HIGH by alt_bias metres, so it believes it has more room
% than it does and starts braking too late. It cannot detect this: every sensor it has is
% self-consistent, and the fault is in the sensor itself.
%
% The sidecar reads TRUE state. It is not trying to fly better than the pilot - it does
% not know where the pad is. It only enforces one thing: never enter a state from which
% no recovery exists. Its altitude barrier is
%
%     h_alt = y - dy^2 / (2 * a_max)
%
% i.e. altitude minus the distance needed to stop at full thrust. When the pilot's late
% braking would drive that to zero, the sidecar takes authority and brakes.
%
% Expected: guardian OFF crashes, guardian ON lands. At 10 m of bias this is 0% vs 100%
% over 25 episodes; a single run is one draw from that.
%
% Inputs:
%   alt_bias - (Optional) metres the altimeter over-reads. Default 10.
%   animate  - (Optional) show the visualiser. Default true.
%
% Outputs:
%   results - 1x2 struct array (guardian off, then on) with outcome and touchdown state

    if nargin < 1 || isempty(alt_bias), alt_bias = 10;   end
    if nargin < 2 || isempty(animate),  animate  = true; end

    here = fileparts(mfilename('fullpath'));
    addpath(genpath(here));
    p = get_sim_params();

    modes = {'off', 'on'};
    results = struct('guardian', {}, 'outcome', {}, 'impact', {}, ...
                     'touchdown_dy', {}, 'touchdown_dx', {}, 'vetoes', {});

    fprintf('\n=== SAFETY SIDECAR RESCUE DEMO ===\n');
    fprintf('Fault: altimeter reads %g m HIGH. The pilot cannot detect this.\n', alt_bias);
    fprintf('Identical scenario and controller in both runs; only the barrier differs.\n\n');

    for i = 1:2
        mode = modes{i};
        ep = fly_once(p, mode, alt_bias);

        fprintf('  guardian %-3s : %-8s  impact %6.2f m/s  (dy %+6.2f, dx %+6.2f)  vetoes %d\n', ...
            upper(mode), ep.outcome, ep.impact, ep.touchdown_dy, ep.touchdown_dx, ep.vetoes);

        results(i) = struct('guardian', mode, 'outcome', ep.outcome, ...
                            'impact', ep.impact, 'touchdown_dy', ep.touchdown_dy, ...
                            'touchdown_dx', ep.touchdown_dx, 'vetoes', ep.vetoes);

        if animate
            fprintf('    launching visualiser (guardian %s)...\n', upper(mode));
            animate_lunar_lander(ep.t, ep.states(1,:), ep.states(2,:), ep.states(4,:), ...
                ep.states(5,:), ep.controls(1,:), ep.states(7,:), ep.veto, p);
        end
    end

    fprintf('\nTouchdown limits: |dy| <= %.1f m/s, |dx| <= %.1f m/s, |theta| <= %.2f rad\n', ...
        p.max_touchdown_dy, p.max_touchdown_dx, p.max_touchdown_tilt);
    fprintf('Full sweep across all faults and magnitudes: run_fault_injection_study\n\n');
end


function ep = fly_once(p, guardian, alt_bias)
% One episode, Phase 1, fixed seed so both arms face the identical initial condition.
    env = LunarLanderEnv('DenseBaseline', guardian);
    select_phase(env, 'touchdown');
    rng(101);
    reset(env);

    n_max = p.max_agent_steps;
    states   = zeros(8, n_max + 1);
    controls = zeros(2, n_max + 1);
    veto     = false(1, n_max + 1);
    states(:,1) = env.State;

    n = 1;
    for i = 1:n_max
        s_true = env.State;
        % The pilot sees a corrupted altitude; the sidecar sees the truth. That split is
        % the Perception Gatekeeper boundary.
        s_seen = apply_sensor_fault(s_true, 'alt_bias', alt_bias);
        u = scripted_pilot(s_seen, p);

        [~, ~, done, logs] = step(env, command_to_action(u, s_true, p));

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
    ep.touchdown_dy = env.State(4);
    ep.touchdown_dx = env.State(3);
    ep.impact   = sqrt(env.State(3)^2 + env.State(4)^2);
end
