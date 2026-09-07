function ep = fly_with_faults(controller, draw, params, opts)
% FLY_WITH_FAULTS One episode flown by a degraded controller on a possibly off-nominal plant.
%
% The single flight loop behind both fault studies. The enumerated sweep builds a draw with
% one fault in it; the Monte Carlo study builds draws with several. Neither carries its own
% copy of this loop, because two copies is how the same experiment ends up measured two
% slightly different ways.
%
% --- WHAT A DRAW IS ---
% A struct describing everything that is wrong with this episode:
%
%   .sensors   struct array of perception faults, each {type, magnitude, onset}. Applied in
%              sequence, so a frozen altimeter and an under-read descent rate can be active
%              at once - which the enumerated sweep, being one-factor-at-a-time, can never
%              express.
%   .thrust    multiplicative scale on commanded main thrust (1.0 = healthy)
%   .thrust_onset  agent step at which that degradation begins
%   .dry_mass  plant mass override, [] for nominal
%   .ic_scale  multiplier on the initial condition's offset from the phase nominal
%
% Onsets matter. A fault present from t=0 is one the controller has effectively been
% trimmed against by its own feedback; a fault that appears at 200 m during a descent is a
% different and harder event, and only the second resembles real degradation.
%
% --- THE PERCEPTION BOUNDARY ---
% The controller reads corrupted state. The barrier always reads TRUE state. That split is
% the Perception Gatekeeper from the paper and it is what gives the barrier authority the
% nominal controller does not have. Actuator faults are applied to the COMMAND after the
% controller has chosen it, so the controller cannot see those either.
%
% Inputs:
%   controller - struct: .kind 'pilot' | 'agent', and .agent when kind is 'agent'
%   draw       - fault specification as above; see sample_ood_draw
%   params     - get_sim_params
%   opts       - struct: .phase, .guardian ('on'|'off'), .seed
%
% Outputs:
%   ep - struct with .outcome, .impact, .vetoes, .steps, .touchdown_dy, .touchdown_dx
%
% See also SAMPLE_OOD_DRAW, RUN_OOD_STUDY, RUN_FAULT_INJECTION_STUDY, APPLY_SENSOR_FAULT.

    env = LunarLanderEnv('DenseBaseline', opts.guardian);

    % Plant perturbation goes in BEFORE reset, because the initial fuel load and every
    % derivative downstream read it. The barrier sees the same altered plant the vehicle
    % actually has - it is not being handicapped, it is being asked whether its own model
    % still bounds a vehicle that is heavier than expected.
    if isfield(draw, 'dry_mass') && ~isempty(draw.dry_mass)
        env.params.dry_mass = draw.dry_mass;
    end

    select_phase(env, opts.phase);
    rng(opts.seed);
    reset(env);

    % Push the initial condition outside the training box if the draw asks for it. The
    % environment randomises within a phase, but the agent has SEEN that distribution;
    % scaling the offset from nominal is what makes a condition genuinely unfamiliar.
    if isfield(draw, 'ic_scale') && ~isempty(draw.ic_scale) && draw.ic_scale ~= 1
        env.State = stretch_initial_condition(env.State, opts.phase, draw.ic_scale, params);
    end

    cap = params.phase_max_steps(opts.phase);
    history = {};
    landed_impacts = NaN;

    for i = 1:cap
        s_true = env.State;

        % History is kept unconditionally: a delay fault may appear partway through a
        % draw, and it needs states from before its own onset to be meaningful.
        history{end+1} = s_true; %#ok<AGROW>
        if numel(history) > 128, history(1) = []; end

        s_seen = apply_draw_sensors(s_true, draw, i, history);
        u = command_from(controller, s_seen, s_true, params);

        % Actuator degradation, applied after the controller has committed.
        if isfield(draw, 'thrust') && ~isempty(draw.thrust)
            onset = 0;
            if isfield(draw, 'thrust_onset'), onset = draw.thrust_onset; end
            if i >= onset
                u(1) = u(1) * draw.thrust;
            end
        end

        % Mass for the inverse map comes from TRUE state. A corrupted fuel gauge would be
        % a different experiment, and folding it in here would confound this one.
        [~, ~, done] = step(env, command_to_action(u, s_true, params));
        if done, break; end
    end

    ep.outcome = env.Outcome;
    ep.vetoes  = env.VetoCount;
    ep.steps   = i;
    ep.touchdown_dx = env.State(3);
    ep.touchdown_dy = env.State(4);

    % Only episodes that reached the ground have an impact speed. Timeouts and out-of-box
    % episodes get NaN rather than zero: scoring a vehicle that never arrived as a perfect
    % landing would flatter exactly the failure mode worth seeing.
    if any(strcmp(ep.outcome, {'landed', 'crashed'}))
        landed_impacts = hypot(env.State(3), env.State(4));
    end
    ep.impact = landed_impacts;
end


function s_seen = apply_draw_sensors(s_true, draw, step_idx, history)
% Compose every active perception fault, in order.
    s_seen = s_true;
    if ~isfield(draw, 'sensors') || isempty(draw.sensors)
        return;
    end
    for k = 1:numel(draw.sensors)
        f = draw.sensors(k);
        if step_idx < f.onset
            continue;
        end
        % Each fault corrupts the ALREADY-CORRUPTED state, not the true one, so a bias and
        % a freeze compose the way two real sensor faults would rather than the later one
        % overwriting the earlier.
        s_seen = apply_sensor_fault(s_seen, f.type, f.magnitude, history);
    end
end


function u = command_from(controller, s_seen, s_true, params)
% One controller interface for both arms of the study.
    switch controller.kind
        case 'pilot'
            % braking_guidance, not scripted_pilot: it covers the whole envelope and
            % delegates to the terminal law below 2.5 km. Calling the terminal law
            % directly cannot fly a powered descent at all.
            u = braking_guidance(s_seen, params);

        case 'agent'
            a = getAction(controller.agent, {get_ai_observation(s_seen, params)});
            if iscell(a), a = a{1}; end
            % Normalised action to physical command, using TRUE mass, so that an actuator
            % fault can be applied in Newtons before the environment maps it back.
            u = action_to_command(reshape(double(a), [], 1), s_true, params);

        otherwise
            error('flyWithFaults:UnknownController', ...
                'controller.kind must be ''pilot'' or ''agent'', got ''%s''.', controller.kind);
    end
end


function s = stretch_initial_condition(s, phase, scale, params)
% Push the start outside the box the agent trained in, by scaling the episode's DEVIATION
% FROM ITS PHASE NOMINAL - never the absolute state.
%
% The first version scaled absolute values, which is wrong for any phase whose nominal is
% not zero and catastrophically wrong for Phase 4. There, init_x is -550,000 m and
% init_dx is +1697 m/s, so a 2x "stretch" started the vehicle 1,100 km downrange - outside
% the 600 km flight box, out of bounds at step one - travelling at twice orbital velocity.
% That is not an off-nominal initial condition, it is a different mission, and it silently
% turned most Phase 4 draws into instant out-of-bounds episodes that the study then scored
% as the barrier failing.
%
% Altitude is deliberately left alone. Raising it hands the vehicle more room, which makes
% the task easier and would flatter the result rather than stress it.
    nom = phase_nominal(phase, s, params);

    for idx = [1 3 4 5 6]          % downrange, lateral rate, descent rate, pitch, body rate
        s(idx) = nom(idx) + (s(idx) - nom(idx)) * scale;
    end
end


function nom = phase_nominal(phase, s, params)
% The deterministic centre of each phase's initial-condition distribution. Signs for the
% drift terms are taken from the drawn state, because the environment randomises drift
% direction per episode and a fixed sign would turn a stretch into a reversal.
    drift = sign(s(3));
    if drift == 0, drift = 1; end

    switch phase
        case 1, nom = [0; 50;   0;            -2;  0;                0];
        case 2, nom = [0; 500;  drift * 10;  -10;  0;                0];
        case 3, nom = [0; 2500; drift * 20;  -25;  0;                0];
        case 4, nom = [-params.pdi_downrange; params.pdi_altitude; ...
                        params.pdi_velocity;  params.pdi_descent; ...
                        params.pdi_pitch;     0];
        otherwise
            % Unknown phase: treat the drawn state as its own nominal, so the stretch
            % becomes a no-op rather than an arbitrary perturbation.
            nom = s(1:6);
    end
    nom = [nom(:); 0; 0];
end
