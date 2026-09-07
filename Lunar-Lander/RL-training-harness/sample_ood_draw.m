function draw = sample_ood_draw(params, phase, opts)
% SAMPLE_OOD_DRAW Draws one off-nominal condition the controller was never trained for.
%
% --- WHY SAMPLE RATHER THAN ENUMERATE ---
% run_fault_injection_study asks whether the barrier holds against the faults we thought
% of: four types, five hand-picked magnitudes, strictly one at a time. That is the question
% you would ask of a FAULT-TOLERANT CONTROLLER, which has to anticipate each failure mode.
%
% A runtime barrier claims something different - that it enforces a state-space property
% and therefore does not need the failure mode enumerated in advance. Testing it only
% against a list quietly concedes the thing the architecture is claiming. This samples a
% continuum instead, with faults combined, magnitudes continuous, onsets mid-flight and the
% plant itself off-nominal.
%
% It also lets the architecture's boundary be DISCOVERED rather than confirmed. The known
% limit - control delay beyond ~20 steps - was found because delay happened to be in the
% enumerated spec. A sample finds the edge without being told where to look.
%
% --- THE HONEST LIMIT ---
% This does not escape the enumeration problem, it widens it. "Faults I thought of" becomes
% "a sampling distribution I chose", and the distribution below is a choice like any other.
% A continuum beats five points and compound beats one-at-a-time, but coverage claims from
% this study are claims about THIS box. What actually escapes the problem is proving
% forward invariance of the barrier set, which is a different piece of work.
%
% --- WHY EVERY FAULT HERE IS PERSISTENT ---
% Zero-mean noise was tried and measured: action noise up to sigma = 0.5 left landing rates
% at ~100% with and without the barrier, because a loop re-deciding at 10 Hz rejects
% zero-mean disturbance by construction. Sampling that way burns the budget on conditions
% any feedback controller shrugs off. Every fault drawn below is a BIAS, a FREEZE, a
% DEGRADATION or a DELAY - something that does not average out.
%
% Inputs:
%   params - get_sim_params
%   phase  - curriculum phase this draw is for (fault onsets scale with its length)
%   opts   - (Optional) struct:
%              .p_sensor    probability each sensor fault type is active (default 0.45)
%              .p_thrust    probability of actuator degradation      (default 0.40)
%              .p_mass      probability of an off-nominal dry mass   (default 0.35)
%              .p_ic        probability of a stretched initial state (default 0.35)
%              .max_ic      largest initial-condition stretch        (default 2.0)
%              .allow_delay include control delay in the pool        (default true)
%
% Outputs:
%   draw - the fault specification consumed by fly_with_faults, plus .label and .severity
%
% See also FLY_WITH_FAULTS, RUN_OOD_STUDY, APPLY_SENSOR_FAULT.

    if nargin < 3, opts = struct(); end
    if ~isfield(opts,'p_sensor'),    opts.p_sensor    = 0.45; end
    if ~isfield(opts,'p_thrust'),    opts.p_thrust    = 0.40; end
    if ~isfield(opts,'p_mass'),      opts.p_mass      = 0.35; end
    if ~isfield(opts,'p_ic'),        opts.p_ic        = 0.35; end
    if ~isfield(opts,'max_ic'),      opts.max_ic      = 2.0;  end
    if ~isfield(opts,'allow_delay'), opts.allow_delay = true; end

    cap = params.phase_max_steps(phase);

    draw = struct();
    draw.sensors      = struct('type', {}, 'magnitude', {}, 'onset', {});
    draw.thrust       = 1.0;
    draw.thrust_onset = 0;
    draw.dry_mass     = [];
    draw.ic_scale     = 1.0;
    parts = {};

    % --- PERCEPTION FAULTS ---
    % Ranges are deliberately wider than the enumerated sweep's, and open at the top end,
    % so the sample includes magnitudes past anything previously tested.
    pool = { 'alt_bias',   @() 2 + 48 * rand()^1.5,  'altimeter +%.1f m'; ...
             'vel_bias',   @() 0.2 + 0.75 * rand(),  'descent under-read %.2f'; ...
             'alt_freeze', @() 5 + 70 * rand(),      'altimeter frozen below %.0f m' };
    if opts.allow_delay
        pool(end+1,:) = { 'delay', @() 2 + 28 * rand(), 'control delayed %.0f steps' };
    end

    for k = 1:size(pool,1)
        if rand() < opts.p_sensor
            mag = pool{k,2}();
            % Onset anywhere in the first 70% of the episode. A fault that appears at
            % altitude is a different event from one present at t = 0, which the
            % controller's own feedback has effectively been trimmed against.
            onset = round(rand()^2 * 0.7 * cap);
            draw.sensors(end+1) = struct('type', pool{k,1}, ...
                                         'magnitude', mag, 'onset', onset);
            parts{end+1} = sprintf(pool{k,3}, mag); %#ok<AGROW>
        end
    end

    % --- ACTUATOR DEGRADATION ---
    if rand() < opts.p_thrust
        loss = 0.05 + 0.50 * rand();          % up to half the engine, past the swept 40%
        draw.thrust = 1 - loss;
        draw.thrust_onset = round(rand()^2 * 0.7 * cap);
        parts{end+1} = sprintf('engine -%.0f%%', 100*loss);
    end

    % --- OFF-NOMINAL PLANT ---
    % The heavy-rover case from the issue tracker, sampled rather than fixed. Note this
    % tests the AGENT's generalisation more than the barrier's coverage: the barrier reads
    % total mass from state, so a heavier vehicle is inside its model and it will simply
    % compute a smaller available deceleration and brake earlier.
    if rand() < opts.p_mass
        draw.dry_mass = params.dry_mass * (0.85 + 0.75 * rand());   % 0.85x to 1.60x
        parts{end+1} = sprintf('dry mass %.2fx', draw.dry_mass / params.dry_mass);
    end

    % --- OFF-DISTRIBUTION INITIAL CONDITION ---
    if rand() < opts.p_ic
        draw.ic_scale = 1.15 + (opts.max_ic - 1.15) * rand();
        parts{end+1} = sprintf('IC %.2fx', draw.ic_scale);
    end

    if isempty(parts)
        draw.label = 'nominal';
    else
        draw.label = strjoin(parts, ' + ');
    end

    % A crude count of simultaneous insults, reported so the results can be split by how
    % compound a draw was. Whether the barrier degrades gracefully as faults stack is a
    % more useful question than a single pooled rate.
    draw.severity = numel(draw.sensors) + (draw.thrust < 1) + ...
                    (~isempty(draw.dry_mass)) + (draw.ic_scale ~= 1);
end
