function s_seen = apply_sensor_fault(s_true, fault, magnitude, history)
% APPLY_SENSOR_FAULT Corrupts the state a controller perceives, leaving reality intact.
%
% Returns what the PRIMARY CONTROLLER sees. The true state is unchanged, and the safety
% sidecar always reads the true state - that split is the Perception Gatekeeper boundary
% from the paper made concrete, and it is what gives the barrier authority the nominal
% controller does not have.
%
% --- WHY THESE FAULTS AND NOT NOISE ---
% Zero-mean sensor noise measures nothing here. Measured: injecting Gaussian action noise
% up to sigma = 0.5 left the pilot's landing rate at ~100% with and without the guardian,
% because a PD loop re-deciding at 10 Hz rejects zero-mean disturbance by construction.
% A runtime barrier earns its place against faults the controller cannot see or correct,
% so every model here is SYSTEMATIC.
%
% Supported faults:
%   'none'        - passthrough
%   'alt_bias'    - altimeter reads HIGH by `magnitude` metres, so the controller brakes
%                   too late. Perception fault. Measured: a 10 m bias takes the classical
%                   pilot from 100% landings to 0%; the sidecar restores 100%.
%   'vel_bias'    - descent rate under-read by a fraction `magnitude` in [0,1]. Harder
%                   than alt_bias because it corrupts the controller's own damping term.
%   'delay'       - controller acts on state from `magnitude` agent steps ago. Included
%                   BECAUSE the sidecar handles it badly: beyond ~20 steps there is no
%                   benefit at all. That is the architecture's boundary condition and it
%                   belongs in the paper.
%   'thrust_loss' - ACTUATOR fault, not a sensor one. The state passes through unchanged;
%                   the caller scales the commanded thrust. Accepted here so callers can
%                   dispatch every fault type through one switch.
%
% Inputs:
%   s_true    - 8x1 true physics state
%   fault     - char, one of the above
%   magnitude - fault size; units depend on the fault (see above)
%   history   - (Optional) cell array of prior states, oldest first. Required by 'delay'.
%
% Outputs:
%   s_seen - 8x1 state as perceived by the controller

    if nargin < 4, history = {}; end

    s_seen = s_true;

    switch fault
        case {'none', 'thrust_loss'}
            % thrust_loss acts on the command, not the observation.

        case 'alt_bias'
            s_seen(2) = s_true(2) + magnitude;

        case 'vel_bias'
            s_seen(4) = s_true(4) * (1 - magnitude);

        case 'alt_freeze'
            % Altimeter stops updating below `magnitude` metres: the controller keeps
            % seeing that altitude while it continues to descend. More severe than a
            % constant bias, because the error grows as the ground approaches.
            if s_true(2) < magnitude
                s_seen(2) = magnitude;
            end

        case 'delay'
            if ~isempty(history)
                s_seen = history{max(1, numel(history) - round(magnitude))};
            end

        otherwise
            error('applySensorFault:UnknownFault', ...
                ['Unknown fault model: %s. Supported: none, alt_bias, vel_bias, ' ...
                 'alt_freeze, delay, thrust_loss.'], fault);
    end
end
