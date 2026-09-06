function norm_state = get_ai_observation(raw_state, params)
% GET_AI_OBSERVATION Compresses true physics state vector into roughly [-1, 1] bounds
% and calculates the barrier states (h_alt, h_fuel) so the AI can see the safety net.
%
% Inputs:
%   raw_state - 8x1 vector: [x; y; dx; dy; theta; dtheta; m_main_fuel; m_rcs_fuel]
%   params    - struct with hardware limits
%
% Outputs:
%   norm_state - 10x1 normalized vector

    if nargin < 2
        params = get_sim_params();
    end

    % Wrap angle theta to [-pi, pi] so it never explodes outside normalized bounds
    wrapped_theta = atan2(sin(raw_state(5)), cos(raw_state(5)));
    
    % Unpack state for barrier calculations
    y_pos = raw_state(2);
    dx = raw_state(3);
    dy = raw_state(4);
    theta = raw_state(5);
    m_main_fuel = raw_state(7);
    m_rcs_fuel = raw_state(8);
    m_total = params.dry_mass + m_main_fuel + m_rcs_fuel;
    
    % Calculate centrifugal lift & apparent gravity
    a_centrifugal = (dx^2) / (params.r_lunar + y_pos);
    g_apparent = max(0, params.gravity - a_centrifugal);
    
    % Altitude Barrier (h_alt)
    a_max_upright = (params.max_main_thrust / m_total) - g_apparent;
    if dy < 0
        h_alt = y_pos - ((dy^2) / (2 * max(0.1, a_max_upright)));
    else
        h_alt = y_pos; % No braking distance required if flying upwards
    end
    
    % Fuel Barrier (h_fuel)
    hover_time_reserve = 3.0;
    thrust_to_hover = m_total * g_apparent;
    burn_rate_at_hover = params.max_mass_burn_rate * (thrust_to_hover / params.max_main_thrust);
    safety_buffer_fuel = burn_rate_at_hover * hover_time_reserve;
    
    a_max = (params.max_main_thrust * cos(theta) / m_total) - g_apparent;
    if dy < 0 && a_max > 0
        t_stop = abs(dy) / a_max;
        fuel_needed_to_stop = params.max_mass_burn_rate * t_stop;
    else
        fuel_needed_to_stop = 0;
    end
    h_fuel = m_main_fuel - (fuel_needed_to_stop + safety_buffer_fuel);

    % --- NORMALISATION: SIGNED LOG, NOT LINEAR ---
    % The scenario spans five orders of magnitude. A powered descent begins 550 km short
    % of the pad at 1697 m/s; it ends needing to resolve 0.5 m/s of lateral drift, because
    % that is the touchdown limit. No linear scale serves both: divide by 1700 and the
    % entire landing regime collapses into 0.0003 of the range, which is exactly the
    % blindness that once made closing a FAILING 2 m/s drift to a PASSING 0.5 m/s worth
    % 0.06 reward and cost five training runs.
    %
    % A signed log compresses the far field and keeps resolution near zero, where the task
    % is decided. Measured on lateral velocity, the gap between 0.5 and 2 m/s:
    %
    %     linear /100    0.015 of range      linear /1700   0.0009
    %     signed log     0.094 of range      (6x better than /100, and it reaches 1700)
    %
    % Gradient near zero is 1/(K*v0) against 1/vmax for a linear scale - about 13x more
    % resolution at the touchdown end while still spanning orbital velocity.
    norm_state = [
        signed_log(raw_state(1), 10,   params.max_abs_x);   % downrange arc length
        signed_log(raw_state(2), 10,   params.max_alt);     % altitude
        signed_log(raw_state(3), 1,    2000);               % horizontal velocity
        signed_log(raw_state(4), 1,    200);                % vertical velocity
        wrapped_theta / pi;
        raw_state(6) / pi;
        raw_state(7) / 8200;
        raw_state(8) / 300;
        signed_log(h_alt,  10, params.max_alt);             % altitude margin
        h_fuel / 8200                                       % fuel margin
    ];
end


function n = signed_log(v, v0, vmax)
% Signed logarithmic normaliser, odd about zero and reaching +/-1 at +/-vmax.
%
%   n(v) = sign(v) * log(1 + |v|/v0) / log(1 + vmax/v0)
%
% v0 sets where resolution is concentrated: below it the map is nearly linear with slope
% 1/(v0*log(1+vmax/v0)), above it compressive. Choose v0 at the scale the task is decided
% on - 1 m/s for velocities against a 0.5 m/s touchdown limit, 10 m for positions.
%
% Values beyond vmax are not clipped. They exceed +/-1, which is honest: a network fed a
% saturated input cannot tell "far" from "very far", and the flight box terminates the
% episode shortly afterwards anyway.
    n = sign(v) .* log(1 + abs(v) / v0) ./ log(1 + vmax / v0);
end
