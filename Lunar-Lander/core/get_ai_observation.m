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

    norm_state = [
        raw_state(1) / 1000;    % Max expected horizontal drift
        raw_state(2) / 3000;    % Max expected altitude
        raw_state(3) / 100;     % Max expected horizontal velocity
        raw_state(4) / 100;     % Max expected vertical velocity
        wrapped_theta / pi;
        raw_state(6) / pi;
        raw_state(7) / 8200;
        raw_state(8) / 300;
        h_alt / 3000;           % Normalize h_alt against max altitude
        h_fuel / 8200           % Normalize h_fuel against max fuel
    ];
end
