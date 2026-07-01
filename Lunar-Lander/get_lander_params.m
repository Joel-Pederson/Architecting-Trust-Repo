function params = get_lander_params()
    % GET_LANDER_PARAMS: Single source of truth for the physical universe.
    % Apollo 11 Specifications
    params.dry_mass = 4280;           % kg (Actual empty weight of LEM)
    params.gravity = 1.62;            % m/s^2 (Lunar gravity)
    params.inertia = 24000;           % kg*m^2 (Calculated for a 4.3m x 7m box)
    params.max_main_thrust = 45040;   % Newtons (Actual thrust of the LEM DPS)
    params.max_mass_burn_rate = 15.6; % kg/s (Approximate DPS max flow rate)
    params.max_side_torque = 2000;    % Newton-meters (Physical maximum torque limit for the Reaction Control System (RCS))
end