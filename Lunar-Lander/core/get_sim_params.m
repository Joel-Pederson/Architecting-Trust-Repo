function params = get_sim_params()
    % Lander Parameters: Single source of truth for the physical universe
    % Apollo 11 Specifications - State based off the historical Apollo Powered Descent Initiation (PDI)
    params.dry_mass = 4280;           % kg (Actual empty weight of LEM)
    params.gravity = 1.62;            % m/s^2 (Lunar gravity)
    params.r_lunar = 1737400;         % meters (Radius of the moon)
    
    % Dynamic Inertia Parameters
    params.inertia_dry = 24000;       % kg*m^2 (Inertia of the empty lander)
    params.inertia_fuel_full = 45000; % kg*m^2 (Inertia contribution of full fuel tanks)
    
    % Engine & Fuel Parameters
    params.max_main_thrust = 45040;   % Newtons (Actual thrust of the LEM DPS)
    params.max_mass_burn_rate = 15.6; % kg/s (Approximate DPS max flow rate)
    params.max_main_fuel = 8200;      % kg (Total main engine propellant)
    
    params.max_side_torque = 2000;    % Newton-meters (RCS Maximum Torque)
    params.max_rcs_burn_rate = 0.5;   % kg/s (RCS flow rate at max torque)
    params.max_rcs_fuel = 300;        % kg (Total RCS propellant)

    % Simulation Parameters
    params.dt = 0.02;                 % Simulation Timing (50Hz) seconds per frame
    params.max_steps = 45000;         % Maximum allowable time steps (900 seconds / 15 minutes total)

end