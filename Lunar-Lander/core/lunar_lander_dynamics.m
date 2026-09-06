function dxdt = lunar_lander_dynamics(x, u, params)
    % LUNAR_LANDER_DYNAMICS Calculates the continuous-time derivatives of the state.
    % Includes dynamic mass depletion based on fuel consumption.
    %
    % --- COORDINATE SYSTEM ---
    % Origin (0,0) : The target landing pad on the surface.
    % Y-Axis       : Positive is UP (Altitude). Gravity acts in the -Y direction.
    % X-Axis       : Positive is RIGHT.
    % Rotation     : Positive Theta is Counter-Clockwise (Nose tilted left).
    %
    % --- STATE VECTOR (x) ---
    %   x(1) : x_pos       (Horizontal position, meters)
    %   x(2) : y_pos       (Vertical altitude, meters)
    %   x(3) : dx          (Horizontal velocity, m/s)
    %   x(4) : dy          (Vertical velocity, m/s)
    %   x(5) : theta       (Angle from vertical, radians. + is CCW/tilt left)
    %   x(6) : dtheta      (Angular velocity, rad/s)
    %   x(7) : m_main_fuel (Current main fuel mass, kg)
    %   x(8) : m_rcs_fuel  (Current RCS propellant mass, kg)
    %
    % --- CONTROL VECTOR (u) ---
    %   u(1) : T_main  (Main engine thrust, Newtons. Bounded: 0 to T_max)
    %   u(2) : Tau_side(RCS Side engine torque, Newton-meters)
    
    % --- 1. Unpack State ---
    % Position is tracked for the environment, but not used to calculate derivatives
    x_pos       = x(1); 
    y_pos       = x(2); 
    dx          = x(3);
    dy          = x(4);
    theta       = x(5);
    dtheta      = x(6);
    m_main_fuel = x(7); 
    m_rcs_fuel  = x(8);
    
    % --- 2. Unpack Controls & Parameters ---
    T_main   = u(1); 
    Tau_side = u(2); % Renamed locally to Tau_side to clarify it is a Torque (Nm)
    
    m_dry    = params.dry_mass; % kg
    g        = params.gravity;  % m/s^2 (Magnitude only. Direction is handled in equations)
    r_lunar  = params.r_lunar;  % m
    
    I_dry    = params.inertia_dry; % kg*m^2
    I_fuel_full = params.inertia_fuel_full; % kg*m^2
    
    T_max    = params.max_main_thrust;
    mdot_max = params.max_mass_burn_rate;
    Tau_max  = params.max_side_torque;
    mdot_rcs = params.max_rcs_burn_rate;
    
    max_main = params.max_main_fuel;
    max_rcs  = params.max_rcs_fuel;
    
    % --- 3. Dynamic Mass Calculation ---
    % Hardware Constraint: If a tank is empty, its engine cuts out completely
    if m_main_fuel <= 0
        m_main_fuel = 0;
        T_main = 0; % No fuel = no thrust, overriding AI/Sidecar commands
    end
    if m_rcs_fuel <= 0
        m_rcs_fuel = 0;
        Tau_side = 0; % No RCS propellant = no torque
    end
    
    m_total = m_dry + m_main_fuel + m_rcs_fuel;
    
    % --- 4. Higher Fidelity Physics Extensions ---
    
    % 4a. Centrifugal Lift
    % At orbital velocities, centrifugal force counteracts gravity.
    %
    % NOTE ON THE COORDINATE FRAME. This makes the model CURVILINEAR, not Cartesian:
    % y_pos is altitude above the surface and x_pos is downrange ARC LENGTH along it, with
    % gravity always normal to the surface. That is the standard flat-Moon approximation
    % used in descent guidance, and it is what makes a 550 km powered descent meaningful -
    % in a true Cartesian frame the surface would curve 48 km away from the lander over
    % that range, but here the surface is at y = 0 by construction.
    r = r_lunar + y_pos;
    a_centrifugal = (dx^2) / r;

    % Inverse-square gravity rather than the surface value. 1.7% at PDI altitude - small,
    % but free, and it removes a systematic bias from a 480 s braking burn.
    g_local = g * (r_lunar / r)^2;
    g_apparent = max(0, g_local - a_centrifugal); % Clamp: gravity cannot go negative below escape velocity.
    
    % 4b. Dynamic Moment of Inertia
    % The ship gets easier to spin as fuel is burned. Interpolate based on remaining fuel mass.
    fuel_ratio = (m_main_fuel + m_rcs_fuel) / (max_main + max_rcs);
    I_current = I_dry + (fuel_ratio * I_fuel_full);
    
    % --- 5. Equations of Motion (Newton's Laws) ---
    % F = ma --> a = F/m
    % X-axis: Main thrust pushing sideways based on tilt angle.
    % (If nose leans left (+theta), thrust points right. Nose leans right (-theta), thrust points left).
    % The curvilinear coupling term -2*dx*dy/r is the price of measuring x as arc length:
    % as the vehicle descends, conservation of angular momentum changes its downrange
    % rate. Worth ~0.06 m/s^2 against 3.52 m/s^2 of thrust at PDI (1.7%), the same order
    % as the gravity correction above, and it is one term.
    ddx = (-T_main * sin(theta)) / m_total - (2 * dx * dy) / r;
    
    % Y-axis: Main thrust pushing up, fighting apparent gravity.
    % Gravity is subtracted here, which is why params.gravity must be a positive magnitude.
    ddy = ((T_main * cos(theta)) / m_total) - g_apparent;
    
    % Rotation: Side thrust acting on the dynamic moment of inertia.
    % angular_acceleration = torque / inertia
    ddtheta = Tau_side / I_current;
    
    % --- 6. Mass Depletion (Dual Tanks) ---
    % Fuel burns proportionally to the percentage of thrust commanded.
    % Defensive programming: abs(T_main) ensures logic doesn't magically create fuel if AI commands negative thrust.
    % max fuel rate consumption times percent of throttle being used
    dm_main_fuel = -mdot_max * (abs(T_main) / T_max);
    dm_rcs_fuel  = -mdot_rcs * (abs(Tau_side) / Tau_max);
    
    % If tank is already empty, stop subtracting mass
    if m_main_fuel <= 0, dm_main_fuel = 0; end
    if m_rcs_fuel <= 0, dm_rcs_fuel = 0; end
    
    % --- 7. Pack Derivatives for Integration ---
    dxdt = [dx; dy; ddx; ddy; dtheta; ddtheta; dm_main_fuel; dm_rcs_fuel];