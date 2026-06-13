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
    %   x(1) : x_pos   (Horizontal position, meters)
    %   x(2) : y_pos   (Vertical altitude, meters)
    %   x(3) : dx      (Horizontal velocity, m/s)
    %   x(4) : dy      (Vertical velocity, m/s)
    %   x(5) : theta   (Angle from vertical, radians. + is CCW/tilt left)
    %   x(6) : dtheta  (Angular velocity, rad/s)
    %   x(7) : m_fuel  (Current fuel mass, kg)
    %
    % --- CONTROL VECTOR (u) ---
    %   u(1) : T_main  (Main engine thrust, Newtons. Bounded: 0 to T_max)
    %   u(2) : Tau_side(RCS Side engine torque, Newton-meters)
    
    % --- 1. Unpack State ---
    % Position is tracked for the environment, but not used to calculate derivatives
    x_pos    = x(1); 
    y_pos    = x(2); 
    dx       = x(3);
    dy       = x(4);
    theta    = x(5);
    dtheta   = x(6);
    m_fuel   = x(7); 
    
    % --- 2. Unpack Controls & Parameters ---
    T_main   = u(1); 
    Tau_side = u(2); % Renamed locally to Tau_side to clarify it is a Torque (Nm)
    
    m_dry    = params.dry_mass; % kg
    g        = params.gravity;  % m/s^2 (Magnitude only. Direction is handled in equations)
    I        = params.inertia;  % kg*m^2
    
    T_max    = params.max_main_thrust;
    mdot_max = params.max_mass_burn_rate; 
    
    % --- 3. Dynamic Mass Calculation ---
    % Hardware Constraint: If the tank is empty, the engine cuts out completely
    if m_fuel <= 0
        m_fuel = 0;
        T_main = 0; % No fuel = no thrust, overriding AI/Sidecar commands
    end
    
    m_total = m_dry + m_fuel;
    
    % --- 4. Equations of Motion (Newton's Laws) ---
    
    % X-axis: Main thrust pushing sideways based on tilt angle.
    % (If nose leans left (+theta), thrust points right. Nose leans right (-theta), thrust points left).
    ddx = (-T_main * sin(theta)) / m_total;
    
    % Y-axis: Main thrust pushing up, fighting gravity.
    % Gravity is subtracted here, which is why params.gravity must be a positive magnitude.
    ddy = ((T_main * cos(theta)) / m_total) - g;
    
    % Rotation: Side thrust acting on the moment of inertia.
    % angular_acceleration = torque / inertia
    ddtheta = Tau_side / I;
    
    % --- 5. Mass Depletion (The Burn Rate) ---
    % Fuel burns proportionally to the percentage of main thrust commanded.
    % Defensive programming: abs(T_main) ensures we don't magically create fuel if AI commands negative thrust.
    dm_fuel = -mdot_max * (abs(T_main) / T_max);
    
    % If tank is already empty, stop subtracting mass to prevent negative fuel
    if m_fuel <= 0
        dm_fuel = 0;
    end
    
    % --- 6. Pack Derivatives for Integration ---
    dxdt = [dx; dy; ddx;