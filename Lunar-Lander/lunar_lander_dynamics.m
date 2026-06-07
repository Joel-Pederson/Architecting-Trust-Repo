function dxdt = lunar_lander_dynamics(x, u, params)
    % LUNAR_LANDER_DYNAMICS Computes the continuous-time derivative of the
    % state. 
    % 
    % Assumptions: 2D motion plane. Constant mass. Perfect sensor state reading
    %
    % State Vector (x):
    %   x(1) : x_pos   (Horizontal position, meters)
    %   x(2) : y_pos   (Vertical altitude, meters)
    %   x(3) : dx      (Horizontal velocity, m/s)
    %   x(4) : dy      (Vertical velocity, m/s)
    %   x(5) : theta   (Angle from vertical, radians. + is CCW/tilt left)
    %   x(6) : dtheta  (Angular velocity, rad/s)
    %
    % Control Vector (u):
    %   u(1) : T_main  (Main engine thrust, Newtons. Always positive/up relative to lander)
    %  u(2) : T_side  (Side engine torque, Newton-meters)

    % --- 1. Unpack State & Controls ---
    dx = x(3);
    dy = x(4);
    theta = x(5);
    dtheta = x(6);
    
    T_main = u(1);
    T_side = u(2);
    
    % --- 2. Unpack Physical Parameters ---
    m = params.mass;       % kg
    g = params.gravity;    % m/s^2 (magnitude, 9.81 for Earth, 1.62 for Moon)
    I = params.inertia;    % kg*m^2
    
    %% Apply Newton-Euler method
    % Sum up the translational forces (F=ma) and rotational torques (τ=Iα), and rearrange them to solve for acceleration

    % --- 3. Translational Dynamics (Newton's Second Law: F = ma) ---
    % Horizontal (X-axis): 
    % The main thruster tilted by theta pushes the lander horizontally.
    % A positive theta (tilted left) means the thrust pushes the lander to the left (-x direction).
    % F_x = -T_main * sin(theta) + T_side * cos(theta)
    ddx = (1/m) * (-T_main * sin(theta) + T_side * cos(theta));
    
    % Vertical (Y-axis):
    % Gravity pulls down (-mg). The main thruster pushes up, modulated by the tilt angle.
    % F_y = -mg + T_main * cos(theta) + T_side * sin(theta)
    ddy = -g + (1/m) * (T_main * cos(theta) + T_side * sin(theta));
    
    % --- 4. Rotational Dynamics (Euler's Equation: tau = I * alpha) ---
    % Assume the main thruster fires perfectly through the center of mass (0 torque).
    % The side engine directly provides the torque (tau) to rotate the lander. 
    % Divide this torque (T_side) by the moment of inertia (I) to calculate the angular acceleration (alpha).
    ddtheta = T_side / I;
    
    % --- 5. Return State Derivative ---
    dxdt = [dx; dy; ddx; ddy; dtheta; ddtheta];
end