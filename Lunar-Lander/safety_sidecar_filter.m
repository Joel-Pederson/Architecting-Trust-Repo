function [u_actual, VetoTriggered, h_alt, h_fuel] = safety_sidecar_filter(x, u_nominal, params)
    % SAFETY_SIDECAR_FILTER Acts as a dynamic Control Barrier Function (CBF).
    % Filters the AI's action through Altitude and Fuel safety bounds.
    
    % --- 1. Unpack State & Parameters ---
    y_pos   = x(2);
    dy      = x(4);
    m_fuel  = x(7); % Current fuel mass in kg
    
    % The total mass of the lander changes as fuel burns!
    m_dry   = params.dry_mass; 
    m_total = m_dry + m_fuel; 
    
    g       = params.gravity;
    T_max   = params.max_main_thrust; 
    mdot    = params.max_mass_burn_rate; % kg/s burned at 100% thrust
    
    % Initialize outputs
    u_actual = u_nominal;
    VetoTriggered = false;
    
    % --- 2. Altitude Barrier: Calculate Minimum Required Thrust ---
    % "The Action Governor shall assume control authority from the Primary
    % AI Agent when the current altitude is less than or equal to d_stop + 1.5 meters."
    % (Note: This continuous filter satisfies this requirement by calculating the minimum
    % thrust necessary to guarantee this 1.5m buffer is never breached).
    
    safety_buffer_alt = 1.5; % Target hover height (meters)
    T_req_alt = 0;           % Default to 0 required thrust
    
    if dy < 0 
        distance_left = y_pos - safety_buffer_alt;
        
        % Step A: Calculate the absolute maximum braking capability
        a_max = (T_max / m_total) - g; 
        
        if a_max > 0
            % Step B: How much physical distance do we need to stop at 100% thrust?
            d_min_stop = (dy^2) / (2 * a_max);
            
            % Step C: EMERGENCY TRIGGER
            % Only intervene if we are crossing the absolute minimum stopping boundary
            if distance_left <= d_min_stop
                if distance_left > 0
                    % Calculate thrust needed to stop in exactly the distance remaining
                    a_req = (dy^2) / (2 * distance_left);
                    T_req_alt = m_total * (a_req + g);
                else
                    % Below buffer, full panic
                    T_req_alt = T_max; 
                end
            end
        else
            % If gravity is stronger than our max thrust, we are doomed. Panic fire.
            T_req_alt = T_max;
        end
    end
    
    % --- 3. Fuel Barrier: Calculate "Bingo" Fuel Thrust ---
    
    % Calculate a dynamic safety buffer based on "Time to Hover"
    % "The System shall continuously calculate the emergency fuel reserve required to arrest the current vertical velocity and 
    % maintain a 1.0g hover for a duration of 3.0 seconds."
    hover_time_reserve = 3.0; % 3 seconds of emergency hover fuel per requirement
    thrust_to_hover = m_total * g; % F = mg
    
    % Assuming fuel consumption scales linearly with thrust:
    burn_rate_at_hover = mdot * (thrust_to_hover / T_max);
    safety_buffer_fuel = burn_rate_at_hover * hover_time_reserve; 
    
    T_req_fuel = 0;
    fuel_needed_to_stop = 0; % <--- THE FIX: Guarantee the variable exists
    
    if dy < 0
        % How fast can spacecraft physically stop if we floor it?
        a_max = (T_max / m_total) - g;
        t_stop = abs(dy) / a_max; % Time to stop
        
        % How much fuel will that emergency stop cost?
        fuel_needed_to_stop = mdot * t_stop;
        
        % If spacecraft is dangerously close to not having enough fuel to brake,
        % force the AI to execute the suicide burn now.
        if m_fuel <= (fuel_needed_to_stop + safety_buffer_fuel)
            T_req_fuel = T_max;
        end
    end
    
    % --- 4. Action Filter ---
    
    % Physics floor: What is the absolute minimum thrust we need to survive?
    % Take the most restrictive requirement between Altitude and Fuel.
    T_lower_bound = max(T_req_alt, T_req_fuel);
    
    % The hardware ceiling: We cannot physically fire harder than the engine allows.
    T_upper_bound = T_max;
    
    % --- THE FIX: The Sidecar must also obey the laws of physics ---
    % Do not allow the lower bound safety net to exceed the engine's maximum capability
    T_lower_bound = min(T_lower_bound, T_upper_bound);
    
    % Clamp the final requested thrust within the bounds of reality
    u_actual(1) = max(T_lower_bound, min(u_nominal(1), T_upper_bound));
    
    % --- 5. LOGGING & STATE AUGMENTATION ---
    
    % Check if the Sidecar had to alter the AI's command
    % (Adding a small tolerance of 0.1N to account for floating point math)
    if (u_actual(1) - u_nominal(1)) > 0.1 
        VetoTriggered = true;
    end
    
    % Export the barrier values so we can feed them back into the AI's state
    % (This gives the AI "eyes" to see the boundary approaching)
    h_alt  = y_pos - ((dy^2) / (2 * ((T_max / m_total) - g)));
    h_fuel = m_fuel - (fuel_needed_to_stop + safety_buffer_fuel);
end