function [u_actual, VetoTriggered, h_alt, h_fuel] = safety_sidecar_filter(x, u_nominal, params)
    % SAFETY_SIDECAR_FILTER: Deterministic Control Barrier Function (CBF)
    % 
    % --- ARCHITECTURE OVERVIEW ---
    % This function serves as the "Action Governor" in a Simplex Architecture. 
    % It acts as a safety net wrapped around the primary AI agent.
    % While the Agent is treated as an untrusted "black box", 
    % this sidecar uses strictly deterministic, formal Newtonian physics to protect the spacecraft.
    % 
    % It operates statelessly on a microsecond basis, intercepting the AI's requested 
    % actions (u_nominal) and passing them through three distinct survival filters:
    % 1. Altitude (Crash Prevention)
    % 2. Rotation (Pitch Control)
    % 3. Fuel (Bingo Fuel Prevention)
    % 
    % If the AI's request is safe, it passes through untouched. If the request is lethal,
    % the Sidecar vetoes it and injects an emergency survival command.
    
    % --- 1. UNPACK STATE & PARAMETERS ---
    % The sidecar evaluates the exact physical reality of the craft at this exact millisecond.
    dx      = x(3); % Horizontal velocity
    y_pos   = x(2); % Current altitude (meters)
    dy      = x(4); % Current vertical velocity (m/s). Negative means falling.
    theta   = atan2(sin(x(5)), cos(x(5))); % Current pitch angle strictly wrapped to [-pi, pi]
    dtheta  = x(6); % Current angular velocity (rad/s)
    m_main_fuel = x(7); % Current main fuel mass (kg)
    m_rcs_fuel  = x(8); % Current RCS propellant mass (kg)
    
    % Recalculate total mass (Mass dynamically changes as fuel is burned)
    m_dry   = params.dry_mass; 
    m_total = m_dry + m_main_fuel + m_rcs_fuel; 
    
    % Unpack engine and physics constants
    g       = params.gravity;
    r_lunar = params.r_lunar;
    T_max   = params.max_main_thrust; 
    Tau_max = params.max_side_torque;
    mdot    = params.max_mass_burn_rate; 
    
    % --- 1b. HIGHER FIDELITY PHYSICS ---
    % Calculate Centrifugal Lift (orbital mechanics)
    a_centrifugal = (dx^2) / (r_lunar + y_pos);
    g_apparent = max(0, g - a_centrifugal); % Apparent gravity is reduced by orbital velocity
    
    % Initialize the output to assume the AI is safe, until proven otherwise.
    u_actual = u_nominal;
    VetoTriggered = false;
    
    % --- 1c. ATTITUDE RECOVERY COST (shared by every barrier below) ---
    % Three separate barriers need to know what recovering from the current tilt costs,
    % so it is computed ONCE here. Previously each barrier asked "can it brake while
    % tilted like this" via T_max*cos(theta), which goes NEGATIVE past 90 degrees - so the
    % altitude barrier commanded full thrust while inverted, and the fuel barrier declared
    % that no amount of propellant could arrest the descent. Both fired constantly during
    % a powered descent, which is flown deliberately near 90 degrees.
    %
    % The right question is what it costs to RECOVER and then brake: slewing upright takes
    % a bang-bang 2*sqrt(theta/alpha), during which the engine cannot fight gravity, so
    % the vehicle loses altitude and gains descent rate.
    fuel_ratio_now = (m_main_fuel + m_rcs_fuel) / ...
                     (params.max_main_fuel + params.max_rcs_fuel);
    I_now     = params.inertia_dry + fuel_ratio_now * params.inertia_fuel_full;
    alpha_rec = Tau_max / max(I_now, 1);
    t_slew    = 2 * sqrt(max(abs(theta), 1e-6) / alpha_rec);
    drop_slew = abs(dy) * t_slew + 0.5 * g * t_slew^2;   % altitude spent recovering
    dy_after  = abs(dy) + g * t_slew;                    % descent rate once recovered

    % --- 2. ALTITUDE BARRIER (CRASH PREVENTION) ---
    % "The Action Governor shall assume control authority from the Primary
    % AI Agent when the current altitude is less than or equal to d_stop + safety_buffer_alt."
    
    % The safety buffer defines a "Hover Height" above the actual ground (y=0). 
    % This ensures the sidecar catches the ship and stabilizes it in the air, 
    % rather than trying to stop exactly at the millimeter the landing gear hits the dirt.
    safety_buffer_alt = 1.5; % Target hover height (meters)
    T_req_alt = 0;           % Default to 0 required emergency thrust
    
    % How much physical distance exists between the ship and the hover floor?
    distance_left = y_pos - safety_buffer_alt;
    % MARGIN SENTINELS ARE FINITE ON PURPOSE.
    % These two defaults used to be +Inf and -Inf. Both are only ever consumed by
    % comparisons, never by arithmetic, so realmax is exactly equivalent - and it keeps
    % IEEE special-case handling out of the generated flight code entirely. See
    % codegen/README.md: with any Inf in this function, MATLAB Coder must emit
    % rt_nonfinite.c, rtGetInf.c and rtGetNaN.c alongside the barrier, and the artefact
    % stops being straight-line arithmetic. Flight-software review generally objects to
    % Inf and NaN in the first place.
    MARGIN_UNBOUNDED = realmax;    % not descending: no braking boundary applies
    MARGIN_BREACHED  = -realmax;   % cannot arrest at any throttle: always inside the wall

    margin = MARGIN_UNBOUNDED; % Default: no braking boundary applies unless falling
    blending_zone = 5; % Default warning envelope width (meters), rescaled below when falling

    if dy < -0.5 % ACTIVE BRAKING: The ship is falling fast enough to warrant evaluation.
        
        % Braking capability is evaluated for the RECOVERED attitude, not the current one,
        % with the cost of recovering charged separately.
        %
        % The original form was a_max = T_max*cos(theta)/m - g_apparent, which is correct
        % only while the vehicle stays tilted as it is. Past 90 degrees cos(theta) goes
        % NEGATIVE, so the barrier concluded the vehicle could not brake at all and
        % commanded full thrust - which, inverted, accelerates it toward the ground. That
        % never mattered while a separate rule capped tilt at 45 degrees, and became live
        % the moment a powered descent needed to point retrograde. Measured: the barrier
        % forced full thrust at 110 degrees of tilt at 15 km.
        %
        % The honest question is not "can it stop while tilted like this" but "can it stop
        % after recovering", so: charge the altitude lost during the slew, then brake
        % upright with the speed that slew leaves behind.
        a_max = (T_max / m_total) - g_apparent;   % upright, so always the true authority

        if a_max > 0
            % Distance to stop = altitude spent recovering attitude, then the braking
            % distance at the speed that recovery leaves the vehicle carrying.
            d_min_stop = drop_slew + (dy_after^2) / (2 * a_max);
            
            % Margin is the "slack" in the system.
            % If margin == 0, the ship is at the exact point of no return.
            margin = distance_left - d_min_stop;
            blending_zone = max(5, 0.3 * d_min_stop);
        else
            % CRITICAL SCENARIO: Gravity is currently stronger than the available vertical thrust capability.
            % This happens if the ship is tilted too far (e.g. 90 degrees), or if the engine is too weak.
            % It is impossible to stop falling under these conditions, so the vehicle is
            % unconditionally inside the barrier. margin is only compared, never used in
            % arithmetic, so the finite sentinel is exact.
            margin = MARGIN_BREACHED;
        end
        
        % The Blending Zone (set above) prevents violent, structural-damaging binary
        % switching. Instead of waiting until margin == 0 and slamming the throttle from
        % 0% to 100%, the sidecar smoothly ramps up its authority as the boundary nears.
        %
        % The zone is scaled to the ship's actual braking distance rather than being a
        % fixed width. A fixed 50 m window is catastrophic at low speed: on a gentle
        % 2 m/s descent the braking distance is ~1 m, so a 50 m window means the sidecar
        % holds authority continuously below 51 m altitude and the agent never flies the
        % approach itself. Scaling with d_min_stop keeps the barrier proportionate:
        % wide during a fast 25 m/s descent, narrow during a slow terminal hover.
        if margin <= 0
            % POINT OF NO RETURN: Spacecraft has pierced the mathematical boundary. 
            % Absolute maximum panic effort. The AI is entirely locked out of the throttle.
            T_req_alt = T_max;
        elseif margin < blending_zone
            % BLENDING ZONE: The ship is inside the warning envelope. 
            % Ramp up minimum thrust smoothly as it approaches margin == 0.
            ramp_factor = 1 - (margin / blending_zone);
            T_req_alt = T_max * ramp_factor;
        end
        
    elseif distance_left <= 0.5
        % HOVER MODE: Spacecraft has arrived at the safety buffer and arrested its fall.
        % To prevent bouncing, output exactly enough thrust to counteract apparent gravity (F = mg).
        T_req_alt = m_total * g_apparent;
    end
    
    % --- 3. ROTATIONAL CONTROL BARRIER FUNCTION (ACTION GOVERNOR) ---
    % "The RCS side thrusters shall be seized by the action governor to force theta to 0 if the ship exceeds safe bounds, OR if it is in the altitude danger zone."
    
    % --- THE TILT LIMIT IS EARNED, NOT FIXED ---
    % A 45 degree ceiling is correct for a terminal descent and makes a powered descent
    % impossible: braking off orbital velocity requires pointing the engine retrograde, a
    % pitch approaching 90 degrees, sustained for minutes. Measured on identical initial
    % conditions from 15.2 km and 1697 m/s: unguarded the vehicle lands at 0.29 m/s, and
    % with a fixed 45 degree envelope it never gets below 12.5 km and is still doing
    % 549 m/s when the clock expires.
    %
    % The barrier therefore decides the limit from PHYSICS IT MEASURES ITSELF, rather than
    % from a flight phase the controller declares. That distinction matters: a barrier
    % that trusts the component it exists to police is not independent of it, and mode
    % confusion in the nominal controller would silently widen the safety envelope.
    %
    % The question a tilt limit is really asking is not "how far over is it" but "can the
    % recovery be afforded". Recovering from tilt theta means slewing back upright, which
    % the RCS does at alpha = Tau_max / I in a bang-bang time of 2*sqrt(theta/alpha).
    % Through that slew the engine cannot arrest the descent, so the vehicle loses
    %
    %     |dy| * t_rec + 0.5 * g * t_rec^2
    %
    % of altitude. If h_alt - the margin beyond the stopping distance, already computed
    % above - covers that loss with margin, the tilt is recoverable and permitted. Near
    % the ground it never does, so the terminal limit re-emerges on its own rather than
    % being special-cased.
    RECOVERY_MARGIN = 2.0;     % require twice the altitude the slew actually costs

    % Altitude margin beyond the stopping distance, evaluated UPRIGHT: the question is
    % what the vehicle could do once recovered, not what it can do while still tilted.
    a_max_upright_now = (T_max / m_total) - g_apparent;
    h_alt_now = y_pos - (dy^2) / (2 * max(0.1, a_max_upright_now));
    tilt_is_affordable = h_alt_now > RECOVERY_MARGIN * drop_slew;

    % Hard structural ceiling, enforced whatever the altitude margin says. Past this the
    % vehicle is tumbling rather than manoeuvring.
    max_pitch_structural = 2.44;   % ~140 degrees
    if tilt_is_affordable
        max_pitch = max_pitch_structural;
    else
        max_pitch = pi/4;      % 45 degrees - the terminal-descent envelope
    end
    max_pitch_braking = 0.35;  % ~20 degrees - tighter limit while inside the braking envelope

    % The barrier fires in two cases:
    %   1. The ship exceeds 45 degrees of tilt anywhere in the flight envelope. Past this
    %      point cos(theta) has eaten enough of the main engine's vertical component that
    %      recovery authority is genuinely at risk.
    %   2. The ship is inside the altitude braking envelope AND tilted past 20 degrees,
    %      where wasted vertical thrust directly threatens the stopping distance.
    %
    % It deliberately does NOT fire merely because the ship is in the braking envelope.
    % Seizing the RCS for the whole terminal descent pins theta at 0, and since
    % ddx = -T*sin(theta)/m, that freezes horizontal velocity at whatever it was when the
    % barrier engaged. With a touchdown limit of 0.5 m/s lateral and approach drift of
    % 10-20 m/s, that made a safe landing physically unreachable - the agent was being
    % asked to null drift with the only actuator that can do it taken away. The agent now
    % keeps torque authority to fly the approach, and the barrier intervenes only when
    % attitude itself becomes the hazard.
    if abs(theta) > max_pitch || (dy < -0.5 && margin < blending_zone && abs(theta) > max_pitch_braking)
        % High-gain PD controller to aggressively torque the ship to vertical
        % Gains anchored to the TERMINAL envelope, not the currently permitted one. Tying
        % them to a variable max_pitch would weaken the recovery torque precisely when a
        % wide envelope had been granted.
        kp = Tau_max / (pi/4);
        kd = Tau_max / (pi/4);
        
        Tau_req = -kp * theta - kd * dtheta;
        
        % Seize control of the RCS thrusters (u_actual(2))
        u_actual(2) = max(-Tau_max, min(Tau_req, Tau_max));
    end

    % --- 4. FUEL BARRIER (BINGO FUEL) ---
    % "The System shall continuously calculate the emergency fuel reserve required to arrest the current vertical velocity and 
    % maintain a 1.0g hover for a duration of at least 3.0 seconds."
    
    hover_time_reserve = 3.0; % 3 seconds of emergency hover fuel
    thrust_to_hover = m_total * g_apparent; % Thrust required to hover
    
    % Calculate fuel burn rate during hover (Linear scaling based on max flow rate)
    burn_rate_at_hover = mdot * (thrust_to_hover / T_max);
    safety_buffer_fuel = burn_rate_at_hover * hover_time_reserve; 
    
    T_req_fuel = 0;
    fuel_needed_to_stop = 0;
    
    if dy < 0
        % Evaluated UPRIGHT, with the recovery charged separately - the same correction
        % the altitude barrier needed, and for the same reason. Using cos(theta) here made
        % a_max negative past 90 degrees, so t_stop became infinite and the bingo-fuel
        % barrier concluded no amount of propellant could ever arrest the descent. During
        % a braking burn, which is flown deliberately near 90 degrees, that fired
        % constantly on a vehicle with 8 tonnes of usable propellant aboard.
        a_max = (T_max / m_total) - g_apparent;

        % Unlike the margin sentinels above, this one cannot simply be made finite: the
        % old t_stop = Inf was MULTIPLIED by mdot, and realmax * mdot overflows straight
        % back to Inf. So the "cannot arrest" case sets the fuel DEMAND directly instead
        % of routing an infinite time through a multiplication.
        if a_max <= 0
            % Genuinely cannot overcome gravity even pointed straight up, so no quantity
            % of propellant arrests this descent and the bingo condition is
            % unconditionally true. Any demand exceeding a full tank expresses that
            % exactly, because m_main_fuel can never exceed max_main_fuel.
            fuel_needed_to_stop = params.max_main_fuel + 1;
        else
            % Time to recover attitude, then to stop at the speed recovery leaves behind.
            t_stop = t_slew + (abs(dy) + g * t_slew) / a_max;

            % Calculate EXACTLY how much fuel will be consumed executing that stop
            fuel_needed_to_stop = mdot * t_stop;
        end
        
        % "Upon reaching bingo fuel level, the Action Governor shall immediately force a maximum-thrust suicide burn"
        % If the tank level drops to the exact amount of fuel required to stop + the 3-second reserve...
        if m_main_fuel <= (fuel_needed_to_stop + safety_buffer_fuel)
            % Force the AI into a "Suicide Burn" to land the ship NOW before it physically runs out of gas.
            T_req_fuel = T_max;
            
            % BUG FIX: We DO NOT zero out the torque here. The ship might be tilted!
            % We must allow the Rotational CBF to continue fighting to right the ship.
        end
    end
    
    % --- 4b. RCS FUEL BARRIER ---
    % If the sidecar or the AI depletes the RCS fuel, it is physically impossible to output torque.
    if m_rcs_fuel <= 0
        u_actual(2) = 0; % Override AI and Rotational CBF
    end
    
    % --- 5. ACTION FILTER & HARDWARE CLAMPS ---
    
    % Physics Floor: What is the absolute minimum thrust needed to survive this millisecond?
    % The most restrictive requirement between the Altitude CBF and the Fuel CBF is taken.
    T_lower_bound = max(T_req_alt, T_req_fuel);
    
    % Hardware Ceiling: The system cannot physically fire harder than the engine's mechanical limit.
    T_upper_bound = T_max;
    
    % Ensure the Sidecar obeys the laws of physics: Do not allow the safety net to demand more thrust than exists.
    T_lower_bound = min(T_lower_bound, T_upper_bound); % bare minimum threshold to maintain safe flight 
    
    % Combine everything: Allow the AI to command whatever it wants, AS LONG AS it is bounded 
    % between the Survival Floor (T_lower_bound) and the Hardware Ceiling (T_upper_bound).
    u_actual(1) = max(T_lower_bound, min(u_nominal(1), T_upper_bound));
    
    % --- 6. LOGGING & STATE AUGMENTATION ---
    
    % Check if the Sidecar had to MATERIALLY alter the AI's requested command.
    %
    % The tolerance is a fraction of actuator authority, not an absolute newton count. A
    % 0.1 N threshold on a 45 kN engine is a rounding error: it counted sub-newton
    % differences as safety interventions, so an agent flying along the thrust floor
    % registered dozens of "engagements" per descent. That inflates the veto metric the
    % A/B study reports on and makes the barrier look far twitchier than it is.
    thrust_tol = 0.005 * T_max;     % ~225 N of 45 kN
    torque_tol = 0.005 * Tau_max;   % ~10 Nm of 2 kNm

    if abs(u_actual(1) - u_nominal(1)) > thrust_tol || abs(u_actual(2) - u_nominal(2)) > torque_tol
        % This flag is sent back to the environment. The AI receives a large penalty 
        % every time this triggers. This teaches the AI to fear the boundaries and learn 
        % to fly so perfectly that the Sidecar never has to wake up.
        VetoTriggered = true;
    end
    
    % Export the continuous barrier values (h). 
    % By feeding these values directly into the neural network's observation state, 
    % the AI is given "eyes" to mathematically see the invisible boundaries approaching.
    a_max_upright = (T_max / m_total) - g_apparent;
    h_alt  = y_pos - ((dy^2) / (2 * max(0.1, a_max_upright)));
    h_fuel = m_main_fuel - (fuel_needed_to_stop + safety_buffer_fuel);
end