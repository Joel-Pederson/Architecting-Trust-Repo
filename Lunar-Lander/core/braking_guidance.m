function [u_nominal, phase_name] = braking_guidance(x, params)
% BRAKING_GUIDANCE Apollo-style powered descent braking phase (the P63 equivalent).
%
% Spends orbital velocity and delivers the vehicle into the envelope that
% core/scripted_pilot.m already flies competently: roughly 2.5 km altitude, under 30 m/s,
% near the pad. It deliberately does NOT attempt the landing - that controller exists,
% is validated at 100/100/98.3%, and there is no reason to rewrite it.
%
% --- WHY A SEPARATE PHASE ---
% scripted_pilot flies a descent-rate profile with a lateral PD loop. That works from
% 2.5 km, where lateral error is metres and speed is tens of m/s. At powered descent
% initiation the vehicle is 410 km short of the pad doing 1697 m/s, and the problem is not
% "null a drift" but "spend orbital energy" - a different guidance problem, which is why
% Apollo split descent into braking and approach phases.
%
% --- THE THRUST BUDGET IS THE WHOLE PROBLEM ---
% At PDI the vehicle is effectively WEIGHTLESS: centrifugal acceleration at 1697 m/s
% cancels local gravity exactly, so 100% of thrust is available for braking. As speed
% bleeds off, gravity reasserts and claims a growing share:
%
%     1697 m/s -> 0% of thrust needed to hold altitude
%     1000 m/s -> 29%
%      300 m/s -> 44%
%       25 m/s -> 46%
%
% So the guidance commands an acceleration VECTOR and lets the required thrust and pitch
% fall out of it, rather than scheduling either directly.
%
% --- ATTITUDE ---
% Braking requires pointing the engine retrograde, which is a pitch approaching 90 deg -
% far outside the 45 deg the terminal-descent safety barrier permits. That is a genuine
% regime conflict, not a bug in either component, and is handled by making the barrier
% altitude-aware rather than by weakening the guidance.
%
% Inputs:
%   x      - 8x1 physics state [x; y; dx; dy; theta; dtheta; m_main_fuel; m_rcs_fuel]
%            x(1) is downrange ARC LENGTH, negative when short of the pad.
%   params - get_sim_params
%
% Outputs:
%   u_nominal  - 2x1 [T_main (N); Tau_side (Nm)]
%   phase_name - 'braking' | 'handoff'

    x_pos = x(1); y_pos = x(2); dx = x(3); dy = x(4);
    theta = x(5); dtheta = x(6);

    m_total = params.dry_mass + x(7) + x(8);
    r       = params.r_lunar + y_pos;

    % Handoff targets: the entry conditions scripted_pilot is known to fly well.
    Y_HANDOFF  = 2500;      % m
    V_HANDOFF  = 25;        % m/s horizontal
    X_HANDOFF  = -1500;     % m short of the pad, leaving the pilot a normal approach

    if y_pos <= Y_HANDOFF && abs(dx) <= 4 * V_HANDOFF
        u_nominal  = scripted_pilot(x, params);
        phase_name = 'handoff';
        return;
    end
    phase_name = 'braking';

    % --- 1. HORIZONTAL: spend the orbital velocity ---
    % Constant-deceleration solution for arriving at X_HANDOFF doing V_HANDOFF. Range is
    % clamped away from zero so the command stays finite if the vehicle overshoots.
    range_to_go = max(abs(X_HANDOFF - x_pos), 100);
    a_x_req = (V_HANDOFF^2 - dx^2) / (2 * range_to_go);     % negative while braking

    % --- 2. VERTICAL: hold an altitude schedule tied to REMAINING RANGE, not to time ---
    % Tying it to range keeps the profile stable if the braking runs early or late, which
    % a time-based schedule does not.
    total_range = max(abs(X_HANDOFF - (-params.pdi_downrange)), 1);
    frac        = min(1, max(0, range_to_go / total_range));
    y_target    = Y_HANDOFF + (params.pdi_altitude - Y_HANDOFF) * frac;

    % Descent rate that would reach the scheduled altitude over the remaining range.
    t_remaining = 2 * range_to_go / max(abs(dx) + V_HANDOFF, 1);
    dy_target   = (y_target - y_pos) / max(t_remaining, 1);
    dy_target   = max(-60, min(dy_target, 10));

    g_local       = params.gravity * (params.r_lunar / r)^2;
    a_centrifugal = dx^2 / r;
    g_apparent    = g_local - a_centrifugal;    % may be negative: centrifugal can lift

    a_y_req = g_apparent + 0.6 * (dy_target - dy);

    % --- 3. ACCELERATION VECTOR -> THRUST AND PITCH ---
    % ddx = -T*sin(theta)/m and ddy = T*cos(theta)/m, so the commanded direction inverts
    % directly. Pitch is limited to 89 deg: at exactly 90 the vertical component vanishes
    % and altitude control is lost.
    T_req     = m_total * hypot(a_x_req, a_y_req);
    theta_des = atan2(-a_x_req, max(a_y_req, 1e-6));
    theta_des = max(-1.553, min(theta_des, 1.553));

    u_thrust = max(0, min(T_req, params.max_main_thrust));

    % --- 4. ATTITUDE CONTROLLER ---
    % The RCS is weak against the vehicle's inertia (~0.033 rad/s^2), so a 90 deg slew
    % takes ~20 s. Gains are high because the torque saturates anyway; what matters is
    % that the sign is right and the damping prevents overshoot at the limit.
    err_theta = atan2(sin(theta_des - theta), cos(theta_des - theta));
    u_torque  = 60000 * err_theta - 120000 * dtheta;
    u_torque  = max(-params.max_side_torque, min(u_torque, params.max_side_torque));

    % --- 5. DO NOT THRUST WHILE BADLY MISPOINTED ---
    % Firing a 45 kN engine 90 deg away from the commanded direction accelerates the
    % vehicle the wrong way. Throttle back until the attitude is roughly achieved.
    if abs(err_theta) > 0.35
        u_thrust = u_thrust * max(0, 1 - (abs(err_theta) - 0.35) / 0.6);
    end

    u_nominal = [u_thrust; u_torque];
end
