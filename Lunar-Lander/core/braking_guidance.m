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

    % Handoff requires the velocity scripted_pilot can actually absorb. At 4*V_HANDOFF
    % (100 m/s) the braking phase handed over while still doing 94 m/s, and that
    % controller caps lateral demand at 0.6 m/s^2 and gates its descent on lateral error -
    % so it hovered at 125 m indefinitely rather than landing. Braking keeps authority
    % until the velocity is genuinely inside the terminal envelope.
    if y_pos <= Y_HANDOFF && abs(dx) <= 1.5 * V_HANDOFF
        u_nominal  = scripted_pilot(x, params);
        phase_name = 'handoff';
        return;
    end
    phase_name = 'braking';

    % --- 1. HORIZONTAL: track a velocity profile, SIGNED ---
    % The earlier form solved (V^2 - dx^2)/(2*range) with range = abs(...), which cannot
    % represent an overshoot: once the vehicle passed the pad the guidance read the
    % growing distance as "still short", commanded a gentle deceleration and scheduled a
    % CLIMB back to 12 km. Measured, it flew 317 km beyond the target still doing 225 m/s,
    % ascending, until the clock ran out. An absolute value in a guidance law is almost
    % always a lost sign.
    %
    % A signed velocity profile handles both cases with one expression: approach the pad
    % at the speed from which the remaining distance can still be stopped, and if that
    % distance is behind you, the sign flips and the profile flies back.
    range_signed  = X_HANDOFF - x_pos;                  % positive while the pad is ahead

    % Horizontal braking authority is whatever thrust is left after holding altitude, and
    % that varies enormously through the burn: at PDI centrifugal acceleration cancels
    % gravity exactly, so the vehicle is weightless and the ENTIRE engine can brake.
    %
    % A flat "reserve 30% for vertical" makes the trajectory infeasible from the first
    % step: it yields 2.47 m/s^2, which needs 583 km to stop 1697 m/s against the 410 km
    % actually available, so the profile demands the impossible and the vehicle overshoots
    % and diverges. Measured that way: arriving at -701 m/s, having reversed direction.
    g_here   = params.gravity * (params.r_lunar / r)^2 - dx^2 / r;
    T_vert   = m_total * max(g_here, 0);                % thrust that must go upward
    T_horiz  = sqrt(max(params.max_main_thrust^2 - T_vert^2, 0));
    a_brake_avail = max(T_horiz / m_total, 0.1);
    % REQUIRED deceleration, not a velocity profile to track. This distinction cost a
    % couple of iterations: a sqrt(2*a*d) profile evaluates to 1696 m/s at PDI against an
    % actual 1697, concludes the vehicle is exactly on profile, and commands 0.5 m/s^2 of
    % braking. It then falls behind as gravity reasserts and never recovers. Solving for
    % the deceleration the remaining distance DEMANDS commands 3.51 m/s^2 from the first
    % step, which is what the manoeuvre actually needs.
    % ONE signed velocity profile covers approach and overshoot alike. Two branches - a
    % required-deceleration formula ahead of the pad and a velocity kill behind it - fought
    % each other: past the pad the kill term braked the RETURN, so the vehicle settled at
    % exactly the handoff threshold and chattered between modes 200 km downrange while its
    % altitude bled away.
    %
    % The profile is deliberately flown at 85% of available braking. At 100% it evaluates
    % to 1659 m/s at PDI against an actual 1697 - within noise of "already on profile" -
    % and commands almost no braking, then falls behind as gravity reasserts. The 15%
    % margin makes the profile bite from the first step and absorbs the cosine and slew
    % losses that no closed-form profile accounts for.
    % FEEDFORWARD the deceleration the remaining distance demands, then correct with
    % feedback against a velocity profile.
    %
    % Feedback alone does not work: a P-tracker needs an ERROR to command braking, so it
    % runs permanently ahead of its own profile. At PDI the profile caps at 1697 m/s
    % against an actual 1694.5, the controller reads a positive error and ACCELERATES, and
    % the vehicle overshot by exactly 200 km whether it started 460 or 550 km out - the
    % distance margin was irrelevant because nothing was braking.
    %
    % Feedforward alone was the original formula, which worked but could not express an
    % overshoot. Together: the feedforward term brakes from the first step, the feedback
    % term supplies the sign handling and absorbs modelling error.
    rng_mag = max(abs(range_signed), 100);
    dir_pad = sign(range_signed);           % +1 while the pad is ahead, -1 once past it
    if dir_pad == 0, dir_pad = 1; end

    % The stopping-distance formula -dx^2/(2s) is only valid while CLOSING on the target.
    % Applied while moving away from it the sign inverts and it accelerates the departure:
    % measured, arriving at -852 m/s after overshooting. So branch on whether the vehicle
    % is closing or opening, which is one comparison and removes the whole failure mode.
    if sign(dx) == dir_pad || dx == 0
        % Closing: decelerate to arrive at V_HANDOFF exactly as the range runs out.
        a_ff = -dir_pad * (dx^2 - V_HANDOFF^2) / (2 * rng_mag);
    else
        % Opening: nothing to schedule, just brake against the motion at full authority.
        a_ff = -sign(dx) * a_brake_avail;
    end

    % FEEDFORWARD ONLY. Adding a velocity-profile feedback term actively hurt: the
    % profile's allowance at 550 km is 1811 m/s against an actual 1694, so the feedback
    % read spare margin and commanded +0.75 m/s^2 AGAINST the feedforward's -2.62,
    % achieving only -1.87 and using 13-30 kN of a 45 kN engine. The vehicle then arrived
    % at the pad still doing 891 m/s. Worse, once past the target the same profile
    % authorised a 1037 m/s return leg and it accelerated to -852 m/s.
    %
    % The two terms answer different questions - "arrive at V_HANDOFF" versus "be able to
    % stop from here" - and summing them lets the laxer one erode the stricter. The
    % feedforward alone is the requirement, and the closing/opening branch above supplies
    % the sign handling that was the only thing it ever lacked.
    a_x_req = max(-a_brake_avail, min(a_ff, a_brake_avail));
    range_to_go = max(abs(range_signed), 100);

    % --- 2. VERTICAL: hold an altitude schedule tied to REMAINING RANGE, not to time ---
    % Tying it to range keeps the profile stable if the braking runs early or late, which
    % a time-based schedule does not.
    total_range = max(abs(X_HANDOFF - (-params.pdi_downrange)), 1);
    frac        = min(1, max(0, range_to_go / total_range));
    y_target    = Y_HANDOFF + (params.pdi_altitude - Y_HANDOFF) * frac;

    % Never schedule a CLIMB. The profile exists to bring the vehicle down along a
    % descent corridor; if it is already below the corridor the correct action is to hold,
    % not to spend propellant regaining altitude it is about to give back.
    y_target    = min(y_target, y_pos);

    % Descent rate that would reach the scheduled altitude over the remaining range.
    t_remaining = 2 * range_to_go / max(abs(dx) + V_HANDOFF, 1);
    dy_target   = (y_target - y_pos) / max(t_remaining, 1);
    dy_target   = max(-60, min(dy_target, 10));

    g_apparent = g_here;    % computed above, where the braking budget needed it

    a_y_req = g_apparent + 0.6 * (dy_target - dy);

    % --- 3. ACCELERATION VECTOR -> THRUST AND PITCH ---
    % ddx = -T*sin(theta)/m and ddy = T*cos(theta)/m, so the commanded direction inverts
    % directly. Pitch is limited to 89 deg: at exactly 90 the vertical component vanishes
    % and altitude control is lost.
    % The engine can only PUSH along the body axis, so a demand for net DOWNWARD
    % acceleration beyond what gravity already supplies is not achievable - the best
    % available is to point fully horizontal and let the vehicle fall at g while braking.
    % Clamping the vertical demand at zero for BOTH the magnitude and the angle expresses
    % that. Using the raw negative value in the magnitude commanded large thrust at 89
    % degrees, which brakes hard and refuses to descend: measured, the vehicle stalled at
    % 6 km with dy = +2.46 m/s, climbing, and ran out of clock still doing 224 m/s.
    a_y_cmd   = max(a_y_req, 0);
    T_req     = m_total * hypot(a_x_req, a_y_cmd);
    theta_des = atan2(-a_x_req, max(a_y_cmd, 1e-6));
    theta_des = max(-1.553, min(theta_des, 1.553));

    u_thrust = max(0, min(T_req, params.max_main_thrust));

    % --- 4. ATTITUDE CONTROLLER (deliberately UNSATURATED) ---
    % Gains are sized so the command stays inside the RCS limit for the attitude errors
    % actually seen in flight, rather than slamming between the stops.
    %
    % This is a DEMONSTRATOR, and it will be fitted by least-squares regression. The
    % earlier gains (60000 against a 2000 Nm limit) saturated for any error beyond
    % 0.033 rad, making the torque signal bang-bang: +1, -1, +1, flipping constantly.
    % Mean-squared error fits the CONDITIONAL MEAN of that, which is approximately zero,
    % so the cloned network learned to command no torque at all - it never held the
    % retrograde attitude and never braked. Measured: expert torque +/-1.000 against a
    % cloned output of 0.028, and 0% of powered descents landed.
    %
    % That is the classic behaviour-cloning failure on multimodal actions: the mean of two
    % valid commands is not itself a valid command. The fix is not a larger network, it is
    % a demonstrator whose actions are a smooth function of state.
    %
    % kp is chosen so a 0.4 rad error uses about half the available torque, keeping the
    % command in the linear region through a normal slew. The slew is torque-limited by
    % physics regardless, so softer gains cost almost nothing in performance.
    err_theta = atan2(sin(theta_des - theta), cos(theta_des - theta));
    % SMOOTH SATURATION. The discontinuity that defeated cloning came from the hard
    % clamp, not from the gain: with kp = 60000 against a 2000 Nm limit the command was
    % pinned at +/-1 with a transition region 0.03 rad wide, which a smooth network cannot
    % represent and averages to approximately zero. Softening the gains instead made the
    % command learnable but the attitude lag cost 46% of commanded braking and overshot
    % the pad by 200 km.
    %
    % tanh keeps the full authority of a stiff controller while making the command a
    % smooth function of state, with a transition ~0.2 rad wide that a network can fit.
    kp_att = 12000;
    kd_att = 30000;
    u_torque = params.max_side_torque * ...
               tanh((kp_att * err_theta - kd_att * dtheta) / params.max_side_torque);

    % --- 5. DO NOT THRUST WHILE BADLY MISPOINTED ---
    % Firing a 45 kN engine 90 deg away from the commanded direction accelerates the
    % vehicle the wrong way. Throttle back until the attitude is roughly achieved.
    if abs(err_theta) > 0.35
        u_thrust = u_thrust * max(0, 1 - (abs(err_theta) - 0.35) / 0.6);
    end

    u_nominal = [u_thrust; u_torque];
end
