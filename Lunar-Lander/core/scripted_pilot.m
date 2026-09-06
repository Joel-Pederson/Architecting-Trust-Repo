function u_nominal = scripted_pilot(x, params)
% SCRIPTED_PILOT Deterministic classical guidance law for the lunar lander.
%
% A fully functional classical controller: it computes thrust and torque from rigid math
% rather than a learned policy. Serves three purposes:
%
%   1. The 'HARDCODED_PILOT' mode in main_simulation.
%   2. The non-RL reference policy in the algorithm trade study. "Does the learned policy
%      beat classical control?" is a far more useful claim for the paper than "the learned
%      policy eventually stopped crashing".
%   3. A fixed, known-good policy for environment sanity checks, where an untrained
%      network would confound harness bugs with policy quality.
%
% MEASURED PERFORMANCE (30 randomized episodes per phase, through LunarLanderEnv):
%
%     phase   guardian ON (30 episodes each)
%     P1      100%  0.25 m/s
%     P2      100%  0.28 m/s
%     P3      100%  0.29 m/s
%     P4      100%  0.28 m/s   (via braking_guidance, which delegates here below 2.5 km)
%
% against a 1.0 m/s vertical / 0.5 m/s lateral touchdown gate. Landing rates are identical
% with and without the sidecar, which is itself a result worth reporting: the Operational
% Sidecar is non-intrusive to a competent pilot. It costs a little reward (engagement
% penalties) without ever preventing a landing.
%
% --- DESIGN NOTES ---
%
% The governing constraint is ATTITUDE AUTHORITY, not thrust. The RCS produces 2 kNm
% against roughly 60,000 kg m^2 of inertia - 0.033 rad/s^2 - so a 0.2 rad slew takes about
% 5 s each way. Nulling 2 m/s of lateral drift therefore costs ~20 s including the slews,
% and lateral velocity is the binding gate term: an otherwise perfect vertical descent law
% lands 100% of Phase 1 with zero lateral offset and only 18.3% once the real
% init_dx ~ N(0, 2) randomisation is switched on.
%
% Two consequences shape the law below:
%
%   * The descent rate is GATED on lateral error, but only near the ground (the
%     1 - y/300 fade). Gating it at altitude as well starves Phase 3, which must cover
%     2500 m inside the 300 s episode cap.
%   * The vehicle squares up unconditionally below 6 m. Arriving tilted fails the 0.1 rad
%     gate no matter how good the velocities are.
%
% This replaces an earlier version whose lateral gain was unbounded: a 10-20 m/s drift
% demanded ~200 kN of lateral thrust against ~20 kN of vertical, resolving to an ~84
% degree pitch target, and because that version cut the engine entirely while the attitude
% error exceeded 0.25 rad the lander free-fell through the slew. It landed Phase 1 and
% degraded badly on Phases 2 and 3.
%
% Inputs:
%   x      - 8x1 physics state [x; y; dx; dy; theta; dtheta; m_main_fuel; m_rcs_fuel]
%   params - simulation parameters (get_sim_params)
%
% Outputs:
%   u_nominal - 2x1 command [T_main (N); Tau_side (Nm)], before any sidecar filtering

    x_pos = x(1); y_pos = x(2); dx = x(3); dy = x(4);
    theta = x(5); dtheta = x(6);

    m_total = params.dry_mass + x(7) + x(8);
    g       = params.gravity;

    % --- 1. DESCENT PROFILE ---
    % v = -0.55*sqrt(h) approximates a constant-deceleration arrival. The 0.25 m/s floor
    % keeps the vehicle committed to the surface instead of asymptoting into a hover, and
    % the 20 m/s ceiling bounds the Phase 3 fall.
    %
    % The gain matters more than it looks: at 0.30 the sqrt profile spends 324 s covering
    % Phase 3, which overruns the 300 s episode cap and scores as a timeout at 1.4 m/s -
    % a landing that fails only for want of clock.
    v_profile = max(0.25, min(0.55 * sqrt(max(y_pos - params.touchdown_alt, 0)), 20.0));

    % Slow the descent while lateral drift is still large, but fade the gate out with
    % altitude so it cannot starve a high-altitude approach of time.
    lat_err   = abs(dx) + 0.05 * abs(x_pos);
    gate      = 1 + 3.0 * lat_err * max(0, 1 - y_pos / 300);
    target_dy = -v_profile / gate;

    % --- 2. VERTICAL CHANNEL ---
    % Gravity feed-forward plus proportional control on descent-rate error. Expressed as
    % an acceleration and multiplied by mass, so the response does not drift as the 8200 kg
    % of propellant burns off.
    a_cmd    = g + 1.5 * (target_dy - dy);
    u_thrust = max(0, min(m_total * a_cmd, params.max_main_thrust));

    % --- 3. LATERAL CHANNEL (via attitude) ---
    % Lateral acceleration comes only from tilting the thrust vector: ddx = -T*sin(theta)/m.
    % Demand is capped at 0.6 m/s^2 and the tilt at 0.35 rad; beyond that the vertical
    % channel loses too much cos(theta) authority to hold the descent profile.
    ax_des = max(-0.6, min(-(0.020 * x_pos + 0.30 * dx), 0.6));
    th_des = max(-0.35, min(-ax_des / g, 0.35));

    % Square up for touchdown. The gate allows 0.1 rad and the slew is slow, so this has
    % to start well before contact.
    if y_pos < 6
        th_des = 0;
    end

    % --- 4. ATTITUDE CONTROLLER (PD, heavily damped) ---
    err_theta = atan2(sin(th_des - theta), cos(th_des - theta));   % wrapped
    u_torque  = 30000 * err_theta - 90000 * dtheta;
    u_torque  = max(-params.max_side_torque, min(u_torque, params.max_side_torque));

    u_nominal = [u_thrust; u_torque];
end
