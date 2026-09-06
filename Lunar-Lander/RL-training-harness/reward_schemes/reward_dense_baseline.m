function [Reward, IsDone, Outcome] = reward_dense_baseline(x, x_prev, u_actual, u_prev, VetoEngaged, params, weights) %#ok<INUSD>
% REWARD_DENSE_BASELINE Calculates the continuous reinforcement learning score.
%
% Reward Philosophy (Dense Baseline):
%   - Dense Rewards (Calculated every physics substep):
%       * Fuel Penalty: scaled per SECOND of burn, so the cost of a manoeuvre does not
%         change if the integration step changes.
%
%   NOTE: potential-based shaping is NOT computed here. It is applied once per AGENT step
%   by LunarLanderEnv, in the discounted form gamma*Phi(s') - Phi(s) that policy
%   invariance actually requires. See shaping_potential.m.
%
%   - Event Rewards:
%       * Sidecar engagement: charged once per engagement, on the rising edge only.
%
%   - Sparse Rewards (Calculated at termination):
%       * Success:  +500 raw (soft touchdown within Apollo LM gear limits)
%       * Crash:    -500 raw scaled by impact severity
%       * OOB:      -500 raw
%
% Inputs:
%   x, x_prev   - 8x1 physics state after and before this substep. x_prev is retained
%                 for signature stability across reward schemes; this scheme no longer
%                 reads it now that shaping is applied at the agent step.
%   u_actual    - 2x1 control actually applied [T_main (N); Tau_side (Nm)]
%   u_prev      - 2x1 control applied on the previous substep (unused, kept for
%                 signature stability across reward schemes)
%   VetoEngaged - true ONLY on the substep where the sidecar newly took authority.
%                 Passing a level-triggered flag here reintroduces the per-step tax
%                 that made crashing optimal; see get_reward_weights for the history.
%   params      - simulation parameters (get_sim_params)
%   weights     - reward weights (get_reward_weights)
%
% Outputs:
%   Reward      - scalar reward for this substep, already divided by reward_scale
%   IsDone      - true if this substep ended the episode
%   Outcome     - 'flying' | 'landed' | 'crashed' | 'oob'

    % Unpack state variables
    x_pos = x(1);
    y_pos = x(2);
    dx = x(3);
    dy = x(4);
    theta = x(5);
    dtheta = x(6);

    % Initialize flags
    IsDone = false;
    Outcome = 'flying';
    Reward = 0;

    % Extract dynamic hardware limits from the central struct
    max_T   = params.max_main_thrust;
    max_Tau = params.max_side_torque;
    dt      = params.dt;

    % --- 1. SHAPING IS NOT COMPUTED HERE ---
    % Potential-based shaping moved to LunarLanderEnv.step, applied ONCE PER AGENT STEP.
    %
    % It used to be summed per physics substep as Phi(s') - Phi(s), with no discount.
    % That form is only policy-invariant at gamma = 1 (Ng et al. 1999 require
    % gamma*Phi(s') - Phi(s)), and the undiscounted residual over a 3000-step episode is
    % of order (1-gamma) * |Phi| * horizon ~ 5 reward - the same magnitude as the landing
    % bonus it was supposed to be neutral against.
    %
    % Discounting per substep does not fix it either: gamma_sub = gamma^(1/decimation)
    % applied five times does NOT telescope across the substeps, because the intermediate
    % terms no longer cancel. The only correct place to apply it is at the agent's own
    % decision rate, which is where it now lives. See shaping_potential.m.

    % --- 2. FUEL PENALTIES ---
    norm_T_main = abs(u_actual(1)) / max_T;
    norm_T_side = abs(u_actual(2)) / max_Tau;

    % Priced per second, then converted to this substep. Halving dt no longer doubles
    % the cost of the same physical burn.
    fuel_penalty = (weights.fuel_main * norm_T_main + weights.fuel_rcs * norm_T_side) * dt;

    % Sum the continuous rewards (shaping is added by the environment)
    Reward = Reward + fuel_penalty;

    % --- 3. THE SIDECAR ENGAGEMENT PENALTY ---
    % Rising edge only. Teaches the agent that tripping the barrier is costly, without
    % charging rent for every millisecond the guardian happens to be holding authority.
    if VetoEngaged
        Reward = Reward + weights.sidecar_veto;
    end

    % --- 4. TERMINAL CONDITIONS ---
    % Ground contact is evaluated at the touchdown altitude, which sits just above the
    % sidecar's 1.5 m hover floor so the agent is never required to punch through it.
    if y_pos <= params.touchdown_alt
        IsDone = true;

        % --- MISS RATIO ---
        % Each touchdown criterion expressed as a multiple of its own limit, so 1.0 is
        % exactly on the boundary. The GATE is unchanged - dy, dx and tilt, the Apollo LM
        % gear limits - so what counts as a safe landing is identical to before.
        gate_terms = [abs(dy)    / params.max_touchdown_dy, ...
                      abs(dx)    / params.max_touchdown_dx, ...
                      abs(theta) / params.max_touchdown_tilt];
        miss_gate = max(gate_terms);

        if miss_gate > 1.0
            Outcome = 'crashed';

            % --- WHY THIS IS GRADED, AND IN THREE BANDS ---
            % The gate above is unchanged. What is graded is how badly a failure missed,
            % and it has to be graded across the WHOLE range, because that is the signal
            % an agent follows on its way from "falls out of the sky" to "lands".
            %
            % The previous version had proximity = min(1, (miss-1)/3), saturating at 4x
            % the limit, plus severity = min(1, impact/50). Measured with a controller
            % blended toward free-fall: on Phase 3, braking from 25.06 to 19.18 m/s -
            % real progress - changed the episode reward by -0.02, because both terms
            % were pinned at their ceilings and the extra fuel cost cancelled what little
            % was left. All the credit sat in one step at the moment thrust first
            % exceeded weight.
            %
            % Shaping cannot fix this. Potential-based shaping is policy-invariant by
            % construction, so it telescopes to -Phi(s_0) and contributes nothing to
            % episode-level ordering. Only the terminal reward can grade the impact.
            impact_speed = sqrt(dx^2 + dy^2);

            % Spin is not part of the gate but does make a touchdown worse, so it counts
            % toward how badly the attempt missed.
            miss = max(miss_gate, abs(dtheta) / params.max_touchdown_spin);

            % NEAR: 1x -> 4x the gate. Steepest band; this is where landing is decided.
            f_near = min(1.0, max(0, miss - 1.0) / 3.0);

            % FAR: 4x -> crash_far_limit. The band that was missing entirely.
            f_far  = min(1.0, max(0, miss - 4.0) / (weights.crash_far_limit - 4.0));

            % SEV: raw closing speed, so a smash and an arrival stay distinguishable.
            f_sev  = min(1.0, impact_speed / weights.crash_sev_limit);

            Reward = Reward + weights.crash * ...
                (weights.crash_near_share * f_near + ...
                 weights.crash_far_share  * f_far  + ...
                 weights.crash_sev_share  * f_sev);

        else
            Outcome = 'landed';
            Reward = Reward + weights.success;
        end

    % Out-of-bounds terminal case (excess lateral/vertical displacement)
    elseif abs(x_pos) > params.max_abs_x || y_pos > params.max_alt
        IsDone = true;
        Outcome = 'oob';
        Reward = Reward + weights.oob;
    end

    % --- 5. NEURAL NETWORK SCALING ---
    % Deep networks suffer from exploding gradients if forced to regress large values.
    Reward = Reward / weights.reward_scale;
end
