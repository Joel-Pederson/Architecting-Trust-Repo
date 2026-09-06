function phi = shaping_potential(x, params, weights)
% SHAPING_POTENTIAL Potential function Phi(s) used for reward shaping. RAW units.
%
% Returns an UNSCALED potential; the caller divides by weights.reward_scale, exactly as
% the reward schemes do with their own terms.
%
% The potential has two terms on deliberately different scales:
%
%   DESCENT term  - guides the vehicle from anywhere in the flight box toward the pad.
%                   Normalised by the box (1000 m lateral, 3000 m vertical, 100 m/s), so
%                   it dominates while the lander is still high and fast.
%
%   APPROACH term - measures the touchdown criteria against THEIR OWN limits, faded in
%                   below params.approach_gate_alt.
%
% WHY THE APPROACH TERM EXISTS. With only the descent term, the whole span from a FAILING
% 2 m/s lateral drift to a PASSING 0.5 m/s was worth 0.06 reward against a +5.00 landing
% bonus, because velocity is normalised by 100 m/s. The agent got essentially no signal
% about the criterion that decides every episode. Measured: an ideal vertical descent law
% lands 100% of Phase 1 with zero lateral offset and 18.3% once init_dx ~ N(0,2) is
% switched on, matching P(|dx| < 0.5) for that distribution. Lateral velocity is the
% binding gate term and the reward was blind to it.
%
% POLICY INVARIANCE. Both terms are pure functions of state, so shaping built from this
% potential is policy-invariant (Ng et al. 1999) PROVIDED the caller applies the
% discounted form gamma*Phi(s') - Phi(s) and treats Phi at an absorbing state as 0.
% LunarLanderEnv does both; see its step().
%
% Inputs:
%   x       - 8x1 physics state [x; y; dx; dy; theta; dtheta; m_main_fuel; m_rcs_fuel]
%   params  - get_sim_params (supplies the caps, the gate altitude, touchdown limits)
%   weights - get_reward_weights (supplies shaping and approach gains)
%
% Output:
%   phi - scalar potential, raw units (divide by weights.reward_scale to compare with
%         the +/-500 terminal outcomes)

    k = weights.shaping;

    % --- DESCENT TERM ---
    norm_x  = x(1) / 1000;
    norm_y  = x(2) / 3000;
    norm_dx = x(3) / 100;
    norm_dy = x(4) / 100;

    dist  = min(sqrt(norm_x^2  + norm_y^2),  params.shaping_max_dist);
    speed = min(sqrt(norm_dx^2 + norm_dy^2), params.shaping_max_speed);
    tilt  = min(abs(x(5)),                   params.shaping_max_tilt);

    phi = -k * dist - k * speed - k * tilt;

    % --- FINAL-APPROACH TERM ---
    % Linear fade so the term never appears as a discontinuity the agent could exploit
    % by loitering just above the gate altitude.
    w_gate = max(0, 1 - x(2) / params.approach_gate_alt);
    if w_gate > 0
        miss = abs(x(3)) / params.max_touchdown_dx ...
             + abs(x(4)) / params.max_touchdown_dy ...
             + abs(x(5)) / params.max_touchdown_tilt;
        miss = min(miss, params.approach_max_miss);
        phi = phi - weights.approach * w_gate * miss;
    end
end
