function u_nominal = action_to_command(action, x, params)
% ACTION_TO_COMMAND Maps a normalised agent action to a physical command.
%
% The single definition of the agent's control interface. Every caller that turns a
% [-1, 1] action into Newtons and Newton-metres must go through here.
%
% --- THE MAP ---
% Thrust is GRAVITY-COMPENSATED: action 0 commands exact hover, and +/-1 commands
% +/- one lunar g of NET vertical acceleration. This is standard descent-guidance
% practice (the Apollo LM PGNCS commanded acceleration above a gravity feedforward
% term) and re-parameterises the existing control authority rather than adding any.
%
% It replaces a raw throttle map, u = (a+1)/2 * T_max. The problem with raw throttle is
% not its range - measured, both maps span about the same net acceleration at full mass
% (1.76 vs 1.62 m/s^2 per action unit). It is that raw throttle's plant gain depends on
% MASS: as the 8200 kg of propellant burns off, the same action produces up to
% 5.26 m/s^2 per unit instead of 1.76, a 3x drift in the plant the policy is learning
% against. Compensating makes the action->acceleration map invariant to fuel state.
%
% Torque maps linearly; the side thrusters are bidirectional.
%
% --- WHY THIS IS A SEPARATE FUNCTION ---
% It was previously inlined in LunarLanderEnv.step, and main_simulation kept its own
% copy of the OLD raw-throttle map. An agent flown through main_simulation was therefore
% driving a different plant than it trained on - at full tanks, action 0 commanded
% 22.5 kN there against the environment's 20.3 kN hover, and the mass-invariance property
% was absent entirely. Two definitions of a control interface is one too many.
%
% Inputs:
%   action - 2x1 normalised action [thrust; torque], each in [-1, 1]
%   x      - 8x1 physics state; only the fuel masses x(7), x(8) are read
%   params - get_sim_params
%
% Outputs:
%   u_nominal - 2x1 physical command [T_main (N); Tau_side (Nm)], clamped to hardware
%
% See also COMMAND_TO_ACTION, the exact inverse.

    m_total = params.dry_mass + x(7) + x(8);
    hover_T = m_total * params.gravity;

    u_thrust = max(0, min(hover_T * (1 + action(1)), params.max_main_thrust));
    u_torque = max(-params.max_side_torque, ...
                min(action(2) * params.max_side_torque, params.max_side_torque));

    u_nominal = [u_thrust; u_torque];
end
