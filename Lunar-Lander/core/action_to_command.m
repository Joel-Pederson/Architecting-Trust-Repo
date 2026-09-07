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
% It replaces a raw throttle map, u = (a+1)/2 * T_max, whose problem was not range but
% MASS DEPENDENCE: as 8200 kg of propellant burns off the same action goes from 1.76 to
% 5.26 m/s^2 per unit, a 3x drift in the plant the policy is learning against.
% Compensating makes the action->acceleration map invariant to fuel state.
%
% Torque maps linearly; the side thrusters are bidirectional.
%
% It lives in its own file because it was once inlined in LunarLanderEnv.step while
% main_simulation kept a copy of the OLD map - so an agent flown there drove a different
% plant than it trained on. Two definitions of a control interface is one too many.
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
    hover_T = min(m_total * params.gravity, params.max_main_thrust);

    % PIECEWISE, and the asymmetry is deliberate.
    %
    %   action <= 0 : 0 .. hover      - mass-invariant, action 0 is always exact hover
    %   action >  0 : hover .. T_max  - the FULL engine, whatever the vehicle weighs
    %
    % The original map was hover*(1+a) throughout, which caps thrust at twice the CURRENT
    % weight. That is ample for a terminal descent but throws the engine away during a
    % powered descent: as propellant burns the ceiling falls with the vehicle's mass, from
    % 92% of rated thrust at full tanks to 40% near empty - tightening precisely during a
    % braking burn, when both fuel consumption and thrust demand peak. Measured: the
    % braking phase completes through the raw dynamics and times out through this map.
    %
    % The cost is a slope discontinuity at action 0: the gain is hover_T per unit below it
    % and (T_max - hover_T) above. That is a gain schedule, not a nonlinearity a
    % controller cannot handle, and it buys back the whole upper envelope while keeping
    % the hover point fixed - which is the property that made terminal descent learnable.
    if action(1) <= 0
        u_thrust = hover_T * (1 + action(1));
    else
        u_thrust = hover_T + action(1) * (params.max_main_thrust - hover_T);
    end
    u_thrust = max(0, min(u_thrust, params.max_main_thrust));
    u_torque = max(-params.max_side_torque, ...
                min(action(2) * params.max_side_torque, params.max_side_torque));

    u_nominal = [u_thrust; u_torque];
end
