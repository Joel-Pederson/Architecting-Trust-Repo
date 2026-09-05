function action = command_to_action(u_nominal, x, params)
% COMMAND_TO_ACTION Maps a physical command back into the agent's normalised action space.
%
% The exact inverse of ACTION_TO_COMMAND, including the [-1, 1] clip. Needed wherever a
% classical controller's output has to be fed through the RL action interface: the fault
% injection study, the scripted-pilot regression test, and demonstration generation.
%
% --- THE CLIP MATTERS ---
% Callers must use the CLIPPED action this returns, not the raw quotient, because the
% clipped value is what the environment actually receives. For behaviour cloning in
% particular, recording the unclipped quotient would train the network toward commands
% the plant cannot execute, so the targets would not match the trajectory that produced
% them.
%
% Note the clip is one-sided in practice on the thrust channel: thrust is non-negative,
% so a_thrust >= -1 always holds, and only the upper bound binds - whenever the
% controller asks for more than 2x hover thrust.
%
% Inputs:
%   u_nominal - 2x1 physical command [T_main (N); Tau_side (Nm)]
%   x         - 8x1 physics state; only the fuel masses x(7), x(8) are read
%   params    - get_sim_params
%
% Outputs:
%   action - 2x1 normalised action in [-1, 1]
%
% See also ACTION_TO_COMMAND.

    m_total = params.dry_mass + x(7) + x(8);
    hover_T = m_total * params.gravity;

    a_thrust = u_nominal(1) / hover_T - 1;
    a_torque = u_nominal(2) / params.max_side_torque;

    action = [max(-1, min(a_thrust, 1)); ...
              max(-1, min(a_torque, 1))];
end
