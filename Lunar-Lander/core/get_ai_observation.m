function norm_state = get_ai_observation(raw_state, ~)
% GET_AI_OBSERVATION Compresses true physics state vector into roughly [-1, 1] bounds
% so the Neural Network receives identically normalized inputs during training and deployment.
%
% Inputs:
%   raw_state - 8x1 vector: [x; y; dx; dy; theta; dtheta; m_main_fuel; m_rcs_fuel]
%   ~         - (Optional) params struct (ignored, kept for interface compatibility)
%
% Outputs:
%   norm_state - 8x1 normalized vector

    % Wrap angle theta to [-pi, pi] so it never explodes outside normalized bounds
    wrapped_theta = atan2(sin(raw_state(5)), cos(raw_state(5)));

    norm_state = [
        raw_state(1) / 500000;
        raw_state(2) / 20000;
        raw_state(3) / 2000;
        raw_state(4) / 150;
        wrapped_theta / pi;
        raw_state(6) / pi;
        raw_state(7) / 8200;
        raw_state(8) / 300
    ];
end
