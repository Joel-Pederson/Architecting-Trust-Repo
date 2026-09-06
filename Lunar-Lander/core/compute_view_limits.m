function [min_x, max_x, min_y, max_y] = compute_view_limits(x, y)
% COMPUTE_VIEW_LIMITS Axis bounds that fit a trajectory's actual flight envelope.
%
% Extracted from animate_lunar_lander so it can be regression-tested. The original code
% hard-coded floors of 15,000 m on the ceiling and 1,000 m of lateral padding, which meant
% a 2,500 m descent rendered inside the bottom sixth of the frame and a 50 m Phase 1
% approach was a single pixel above the surface - the "lander starts very low to the
% ground" symptom - regardless of what the lander actually did.
%
% Padding is a fraction of the trajectory's own extent, so the view scales with the
% flight rather than against a fixed constant.
%
% Inputs:
%   x, y - trajectory position vectors, TRUE physical units (metres)
%
% Outputs:
%   min_x, max_x, min_y, max_y - axis limits

    % max-min rather than range(): range() lives in the Statistics and Machine Learning
    % Toolbox, and the visualizer should not carry a toolbox dependency.
    % The floor of 100 m keeps a near-stationary trajectory from producing a zero-width
    % axis, which MATLAB rejects.
    x_span = max(max(x) - min(x), 100);
    y_span = max(max(y) - min(y), 100);

    min_x = min(x) - 0.1 * x_span;
    max_x = max(x) + 0.1 * x_span;
    min_y = min(-0.05 * y_span, min(y) - 0.05 * y_span);
    max_y = max(y) + 0.1 * y_span;
end
