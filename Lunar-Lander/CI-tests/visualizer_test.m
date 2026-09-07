function tests = visualizer_test
%VISUALIZER_TEST - Regression guards for the telemetry view scaling.
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
end

function testTrajectoryFillsTheFrame(testCase)
    % The original code hard-coded a 15,000 m ceiling, so a 2,500 m descent occupied the
    % bottom sixth of the plot and looked like it never left the ground. The vertical
    % extent of the flight must occupy most of the vertical axis.
    y = linspace(2500, 2, 500);
    x = linspace(0, 300, 500);

    [~, ~, min_y, max_y] = compute_view_limits(x, y);

    frame_height = max_y - min_y;
    flight_height = max(y) - min(y);
    fill_fraction = flight_height / frame_height;

    verifyGreaterThan(testCase, fill_fraction, 0.7, ...
        sprintf(['Descent fills only %.0f%% of the vertical axis. A hard-coded axis ' ...
                 'floor has been reintroduced.'], 100 * fill_fraction));
end

function testLowAltitudeApproachIsVisible(testCase)
    % A 50 m Phase 1 approach must also fill the frame, not sit as one pixel above the
    % surface. This is the case the fixed 15,000 m ceiling broke most severely.
    y = linspace(50, 2, 200);
    x = linspace(0, 5, 200);

    [~, ~, min_y, max_y] = compute_view_limits(x, y);

    verifyLessThan(testCase, max_y, 200, ...
        'A 50 m approach should not be plotted against a multi-kilometre axis.');
    fill_fraction = (max(y) - min(y)) / (max_y - min_y);
    verifyGreaterThan(testCase, fill_fraction, 0.7, ...
        'Low-altitude approach does not fill the frame.');
end

function testLimitsAreAlwaysValid(testCase)
    % Axis limits must be strictly increasing, including for a hovering trajectory whose
    % extent is effectively zero - MATLAB rejects a zero-width axis.
    cases = { linspace(2500,2,100), linspace(0,300,100); ...
              ones(1,100)*100,      zeros(1,100); ...
              [5 5],               [0 0] };
    for i = 1:size(cases,1)
        y = cases{i,1}; x = cases{i,2};
        [min_x, max_x, min_y, max_y] = compute_view_limits(x, y);
        verifyGreaterThan(testCase, max_x, min_x, 'x limits must be increasing.');
        verifyGreaterThan(testCase, max_y, min_y, 'y limits must be increasing.');
        verifyTrue(testCase, all(isfinite([min_x max_x min_y max_y])), ...
            'Axis limits must be finite.');
    end
end

function testSurfaceIsInFrame(testCase)
    % The ground line at y = 0 must always be visible; a descent plot that crops the
    % surface is useless for judging a landing.
    y = linspace(2500, 2, 300);
    x = linspace(0, 300, 300);
    [~, ~, min_y, ~] = compute_view_limits(x, y);
    verifyLessThanOrEqual(testCase, min_y, 0, 'The surface (y=0) must be within frame.');
end
