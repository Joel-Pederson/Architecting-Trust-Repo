function tests = test_safety_sidecar
%TEST_SAFETY_SIDECAR - Unit tests for the safety_sidecar_filter wrapper
%
% This test harness constructs function-based tests for safety_sidecar_filter.
    tests = functiontests(localfunctions);
end
function setupOnce(testCase)
% Setup shared parameters used across all tests in this file
    testCase.TestData.params = struct('dry_mass', 4280, 'gravity', 1.62, ...
        'inertia', 24000, 'max_main_thrust', 45040, ...
        'max_mass_burn_rate', 15.6, 'max_side_torque', 2000);
end
function testSuicideBurnOverride(testCase)
    % Scenario: Lander is 1 meter off the ground, falling at 50 m/s.
    % The dead AI commands 0 thrust.
    % Test ensures sidecar vetoes and commands emergency thrust.
    x = [0; 1; 0; -50; 0; 0; 8000];
    u_nominal = [0; 0];
    
    % Invoke safety filter to get commanded controls and veto status
    [u_actual, VetoTriggered, ~, ~] = safety_sidecar_filter(x, u_nominal, testCase.TestData.params);
    
    % Assert the Sidecar correctly panicked and seized control
    verifyTrue(testCase, VetoTriggered, 'Sidecar failed to trigger veto in lethal scenario.');
    
    % Assert the Sidecar commanded absolute maximum thrust to try and save the ship
    verifyEqual(testCase, u_actual(1), testCase.TestData.params.max_main_thrust, 'RelTol', 1e-4, ...
        'Sidecar did not command max thrust during a critical boundary breach.');
end