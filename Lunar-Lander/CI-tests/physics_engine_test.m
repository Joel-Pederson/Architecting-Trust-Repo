function tests = physics_engine_test
%TEST_DYNAMICS - Create function-based tests for lunar lander dynamics
%
% Input arguments:
% None
%
% Output arguments:
% tests - functiontests structure for use with runtests
    tests = functiontests(localfunctions);
end
function setupOnce(testCase)
    % Dynamically add the entire repository (and all subfolders) to the MATLAB path
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    % Dynamically load the exact universe parameters from the central config
    testCase.TestData.params = get_sim_params();
    
end
function testEmptyFuelTank(testCase)
    % Scenario: Fuel tank is completely empty (m_fuel = 0), but AI commands 100% thrust
    x = [0; 1000; 0; -10; 0; 0; 0; 0]; 
    u = [testCase.TestData.params.max_main_thrust; 0];
    
    % Compute state time-derivative for given state and input
    dxdt = lunar_lander_dynamics(x, u, testCase.TestData.params);
    
    % dm_fuel (the first derivative of the 7th element) should be exactly 0, preventing negative mass
    verifyEqual(testCase, dxdt(7), 0, 'Main fuel mass derivative should be 0 when tank is empty.');
    verifyEqual(testCase, dxdt(8), 0, 'RCS fuel mass derivative should be 0 when tank is empty.');
    
    % ddy should be pure gravity, ignoring the commanded thrust - but gravity AT THIS
    % ALTITUDE, not the surface value. The dynamics uses inverse-square falloff, which is
    % 0.12% at 1 km and 1.7% at the 15.2 km powered-descent start. This assertion
    % previously encoded the constant-gravity simplification.
    prm = testCase.TestData.params;
    g_at_alt = prm.gravity * (prm.r_lunar / (prm.r_lunar + x(2)))^2;
    verifyEqual(testCase, dxdt(4), -g_at_alt, 'RelTol', 1e-6, ...
        'Lander should be in pure freefall when out of fuel, regardless of thrust command.');

    % And the falloff must actually be present: asserting only the surface value would
    % pass just as happily with gravity hardcoded.
    verifyLessThan(testCase, abs(dxdt(4)), prm.gravity, ...
        'Gravity is not falling off with altitude; the inverse-square term is missing.');
end