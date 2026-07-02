function tests = test_dynamics
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
    % Dynamically load the exact universe parameters from the central config
    testCase.TestData.params = get_sim_params();
end
function testEmptyFuelTank(testCase)
    % Scenario: Fuel tank is completely empty (m_fuel = 0), but AI commands 100% thrust
    x = [0; 1000; 0; -10; 0; 0; 0]; 
    u = [testCase.TestData.params.max_main_thrust; 0];
    
    % Compute state time-derivative for given state and input
    dxdt = lunar_lander_dynamics(x, u, testCase.TestData.params);
    
    % dm_fuel (the first derivative of the 7th element) should be exactly 0, preventing negative mass
    verifyEqual(testCase, dxdt(7), 0, 'Fuel mass derivative should be 0 when tank is empty.');
    
    % ddy (the 4th derivative) should be exactly -gravity, ignoring the commanded thrust
    verifyEqual(testCase, dxdt(4), -testCase.TestData.params.gravity, 'RelTol', 1e-4, ...
        'Lander should be in pure freefall when out of fuel, regardless of thrust command.');
end