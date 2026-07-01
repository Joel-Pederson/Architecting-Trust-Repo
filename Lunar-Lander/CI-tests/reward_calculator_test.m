function tests = test_reward_calculator
    % Main function to group all tests in this file
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Dynamically load the exact universe parameters from the central configs
    testCase.TestData.params = get_lander_params();
    testCase.TestData.weights = get_reward_weights();
end

function testCatastrophicCrash(testCase)
    % Scenario: Hits the ground (y=0) at a lethal 50 m/s
    x = [0; 0; 0; -50; 0; 0; 1000]; 
    u_actual = [0; 0];
    u_prev = [0; 0];
    VetoTriggered = false;
    
    [Reward, IsDone] = calculate_reward(x, u_actual, u_prev, VetoTriggered, testCase.TestData.params);
    
    % Assert the simulation ended
    verifyTrue(testCase, IsDone, 'Simulation should terminate on ground contact.');
    
    % Assert the reward is severely negative (allowing room for minor continuous penalties)
    % Expected reward should be roughly the crash penalty minus a few fractional points.
    % We use 0.95 (95%) so the test doesn't fail due to minor fuel/distance penalties.
    expected_threshold = testCase.TestData.weights.crash * 0.95; 
    verifyLessThan(testCase, Reward, expected_threshold, 'Agent should receive massive penalty for crashing.');
end

function testSoftLanding(testCase)
    % Scenario: Perfect touchdown (y=0) at a gentle 0.5 m/s
    x = [0; 0; 0; -0.5; 0; 0; 1000]; 
    u_actual = [0; 0];
    u_prev = [0; 0];
    VetoTriggered = false;
    
    [Reward, IsDone] = calculate_reward(x, u_actual, u_prev, VetoTriggered, testCase.TestData.params);
    
    % Assert the simulation ended
    verifyTrue(testCase, IsDone, 'Simulation should terminate on ground contact.');
    
    % Assert the agent gets the massive success payout
    % We use 0.90 (90%) to allow room for the distance/fuel penalties accrued during flight.
    expected_threshold = testCase.TestData.weights.success * 0.90; 
    verifyGreaterThan(testCase, Reward, expected_threshold, 'Agent should receive massive bonus for safe landing.');
end