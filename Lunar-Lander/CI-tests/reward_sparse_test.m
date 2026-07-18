function tests = reward_sparse_test
%REWARD_SPARSE_TEST - Unit tests for the sparse reward scheme
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Dynamically add the entire repository to path
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    
    testCase.TestData.params = get_sim_params();
    testCase.TestData.weights = get_reward_weights();
end

function testMidFlight(testCase)
    % Scenario: Mid flight, no crash or success
    x = [0; 5000; 0; 0; 0; 0; 0; 0];
    [Reward, IsDone] = reward_sparse_only(x, [0;0], [0;0], false, testCase.TestData.params);
    
    verifyEqual(testCase, Reward, 0, 'Sparse reward should be exactly 0 during mid-flight.');
    verifyFalse(testCase, IsDone, 'Simulation should not be done mid-flight.');
end

function testCrash(testCase)
    % Scenario: Hits ground too fast
    x = [0; 0; 0; -10; 0; 0; 0; 0];
    [Reward, IsDone] = reward_sparse_only(x, [0;0], [0;0], false, testCase.TestData.params);
    
    verifyEqual(testCase, Reward, testCase.TestData.weights.crash, 'Should receive massive crash penalty.');
    verifyTrue(testCase, IsDone, 'Simulation should terminate on crash.');
end

function testSuccess(testCase)
    % Scenario: Soft touchdown
    x = [0; 0; 0; -0.5; 0; 0; 0; 0];
    [Reward, IsDone] = reward_sparse_only(x, [0;0], [0;0], false, testCase.TestData.params);
    
    verifyEqual(testCase, Reward, testCase.TestData.weights.success, 'Should receive massive success payout.');
    verifyTrue(testCase, IsDone, 'Simulation should terminate on landing.');
end
