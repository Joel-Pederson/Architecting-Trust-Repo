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
    [Reward, IsDone, Outcome] = call_sparse(testCase, x);

    verifyEqual(testCase, Reward, 0, 'Sparse reward should be exactly 0 during mid-flight.');
    verifyFalse(testCase, IsDone, 'Simulation should not be done mid-flight.');
    verifyEqual(testCase, Outcome, 'flying');
end

function testCrash(testCase)
    % Scenario: Hits ground too fast
    x = [0; 0; 0; -10; 0; 0; 0; 0];
    [Reward, IsDone, Outcome] = call_sparse(testCase, x);

    expected = testCase.TestData.weights.crash / testCase.TestData.weights.reward_scale;
    verifyEqual(testCase, Reward, expected, 'Should receive massive crash penalty.');
    verifyTrue(testCase, IsDone, 'Simulation should terminate on crash.');
    verifyEqual(testCase, Outcome, 'crashed');
end

function testSuccess(testCase)
    % Scenario: Soft touchdown
    x = [0; 0; 0; -0.5; 0; 0; 0; 0];
    [Reward, IsDone, Outcome] = call_sparse(testCase, x);

    expected = testCase.TestData.weights.success / testCase.TestData.weights.reward_scale;
    verifyEqual(testCase, Reward, expected, 'Should receive massive success payout.');
    verifyTrue(testCase, IsDone, 'Simulation should terminate on landing.');
    verifyEqual(testCase, Outcome, 'landed');
end

function testSignatureMatchesDenseScheme(testCase)
    % Both reward schemes are dispatched from the same call site in LunarLanderEnv, so
    % their signatures must stay identical. reward_sparse_only previously declared six
    % arguments while the environment passed seven, so `weights` silently received the
    % params struct and every terminal step died on an undefined field.
    verifyEqual(testCase, nargin(@reward_sparse_only), nargin(@reward_dense_baseline), ...
        'Reward schemes must accept the same argument list.');
    verifyEqual(testCase, nargout(@reward_sparse_only), nargout(@reward_dense_baseline), ...
        'Reward schemes must return the same outputs.');
end

function [Reward, IsDone, Outcome] = call_sparse(testCase, x)
    [Reward, IsDone, Outcome] = reward_sparse_only(x, x, [0; 0], [0; 0], false, ...
        testCase.TestData.params, testCase.TestData.weights);
end
