function tests = lunar_lander_env_test
%LUNAR_LANDER_ENV_TEST - Unit tests for the custom RL Environment class
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Dynamically add the entire repository to path
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
end

function testReset(testCase)
    env = LunarLanderEnv('DenseBaseline');
    InitialObservation = reset(env);
    
    verifyEqual(testCase, length(InitialObservation), 8, 'Observation space should have 8 variables.');
    verifyEqual(testCase, InitialObservation(2), 15000, 'Initial altitude should be 15000m (Apollo 11).');
    verifyEqual(testCase, InitialObservation(3), 1700, 'Initial horizontal velocity should be 1700 m/s.');
end

function testActionScalingAndIntegration(testCase)
    env = LunarLanderEnv('DenseBaseline');
    reset(env);
    
    % Force dy to 0 so we can strictly measure gravity
    env.State(4) = 0; 
    
    % Command: [-1, 1] means Minimum Thrust (Engine Off), Max Right Torque
    [~, ~, ~, ~] = step(env, [-1; 1]);
    
    % Since engine is off, vertical acceleration is purely gravity (-1.62)
    % After one step (dt = 0.02), dy should be -1.62 * 0.02
    expected_dy = -env.params.gravity * env.params.dt;
    verifyEqual(testCase, env.State(4), expected_dy, 'RelTol', 1e-4, ...
        'Action scaling failed to map -1 to 0 thrust, or integration time-step is desynced.');
end

function testRewardRouting(testCase)
    % Test that initializing with 'SparseOnly' successfully routes to the sparse logic
    env = LunarLanderEnv('SparseOnly');
    reset(env);
    
    % Take one step mid-flight
    [~, Reward, ~, ~] = step(env, [0; 0]);
    
    % Dense baseline would penalize fuel and distance, but sparse should return exactly 0
    verifyEqual(testCase, Reward, 0, 'Environment failed to route to the SparseOnly reward scheme.');
end
