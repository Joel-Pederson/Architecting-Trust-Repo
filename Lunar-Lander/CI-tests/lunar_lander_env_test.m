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
    
    % The observation is now scaled into [-1, 1] percentage bounds for the neural network.
    % We must unscale it before verifying physical boundaries.
    unscaled_y = InitialObservation(2) * 20000;
    is_valid_alt = (unscaled_y > 30 && unscaled_y < 70) || ...
                   (unscaled_y > 1800 && unscaled_y < 2200) || ...
                   (unscaled_y > 14800 && unscaled_y < 15200);
    verifyTrue(testCase, is_valid_alt, 'Initial altitude should match Phase 1 (~50m), Phase 2 (~2000m), or Phase 3 (~15000m).');
    
    unscaled_dx = InitialObservation(3) * 2000;
    is_valid_vel = (abs(unscaled_dx) < 10) || ...
                   (unscaled_dx > 30 && unscaled_dx < 70) || ...
                   (unscaled_dx > 1600 && unscaled_dx < 1800);
    verifyTrue(testCase, is_valid_vel, 'Initial horizontal velocity should match Curriculum phase (~0, ~50, or ~1700 m/s).');
end

function testActionScalingAndIntegration(testCase)
    env = LunarLanderEnv('DenseBaseline');
    reset(env);
    
    % Force dy to 0 so we can strictly measure gravity acceleration
    env.State(4) = 0; 
    
    % Force dx to 0 to eliminate centrifugal lift. 
    % (At 1700 m/s orbital velocity, centrifugal force completely cancels gravity!)
    env.State(3) = 0; 
    
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
