function tests = agent_builder_test
%AGENT_BUILDER_TEST - Unit tests for the DDPG Agent Builder
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Dynamically add the entire repository to path
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
end

function testBuildDDPGAgent(testCase)
    % Initialize environment to get the observation and action specs
    env = LunarLanderEnv();
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    
    % Get physical universe params to verify clock sync
    params = get_sim_params();
    dt = params.dt;
    
    % Build the agent
    agent = build_ddpg_agent(obsInfo, actInfo, dt);
    
    % Verify it successfully built an rlDDPGAgent object
    verifyClass(testCase, agent, 'rl.agent.rlDDPGAgent', 'Failed to construct a valid DDPG Agent.');
    
    % Verify the agent's internal clock is perfectly synced to the physics engine
    verifyEqual(testCase, agent.AgentOptions.SampleTime, 0.02, ...
        'Agent sample time must be synced with the physics engine dt (0.02s).');
end
