function tests = agent_builder_test
%AGENT_BUILDER_TEST - Unit tests for the modular agent builders
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Dynamically add the entire repository to path
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));

    env = LunarLanderEnv();
    testCase.TestData.obsInfo = getObservationInfo(env);
    testCase.TestData.actInfo = getActionInfo(env);
    testCase.TestData.params  = get_sim_params();
end

function testAllArchitecturesBuild(testCase)
    % Every architecture in the trade study must construct. If one fails here the trade
    % is not a fair comparison - it is a comparison of whatever happened to build.
    types = {'ddpg', 'td3', 'sac', 'ppo'};
    expected = {'rl.agent.rlDDPGAgent', 'rl.agent.rlTD3Agent', ...
                'rl.agent.rlSACAgent', 'rl.agent.rlPPOAgent'};

    for i = 1:numel(types)
        agent = build_agent(types{i}, testCase.TestData.obsInfo, ...
            testCase.TestData.actInfo, testCase.TestData.params.agent_dt);
        verifyClass(testCase, agent, expected{i}, ...
            sprintf('Failed to construct a valid %s agent.', upper(types{i})));
    end
end

function testSampleTimeIsAgentRateNotPhysicsRate(testCase)
    % The clock must be the AGENT sample time (0.1 s), not the physics step (0.02 s).
    % The environment holds each action across control_decimation substeps, so passing
    % params.dt here would shrink every discount horizon by 5x - the exact bug that gave
    % the original harness a 2 second lookahead on a 300 second descent.
    params = testCase.TestData.params;

    verifyEqual(testCase, params.agent_dt, params.dt * params.control_decimation, ...
        'RelTol', 1e-12, 'agent_dt must equal dt * control_decimation.');

    types = {'ddpg', 'td3', 'sac', 'ppo'};
    for i = 1:numel(types)
        agent = build_agent(types{i}, testCase.TestData.obsInfo, ...
            testCase.TestData.actInfo, params.agent_dt);
        verifyEqual(testCase, agent.AgentOptions.SampleTime, params.agent_dt, ...
            'RelTol', 1e-12, sprintf('%s sample time must be the agent rate.', upper(types{i})));
    end
end

function testDiscountHorizonReachesTheLanding(testCase)
    % A discount factor is only meaningful relative to the sample time. Assert the
    % effective horizon covers a substantial fraction of a descent rather than a couple
    % of seconds.
    params = testCase.TestData.params;
    hp = load_hyperparams('ddpg');

    horizon_steps = 1 / (1 - hp.Gamma);
    horizon_seconds = horizon_steps * params.agent_dt;

    verifyGreaterThan(testCase, horizon_seconds, 15, ...
        sprintf(['Discount horizon is only %.1f s. The agent cannot credit-assign to a ' ...
                 'touchdown it never sees.'], horizon_seconds));
end

function testReplayBufferIsLargeEnough(testCase)
    % MATLAB defaults ExperienceBufferLength to 10,000. Against 3,000-step episodes that
    % held roughly three episodes of history, so the agent continuously forgot every
    % landing it had ever seen.
    params = testCase.TestData.params;

    for name = {'ddpg', 'td3', 'sac'}
        agent = build_agent(name{1}, testCase.TestData.obsInfo, ...
            testCase.TestData.actInfo, params.agent_dt);
        buffer_episodes = agent.AgentOptions.ExperienceBufferLength / params.max_agent_steps;
        verifyGreaterThan(testCase, buffer_episodes, 50, ...
            sprintf('%s replay buffer holds only %.1f episodes.', upper(name{1}), buffer_episodes));
    end
end

function testUnknownArchitectureErrors(testCase)
    params = testCase.TestData.params;
    verifyError(testCase, ...
        @() build_agent('dqn', testCase.TestData.obsInfo, testCase.TestData.actInfo, params.agent_dt), ...
        'buildAgent:UnknownAgentType', ...
        'An unsupported architecture must raise rather than silently default.');
end

function testDiscountSeparatesLandingFromFlyingAway(testCase)
    % A discount factor is only "long enough" relative to the reward it has to reach. The
    % harness ran at gamma = 0.995 - a 200-step horizon - while a MEASURED Phase 3 landing
    % takes 1946 agent steps with all the reward at the end. Consequence: the discounted
    % return of landing beat that of simply climbing away by 0.05, and DDPG collapsed to a
    % constant max-climb action (thrust spread 0.0034 across the entire state space).
    %
    % Asserting a horizon in seconds is the wrong test - it is a rule of thumb, and an
    % earlier version of this test tripped on 100 >= 100 in floating point while missing
    % that 100 s was too short anyway. What matters is how much of the terminal reward
    % survives discounting back to the START of the longest episode.
    p = get_sim_params();

    P3_LANDING_STEPS = 1946;   % measured: scripted_pilot, Phase 3, guardian ON
    surviving = p.gamma_agent ^ P3_LANDING_STEPS;

    fprintf('\n  gamma %.4f: Phase 3 landing reward is worth %.0f%% of itself at t=0\n', ...
        p.gamma_agent, 100*surviving);
    verifyGreaterThan(testCase, surviving, 0.25, ...
        sprintf(['Only %.1f%% of the landing reward survives discounting back over a ' ...
                 '%d-step Phase 3 descent. The agent cannot see the goal from the ' ...
                 'start of the episode.'], 100*surviving, P3_LANDING_STEPS));

    % The agents must discount with the value the environment shapes with, or
    % potential-based shaping is being applied against a different objective.
    hp = load_hyperparams('td3');
    verifyEqual(testCase, hp.Gamma, p.gamma_agent, ...
        'Agent gamma has drifted from the environment gamma.');
end

function testPolicyDiversityCheckDoesNotFlagAHealthyPolicy(testCase)
    % check_policy_diversity exists to catch actor saturation - a policy that has stopped
    % being a function of the state. Verified TRUE-POSITIVE against the real collapsed
    % DDPG agent (thrust spread 0.0034 across a 5 m - 2500 m state sweep).
    %
    % This asserts the other half: it must not fire on a policy that is merely BAD. An
    % untrained network is random, not collapsed, so it has to pass - otherwise the check
    % would flag every architecture at episode zero and be useless as a health gate.
    p = get_sim_params();
    env = LunarLanderEnv('DenseBaseline', 'on');
    agent = build_agent('td3', getObservationInfo(env), getActionInfo(env), p.agent_dt);

    [ok, rep] = check_policy_diversity(agent, p);
    fprintf('\n  untrained TD3 action spread: [%.4f %.4f]\n', rep.spread(1), rep.spread(2));
    verifyTrue(testCase, ok, ...
        sprintf(['Diversity check flagged an untrained (random, therefore ' ...
                 'state-dependent) policy: %s'], rep.reason));

    % TRUE POSITIVE, against the real collapsed agent kept as a regression fixture.
    % Skipped rather than failed if the artifact has been cleaned, so the suite stays
    % runnable on a fresh clone.
    fixture = fullfile(fileparts(mfilename('fullpath')), '..', ...
                       'trade_agent_ddpg_gamma995_saturated.mat');
    if isfile(fixture)
        d = load(fixture, 'agent');
        [bad_ok, bad_rep] = check_policy_diversity(d.agent, p);
        fprintf('  saturated DDPG spread:      [%.4f %.4f]\n', ...
            bad_rep.spread(1), bad_rep.spread(2));
        verifyFalse(testCase, bad_ok, ...
            'Diversity check failed to flag the known-collapsed DDPG agent.');
    end
end
