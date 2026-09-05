function tests = imitation_pipeline_test
%IMITATION_PIPELINE_TEST - Contracts of the demonstration-seeding path.
%
% Four RL architectures produced zero landings from scratch; cloning the classical pilot
% produces 80%. That makes this path load-bearing, and three of its failure modes are
% silent - they produce a worse policy rather than an error:
%
%   - ResetExperienceBufferBeforeTraining defaulting TRUE, which discards every
%     demonstration before the first update
%   - the toolbox's cell-array experience format, which MathWorks' own documented example
%     gets wrong
%   - cloning a Gaussian actor against raw actions when its mean head is unsquashed, which
%     trains the policy toward tanh(a*) instead of a*
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    testCase.TestData.params = get_sim_params();
    % One small dataset shared by the whole file; generating demos is the slow part.
    testCase.TestData.demos = generate_demonstrations( ...
        struct('n_per_phase', 2, 'outfile', ''));
end

function testSeedingPopulatesTheReplayBuffer(testCase)
    p = testCase.TestData.params;
    demos = testCase.TestData.demos;
    env = LunarLanderEnv('DenseBaseline', 'on');
    agent = build_agent('td3', getObservationInfo(env), getActionInfo(env), p.agent_dt);

    verifyEqual(testCase, agent.ExperienceBuffer.Length, 0, ...
        'Precondition failed: a fresh agent should have an empty buffer.');

    agent = seed_agent_from_demonstrations(agent, demos, p, struct('verbose', false));

    expected = sum(arrayfun(@(e) size(e.actions, 2), demos.episodes));
    verifyEqual(testCase, agent.ExperienceBuffer.Length, expected, ...
        ['Seeded buffer length does not match the number of demonstration ' ...
         'transitions. The experience format is probably being rejected.']);

    % The buffer must be sampleable, which is what actually exercises the cell-array
    % contract that rlReplayMemory.validateExperience enforces.
    batch = sample(agent.ExperienceBuffer, 4);
    verifySize(testCase, batch.Observation{1}, [10 1 4]);
    verifySize(testCase, batch.Action{1}, [2 1 4]);
end

function testSeededAgentKeepsItsBufferOnTraining(testCase)
    % THE ONE-LINE TRAP. The toolbox default empties the replay buffer before the first
    % update, which would discard every demonstration and make a seeded run look exactly
    % like an ordinary from-scratch failure. Nothing errors and nothing warns.
    p = testCase.TestData.params;
    env = LunarLanderEnv('DenseBaseline', 'on');
    for atype = {'td3', 'ddpg', 'sac'}
        agent = build_agent(atype{1}, getObservationInfo(env), getActionInfo(env), ...
                            p.agent_dt);
        verifyFalse(testCase, agent.AgentOptions.ResetExperienceBufferBeforeTraining, ...
            sprintf(['%s would clear its experience buffer on train(), discarding any ' ...
                     'seeded demonstrations.'], upper(atype{1})));
    end
end

function testCloningRejectsAGaussianActor(testCase)
    % SAC's actor leaves ActorMean UNSQUASHED and applies tanh internally, so a
    % behaviour-cloning target must be atanh(a*). Regressing raw actions against the mean
    % head would train the policy toward tanh(a*) and fail silently, so this must error
    % rather than fit the wrong thing.
    p = testCase.TestData.params;
    env = LunarLanderEnv('DenseBaseline', 'on');
    sac = build_agent('sac', getObservationInfo(env), getActionInfo(env), p.agent_dt);

    verifyError(testCase, ...
        @() pretrain_actor_supervised(sac, testCase.TestData.demos, p, ...
                                      struct('max_epochs', 1, 'verbose', false)), ...
        'pretrainActorSupervised:NotDeterministic', ...
        'Cloning a Gaussian actor against raw actions must be refused, not attempted.');
end

function testCloningActuallyChangesThePolicy(testCase)
    % A weak but load-bearing assertion: the fitted weights must reach the agent. An
    % earlier version could have failed here silently, because setModel requires the layer
    % names to be unchanged and quietly does nothing useful if the network does not match.
    p = testCase.TestData.params;
    env = LunarLanderEnv('DenseBaseline', 'on');
    agent = build_agent('td3', getObservationInfo(env), getActionInfo(env), p.agent_dt);

    x = [0; 50; 1; -2; 0.02; 0; p.max_main_fuel; p.max_rcs_fuel];
    obs = get_ai_observation(x, p);
    before = getAction(agent, {obs}); before = double(before{1});

    agent = pretrain_actor_supervised(agent, testCase.TestData.demos, p, ...
                struct('max_epochs', 3, 'verbose', false));

    after = getAction(agent, {obs}); after = double(after{1});
    verifyGreaterThan(testCase, norm(after - before), 1e-6, ...
        'The cloned network does not appear to have been installed into the agent.');
end

function testPlaybackEntryPointFailsInformatively(testCase)
    % Agent .mat files are gitignored, so a fresh clone of this repo has none. The error
    % must say how to regenerate one rather than surfacing a bare load() failure.
    verifyError(testCase, @() run_trained_agent(true, 'no_such_agent_file.mat'), ...
        'runTrainedAgent:NoAgent', ...
        'Missing agent file should raise a specific, actionable error.');
end
