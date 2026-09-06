function tests = demonstrations_test
%DEMONSTRATIONS_TEST - The recorded dataset must be a faithful record of what happened.
%
% Behaviour cloning regresses a network onto (state, action) pairs. If the recorded action
% is not the action the environment actually received, the network is trained toward a
% trajectory that was never flown - and nothing downstream fails loudly. It just learns
% slightly wrong behaviour and the run looks like ordinary underperformance.
%
% Two specific ways that happens here, both guarded below:
%   - recording the UNCLIPPED quotient from the inverse action map instead of the clipped
%     action, so targets include commands the plant cannot execute
%   - caching NORMALISED observations, which are silently invalidated by any edit to the
%     unowned literals in get_ai_observation
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    testCase.TestData.params = get_sim_params();
end

function testReplayingRecordedActionsReproducesTheTrajectory(testCase)
    % The core fidelity assertion. Replay the recorded action sequence from the recorded
    % initial state and require the same terminal state and outcome.
    p = testCase.TestData.params;
    d = generate_demonstrations(struct('n_per_phase', 2, 'noise_frac', 1.0, 'outfile', ''));
    verifyGreaterThan(testCase, d.n_episodes, 0, 'Generator produced no landed episodes.');

    e = d.episodes(1);
    env = LunarLanderEnv('DenseBaseline', d.meta.guardian);
    reset(env);
    env.State = e.states(:, 1);   % force the recorded initial condition

    replay_reward = 0;
    for k = 1:size(e.actions, 2)
        [~, r, done] = step(env, e.actions(:, k));
        replay_reward = replay_reward + r;
        if done, break; end
    end

    verifyEqual(testCase, env.State, e.states(:, end), 'AbsTol', 1e-9, ...
        ['Replaying the recorded actions did not reproduce the recorded terminal ' ...
         'state. The dataset is not a faithful record and BC would learn noise.']);
    verifyEqual(testCase, env.Outcome, e.outcome, ...
        'Replay produced a different outcome than was recorded.');
    verifyEqual(testCase, replay_reward, sum(e.rewards), 'AbsTol', 1e-9, ...
        'Replay reward does not match the recorded reward.');
end

function testOnlyLandedEpisodesAreKept(testCase)
    d = generate_demonstrations(struct('n_per_phase', 3, 'outfile', ''));
    for i = 1:numel(d.episodes)
        verifyEqual(testCase, d.episodes(i).outcome, 'landed', ...
            'A non-landing episode was kept; the actor would be cloned onto a failure.');
    end
end

function testRecordedActionsAreInsideTheActionSpace(testCase)
    % Guards against recording the raw quotient rather than the clipped action.
    d = generate_demonstrations(struct('n_per_phase', 2, 'noise_frac', 1.0, 'outfile', ''));
    for i = 1:numel(d.episodes)
        a = d.episodes(i).actions;
        verifyLessThanOrEqual(testCase, max(a(:)),  1 + 1e-12, ...
            'Recorded action exceeds +1; the unclipped quotient was stored.');
        verifyGreaterThanOrEqual(testCase, min(a(:)), -1 - 1e-12, ...
            'Recorded action is below -1; the unclipped quotient was stored.');
    end
end

function testStatesAreRawPhysicsNotNormalised(testCase)
    % Raw states are stored so the dataset survives edits to get_ai_observation's
    % normalisation literals. A normalised Phase 3 start would have altitude ~0.83;
    % a raw one is ~2500.
    d = generate_demonstrations(struct('n_per_phase', 4, 'outfile', ''));
    p3 = d.episodes([d.episodes.phase] == 3);
    verifyGreaterThan(testCase, numel(p3), 0, 'No Phase 3 episodes to check.');
    y0 = p3(1).states(2, 1);
    verifyGreaterThan(testCase, y0, 1000, ...
        sprintf(['Phase 3 initial altitude recorded as %.3f. States appear to be ' ...
                 'normalised; they must be raw physics units.'], y0));
end

function testEpisodeArraysAreDimensionallyConsistent(testCase)
    % states must be one column longer than actions, so every transition has both its
    % state and its next state. Off-by-one here would corrupt every NextObservation.
    d = generate_demonstrations(struct('n_per_phase', 2, 'outfile', ''));
    for i = 1:numel(d.episodes)
        e = d.episodes(i);
        n = size(e.actions, 2);
        verifyEqual(testCase, size(e.states, 2), n + 1, ...
            'states must have exactly one more column than actions.');
        verifyEqual(testCase, numel(e.rewards), n, ...
            'rewards must have one entry per action.');
        verifySize(testCase, e.states, [8, n + 1]);
        verifySize(testCase, e.actions, [2, n]);
    end
end
