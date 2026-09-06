function tests = animation_test
%ANIMATION_TEST - Smoke coverage for the visualiser.
%
% The animator was the one component with no test coverage, and it broke twice in a row
% because checkcode and the rest of the suite never open a figure:
%
%   1. A pacing change tuned for 50 Hz telemetry rendered a 10 Hz episode instantly, so
%      the figure appeared already finished and REPLAY looked like a dead button.
%   2. An edit whose text substitution silently missed left `veto_hold` referenced but
%      never defined, so every call errored at the first veto check.
%
% Both would have been caught in seconds by simply CALLING the function. These tests do
% that, headlessly.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    testCase.TestData.params = get_sim_params();
    testCase.TestData.vis = get(0, 'DefaultFigureVisible');
    set(0, 'DefaultFigureVisible', 'off');
end

function teardownOnce(testCase)
    set(0, 'DefaultFigureVisible', testCase.TestData.vis);
    close all force;
end

function testAnimatorRunsOnATrajectoryContainingVetoes(testCase)
    % The specific path that broke: a trajectory where the sidecar actually engages.
    p = testCase.TestData.params;
    ep = fly_with_fault(p, 'on', 12);
    verifyGreaterThan(testCase, nnz(ep.veto), 0, ...
        'Test fixture produced no vetoes, so it does not exercise the veto display.');

    animate_lunar_lander(ep.t, ep.states(1,:), ep.states(2,:), ep.states(4,:), ...
        ep.states(5,:), ep.controls(1,:), ep.states(7,:), ep.veto, p);
    close all force;
    verifyTrue(testCase, true, 'Animator completed on a trajectory with vetoes.');
end

function testAnimatorRunsWithNoVetoesAtAll(testCase)
    % The other branch: an all-false veto vector must not trip the latch logic.
    p = testCase.TestData.params;
    ep = fly_with_fault(p, 'off', 0);
    animate_lunar_lander(ep.t, ep.states(1,:), ep.states(2,:), ep.states(4,:), ...
        ep.states(5,:), ep.controls(1,:), ep.states(7,:), false(size(ep.veto)), p);
    close all force;
    verifyTrue(testCase, true, 'Animator completed with no vetoes.');
end

function testPlaybackIsPacedNotInstant(testCase)
    % Guards the pacing regression: a short agent-rate episode must still take a
    % watchable time rather than rendering in a single frame.
    p = testCase.TestData.params;
    ep = fly_with_fault(p, 'on', 12);
    t0 = tic;
    animate_lunar_lander(ep.t, ep.states(1,:), ep.states(2,:), ep.states(4,:), ...
        ep.states(5,:), ep.controls(1,:), ep.states(7,:), ep.veto, p);
    elapsed = toc(t0);
    close all force;
    verifyGreaterThan(testCase, elapsed, 1.0, ...
        sprintf(['Playback finished in %.2f s. It is rendering effectively instantly, ' ...
                 'which is what made REPLAY look like a dead button.'], elapsed));
end

function testScenarioNamesResolve(testCase)
    % The entry points take names now; a typo must say what the options are.
    verifyEqual(testCase, phase_from_name('touchdown'), 1);
    verifyEqual(testCase, phase_from_name('approach'),  2);
    verifyEqual(testCase, phase_from_name('terminal'),  3);
    verifyEqual(testCase, phase_from_name('orbit'),     4);
    verifyEqual(testCase, phase_from_name(4),           4);   % numbers still accepted
    verifyError(testCase, @() phase_from_name('banana'), 'phaseFromName:Unknown');
end

function ep = fly_with_fault(p, guardian, alt_bias)
% Short Phase 1 episode flown by the classical pilot, with an altimeter fault so the
% barrier engages. Uses the pilot rather than the agent so the fixture does not depend
% on a .mat file that a fresh clone will not have.
    env = LunarLanderEnv('DenseBaseline', guardian);
    env.CurriculumWeights = [1 0 0];
    rng(101); reset(env);
    cap = p.phase_max_steps(1);
    states = zeros(8, cap+1); controls = zeros(2, cap+1); veto = false(1, cap+1);
    states(:,1) = env.State;
    n = 1;
    for i = 1:cap
        s = env.State;
        u = scripted_pilot(apply_sensor_fault(s, 'alt_bias', alt_bias), p);
        [~,~,done,logs] = step(env, command_to_action(u, s, p));
        n = i+1;
        states(:,n) = logs.State; controls(:,n) = logs.Control; veto(n) = logs.VetoActive;
        if done, break; end
    end
    ep.states = states(:,1:n); ep.controls = controls(:,1:n); ep.veto = veto(1:n);
    ep.t = (0:n-1) * p.agent_dt;
end
