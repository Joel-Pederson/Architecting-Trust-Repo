function tests = evaluation_harness_test
%EVALUATION_HARNESS_TEST - Invariants of the measurement code itself.
%
% Every number in the trade study and the guardian A/B comes through evaluate_policy and
% rollout_episode. A defect here does not crash anything - it silently produces plausible
% wrong numbers, which is the most expensive kind of bug in an experimental harness. Two
% of the assertions below are regressions that already happened once.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    testCase.TestData.params = get_sim_params();
end

function testRolloutRestoresExplorationSetting(testCase)
    % rollout_episode disables exploration so stochastic policies are measured on their
    % mean action rather than a sample. It MUST put the setting back: a caller that
    % evaluates mid-training and then continues would otherwise train a policy whose
    % exploration had been silently switched off, and nothing would report that.
    p = testCase.TestData.params;
    env = LunarLanderEnv('DenseBaseline', 'on');
    env.CurriculumWeights = [1 0 0];
    agent = build_agent('sac', getObservationInfo(env), getActionInfo(env), p.agent_dt);

    verifyTrue(testCase, agent.UseExplorationPolicy, ...
        'Precondition failed: a fresh SAC agent should start with exploration on.');

    rollout_episode(env, agent, 20);

    verifyTrue(testCase, agent.UseExplorationPolicy, ...
        ['rollout_episode did not restore UseExplorationPolicy. Any caller that ' ...
         'evaluates and then keeps training loses its exploration silently.']);
end

function testOutcomeRatesSumToOne(testCase)
    % Every episode must be accounted for. If a new terminal outcome is added and not
    % counted here, the rates quietly stop summing and a reader cannot tell what happened
    % to the missing episodes - which is exactly why the A/B table reports all five.
    m = quick_metrics(testCase, 4);
    total = m.landing_rate + m.crash_rate + m.oob_rate + m.stall_rate + m.timeout_rate;
    fprintf('\n  outcome rates sum to %.6f\n', total);
    verifyEqual(testCase, total, 1, 'AbsTol', 1e-12, ...
        sprintf(['Outcome rates sum to %.6f, not 1. Some episodes are unaccounted ' ...
                 'for in the reported breakdown.'], total));
end

function testImpactSpeedIsMeasuredOverGroundedEpisodesOnly(testCase)
    % THE REGRESSION. Averaging impact over timeouts folds in "velocity at the step cap"
    % for a lander that is still airborne, which once produced a reported mean impact of
    % 132.87 m/s on a scenario that starts at 50 m.
    m = quick_metrics(testCase, 4);

    grounded = strcmp(m.outcomes, 'landed') | strcmp(m.outcomes, 'crashed');
    verifyEqual(testCase, m.n_grounded, nnz(grounded), ...
        'n_grounded does not match the number of episodes that reached the ground.');

    if m.n_grounded == 0
        % Legitimately possible; the contract is NaN rather than a fabricated number.
        verifyTrue(testCase, isnan(m.mean_impact_speed), ...
            'With no grounded episodes, mean impact must be NaN, not a made-up value.');
    else
        verifyFalse(testCase, isnan(m.mean_impact_speed), ...
            'Episodes reached the ground but mean impact is NaN.');
        % Sanity bound: nothing in this simulator reaches 500 m/s from a 50 m start.
        verifyLessThan(testCase, m.mean_impact_speed, 500, ...
            'Mean impact is implausible; timeouts are probably being averaged in.');
    end
end

function testPerPhaseCountsAccountForEveryEpisode(testCase)
    % A headline landing rate averaged over a 50 m hover and a 2500 m descent hides which
    % regime works, so the per-phase split has to be complete.
    m = quick_metrics(testCase, 4);
    verifyEqual(testCase, sum(m.n_by_phase), m.n, ...
        'Per-phase episode counts do not sum to the total number of episodes.');
end

function testEvaluationForcesTheUniformCurriculum(testCase)
    % Training uses a P1-weighted diet to make the success region reachable. Scoring on
    % that same easy diet would inflate the landing rate, so evaluate_policy overrides the
    % mix. If that override is removed, the A/B silently flatters itself.
    p = testCase.TestData.params;
    env = LunarLanderEnv('DenseBaseline', 'on');
    env.CurriculumWeights = [1 0 0];            % deliberately skewed
    agent = build_agent('td3', getObservationInfo(env), getActionInfo(env), p.agent_dt);

    evaluate_policy(env, agent, 1);

    verifyEqual(testCase, env.CurriculumWeights, p.eval_curriculum_weights, 'AbsTol', 1e-12, ...
        'evaluate_policy did not force the uniform evaluation curriculum.');
end

function m = quick_metrics(testCase, n)
% Small evaluation on Phase 1 with an untrained agent. Outcome quality is irrelevant
% here - these tests are about the bookkeeping, not the policy.
    p = testCase.TestData.params;
    env = LunarLanderEnv('DenseBaseline', 'on');
    agent = build_agent('td3', getObservationInfo(env), getActionInfo(env), p.agent_dt);
    rng(5);
    m = evaluate_policy(env, agent, n);
end
