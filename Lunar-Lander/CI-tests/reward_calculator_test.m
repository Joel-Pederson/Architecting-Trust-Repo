function tests = reward_calculator_test
% Main function to group all tests in this file
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
%SETUPONCE - Load shared test data used by all test cases
% Dynamically add the entire repository (and all subfolders) to the MATLAB path
scriptPath = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(scriptPath, '..')));

testCase.TestData.params = get_sim_params();
testCase.TestData.weights = get_reward_weights();
end

function testCatastrophicCrash(testCase)
%TESTCATASTROPHICCRASH - Verify heavy penalty and termination on crash
% Scenario: hits the ground at a lethal 50 m/s
params = testCase.TestData.params;
weights = testCase.TestData.weights;

x = [0; 0; 0; -50; 0; 0; 1000; 300];
x_prev = x;                    % No movement between substeps, so shaping contributes 0
u_actual = [0; 0];
u_prev = [0; 0];
VetoEngaged = false;

[Reward, IsDone, Outcome] = reward_dense_baseline(x, x_prev, u_actual, u_prev, ...
    VetoEngaged, params, weights);

verifyTrue(testCase, IsDone, 'Simulation should terminate on ground contact.');
verifyEqual(testCase, Outcome, 'crashed', 'A 50 m/s impact must be classified as a crash.');

% Rewards are returned SCALED by weights.reward_scale, so the threshold must be scaled
% too. Comparing a scaled reward against a raw weight is a 100x error.
expected_threshold = (weights.crash / weights.reward_scale) * 0.95;
verifyLessThan(testCase, Reward, expected_threshold, ...
    'Agent should receive massive penalty for crashing.');
end

function testSoftLanding(testCase)
%TESTSOFTLANDING - Verify large success reward and termination on soft landing
params = testCase.TestData.params;
weights = testCase.TestData.weights;

x = [0; 0; 0; -0.5; 0; 0; 1000; 300];
x_prev = x;
u_actual = [0; 0];
u_prev = [0; 0];
VetoEngaged = false;

[Reward, IsDone, Outcome] = reward_dense_baseline(x, x_prev, u_actual, u_prev, ...
    VetoEngaged, params, weights);

verifyTrue(testCase, IsDone, 'Simulation should terminate on ground contact.');
verifyEqual(testCase, Outcome, 'landed', 'A 0.5 m/s upright touchdown must count as a landing.');

expected_threshold = (weights.success / weights.reward_scale) * 0.90;
verifyGreaterThan(testCase, Reward, expected_threshold, ...
    'Agent should receive massive bonus for safe landing.');
end

function testTouchdownLimitsComeFromParams(testCase)
%TESTTOUCHDOWNLIMITSCOMEFROMPARAMS - The success gate must honour the central limits.
% Guards against the reward scheme and the sidecar drifting apart on what "safe" means.
params = testCase.TestData.params;
weights = testCase.TestData.weights;

% Just inside every limit -> landing
x_ok = [0; 0; params.max_touchdown_dx * 0.9; -params.max_touchdown_dy * 0.9; ...
        params.max_touchdown_tilt * 0.9; 0; 1000; 300];
[~, ~, outcome_ok] = reward_dense_baseline(x_ok, x_ok, [0; 0], [0; 0], false, params, weights);
verifyEqual(testCase, outcome_ok, 'landed', 'Touchdown inside all limits must be a landing.');

% Lateral velocity alone pushed outside its limit -> crash
x_bad = x_ok;
x_bad(3) = params.max_touchdown_dx * 1.5;
[~, ~, outcome_bad] = reward_dense_baseline(x_bad, x_bad, [0; 0], [0; 0], false, params, weights);
verifyEqual(testCase, outcome_bad, 'crashed', 'Excess lateral velocity alone must fail the gate.');
end

function testSurvivalBeatsCrashing(testCase)
%TESTSURVIVALBEATSCRASHING - The inversion that broke the original harness.
%
% The veto penalty was charged per timestep at 50 Hz, making it cost -5.0 reward per
% SECOND while the worst possible crash cost only -7.5. Crashing immediately became the
% optimal policy, and because the guardian was engaged for the entire descent the agent
% could not escape the tax by flying better.
%
% This asserts the INVARIANT rather than one scenario: the maximum veto penalty an
% episode can possibly accrue must be strictly smaller in magnitude than the penalty for
% a hard-impact crash. If that ever inverts again, the agent is once more being taught
% that suicide is cheaper than survival.
params = testCase.TestData.params;
weights = testCase.TestData.weights;

% Worst case the environment can ever charge for guardian activity in one episode
worst_veto_cost = weights.sidecar_veto_budget / weights.reward_scale;

% A hard crash: straight into the surface at 50 m/s
x_crash = [0; 0; 0; -50; 0; 0; 1000; 300];
crash_reward = reward_dense_baseline(x_crash, x_crash, [0; 0], [0; 0], false, params, weights);

verifyGreaterThan(testCase, worst_veto_cost, crash_reward, ...
    sprintf(['Max episode veto cost (%.2f) must be cheaper than a hard crash (%.2f). ' ...
             'If not, the agent is again incentivised to crash on purpose.'], ...
            worst_veto_cost, crash_reward));

% And the budget must be a real constraint, not so large it is never reached
verifyLessThan(testCase, abs(weights.sidecar_veto), abs(weights.sidecar_veto_budget), ...
    'A single engagement should not exhaust the whole episode budget.');
end

function testVetoBudgetIsEnforced(testCase)
%TESTVETOBUDGETISENFORCED - The cap must actually bind in the environment.
%
% Edge-triggering alone does not bound the total cost: a chattering barrier can charge
% repeatedly. Drive many engagements and confirm the accumulated penalty saturates.
weights = testCase.TestData.weights;
budget_scaled = abs(weights.sidecar_veto_budget / weights.reward_scale);

env = LunarLanderEnv('DenseBaseline', 'on');
reset(env);

% Force repeated fresh engagements by alternating between a state that trips the
% barrier and one that clears it, so the rising edge fires over and over.
n_engagements = 100;
safe_state   = [0; 2500; 0;  0;   0; 0; 8000; 300];   % nothing for the guardian to do
lethal_state = [0;   30; 0; -45;  0; 0; 8000; 300];   % guardian must intervene

for k = 1:n_engagements
    env.State = safe_state;
    step(env, [0; 0]);
    env.State = lethal_state;
    step(env, [-1; 0]);
end

verifyGreaterThan(testCase, env.VetoCount, 10, ...
    'Test setup failed to produce repeated guardian engagements.');

% Total charged penalty must never exceed the budget. Other reward terms (shaping,
% fuel) also contribute, so compare against the veto ledger the environment keeps.
verifyLessThanOrEqual(testCase, env.VetoPenaltyPaid / weights.reward_scale, ...
    budget_scaled + 1e-9, ...
    sprintf(['Veto penalty ledger reached %.2f, exceeding the %.2f budget. ' ...
             'A chattering barrier can now out-cost a crash.'], ...
            env.VetoPenaltyPaid / weights.reward_scale, budget_scaled));
end

function testFinalApproachHasUsableGradient(testCase)
%TESTFINALAPPROACHHASUSABLEGRADIENT - The failure that produced zero landings.
%
% The agent must learn to slow from a few m/s to under 1 m/s in the last metres. The
% original penalty scaled impact speed by 50 m/s, which mapped that entire regime into
% 10%% of the penalty range: slowing from 5.0 to 1.05 m/s was worth just +0.395 while
% crossing the final 0.06 m/s paid +5.105. About 93%% of the reward sat in a step
% function at the boundary with almost no gradient leading to it, and 1500 episodes of
% PPO never found it.
%
% Assert there is real signal in the approach, not just at the finish line.
params = testCase.TestData.params;
weights = testCase.TestData.weights;

r_fast = terminal_reward(-5.00, params, weights);   % clearly a crash
r_near = terminal_reward(-1.05, params, weights);   % just outside the limit
r_land = terminal_reward(-0.99, params, weights);   % just inside

approach_signal = r_near - r_fast;
crossing_signal = r_land - r_near;

verifyGreaterThan(testCase, approach_signal, 2.0, ...
    sprintf(['Slowing from 5.0 to 1.05 m/s is worth only %.3f. The final-approach ' ...
             'gradient has collapsed again and the agent cannot learn to land.'], ...
            approach_signal));

% The approach must be a meaningful share of the total, not a rounding error next to
% the terminal bonus.
share = approach_signal / (approach_signal + crossing_signal);
verifyGreaterThan(testCase, share, 0.25, ...
    sprintf('Approach carries only %.0f%% of the signal; the rest is a step function.', ...
            100*share));

% Monotonicity: every reduction in impact speed must be rewarded.
speeds = [-8 -6 -5 -4 -3 -2 -1.5 -1.2 -1.05];
prev = -inf;
for v = speeds
    r = terminal_reward(v, params, weights);
    verifyGreaterThanOrEqual(testCase, r, prev - 1e-9, ...
        sprintf('Reward decreased when slowing down (at dy = %.2f).', v));
    prev = r;
end
end

function testPassFailBoundaryIsUnchanged(testCase)
%TESTPASSFAILBOUNDARYISUNCHANGED - Grading a miss must not move the safety criterion.
% The graded penalty measures how badly a touchdown missed; it must not grant "landed"
% to anything outside the Apollo LM gear limits.
params = testCase.TestData.params;
weights = testCase.TestData.weights;

% Just inside every limit -> landed
[~, ~, oc_in] = reward_dense_baseline( ...
    [0; 0; params.max_touchdown_dx*0.99; -params.max_touchdown_dy*0.99; ...
     params.max_touchdown_tilt*0.99; 0; 1000; 300], ...
    [0;0;0;0;0;0;1000;300], [0;0], [0;0], false, params, weights);
verifyEqual(testCase, oc_in, 'landed');

% Each criterion individually breached -> crashed
lims = {'max_touchdown_dx', 3; 'max_touchdown_dy', 4; 'max_touchdown_tilt', 5};
for i = 1:size(lims,1)
    x = [0; 0; 0; 0; 0; 0; 1000; 300];
    x(lims{i,2}) = params.(lims{i,1}) * 1.01;
    [~, ~, oc] = reward_dense_baseline(x, x, [0;0], [0;0], false, params, weights);
    verifyEqual(testCase, oc, 'crashed', ...
        sprintf('Breaching %s must fail the gate.', lims{i,1}));
end
end

function r = terminal_reward(dy, params, weights)
    x = [0; 2.0; 0; dy; 0; 0; 8000; 300];
    r = reward_dense_baseline(x, x, [0;0], [0;0], false, params, weights);
end
