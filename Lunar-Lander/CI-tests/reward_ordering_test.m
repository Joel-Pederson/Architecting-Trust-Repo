function tests = reward_ordering_test
%REWARD_ORDERING_TEST - Executable specification of the intended preference ordering.
%
% Written after three successive reward changes each introduced the next problem:
%   1. Per-step veto penalty made crashing optimal.
%   2. Flat impact scaling left no gradient in the final approach.
%   3. Raising the crash penalty made HOVERING cheaper than attempting a landing,
%      producing 60% timeouts.
%
% Each was found by a ~30 minute training run. The ordering an agent should prefer is
% knowable in advance, so it belongs in a test that runs in seconds. If a future weight
% change breaks the ordering, this fails immediately instead of after a wasted run.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    testCase.TestData.params  = get_sim_params();
    testCase.TestData.weights = get_reward_weights();
end

function testPreferenceOrdering(testCase)
    p = testCase.TestData.params;
    w = testCase.TestData.weights;

    r_land     = touchdown(-0.5,  p, w);   % safe landing
    r_gentle   = touchdown(-1.05, p, w);   % just missed
    r_moderate = touchdown(-3.0,  p, w);   % moderate crash
    r_hard     = touchdown(-8.0,  p, w);   % hard crash
    r_hover    = full_episode_hover_cost(p, w);
    r_oob      = w.oob / w.reward_scale;

    fprintf('\n  landing        %+7.2f\n  gentle miss    %+7.2f\n  moderate crash %+7.2f\n', ...
        r_land, r_gentle, r_moderate);
    fprintf('  hard crash     %+7.2f\n  300s hover     %+7.2f\n  out of bounds  %+7.2f\n', ...
        r_hard, r_hover, r_oob);

    verifyGreaterThan(testCase, r_land, r_gentle, 'Landing must beat a near miss.');
    verifyGreaterThan(testCase, r_gentle, r_moderate, 'A near miss must beat a moderate crash.');
    verifyGreaterThan(testCase, r_moderate, r_hard, 'A moderate crash must beat a hard crash.');

    % THE ONE THAT BIT: loitering must never be cheaper than committing to a descent.
    verifyGreaterThan(testCase, r_hard, r_hover, ...
        sprintf(['A 300 s hover (%.2f) is cheaper than a hard crash (%.2f). The agent ' ...
                 'will learn to stay airborne and never land.'], r_hover, r_hard));

    verifyGreaterThan(testCase, r_hover, r_oob, 'Flying out of bounds must be the worst outcome.');
end

function testHoveringIsNotAnEscape(testCase)
    % A full-length hover must be clearly worse than any touchdown attempt, with margin -
    % not merely a hair worse.
    p = testCase.TestData.params;
    w = testCase.TestData.weights;
    r_hover = full_episode_hover_cost(p, w);
    % Reference a crash the agent REALISTICALLY produces (DDPG at 600 episodes arrives
    % at 4-17 m/s), not an extreme one. A 40 m/s smash being worse than a hover is
    % correct: hovering genuinely is the safer outcome of those two.
    r_typical_crash = touchdown(-8.0, p, w);
    verifyGreaterThan(testCase, r_typical_crash - r_hover, 0.5, ...
        'Hover and typical-crash costs are too close; the agent has little reason to commit.');
end

function r = touchdown(dy, p, w)
    x = [0; 2.0; 0; dy; 0; 0; 8000; 300];
    r = reward_dense_baseline(x, x, [0; 0], [0; 0], false, p, w);
end

function total = full_episode_hover_cost(p, w)
    % Full cost of holding a hover for a whole episode: the fuel burnt, PLUS the timeout
    % penalty for never reaching the ground. Both matter. Modelling only the fuel was
    % valid when fuel_main carried the anti-loitering job on its own; it no longer does,
    % and a test that measured only half the cost would happily pass a reward in which
    % hovering had quietly become the best available option again.
    m_total = p.dry_mass + p.max_main_fuel + p.max_rcs_fuel;
    throttle = (m_total * p.gravity) / p.max_main_thrust;
    per_substep = (w.fuel_main * throttle) * p.dt / w.reward_scale;
    fuel = per_substep * p.control_decimation * p.max_agent_steps;
    total = fuel + w.timeout / w.reward_scale;
end

function testLongEpisodesCannotFarmShapingReward(testCase)
    % A 3000-step timeout once scored +58.15 - better than any landing - because the
    % discounted shaping form gamma*Phi(s') - Phi(s) leaves a positive (1-gamma)*|Phi|
    % residual on every step when summed UNDISCOUNTED, which is how every metric in this
    % study sums it. Loitering must never pay.
    env = LunarLanderEnv('DenseBaseline', 'off');
    env.CurriculumWeights = [1 0 0];
    reset(env);

    R = 0;
    for i = 1:600                      % 60 s of deliberate hovering
        [~, r, done] = step(env, [0; 0]);   % action 0 = exact hover
        R = R + r;
        if done, break; end
    end

    fprintf('\n  60 s of hovering scores %+7.2f\n', R);
    verifyLessThan(testCase, R, 0, ...
        sprintf(['Hovering for 60 s pays %+.2f. Shaping is being farmed by staying ' ...
                 'airborne, which is exactly the 62%% timeout failure mode.'], R));
end

function testCrashPenaltyGradesTheWholeImpactRange(testCase)
    % THE THIRD FLAT-GRADIENT BUG. Measured with a controller blended toward free-fall:
    % on Phase 3, braking from 25.06 to 19.18 m/s - real, hard-won physical progress -
    % changed the episode reward by -0.02, while the last increment before landing was
    % worth +5.4. The landscape was a cliff, not a slope, so gradient methods had nothing
    % to follow until they were already almost perfect.
    %
    % Two saturations caused it:
    %   proximity = min(1, (miss-1)/3)   saturates at 4x the limit; 19 and 25 m/s scored
    %                                    identically at 1.0.
    %   severity  = min(1, impact/50)    spread 0-50 m/s across the term, so 19 vs 25 m/s
    %                                    differed by 0.05.
    %
    % Potential-based shaping CANNOT fix this: it is policy-invariant by construction, so
    % it telescopes to -Phi(s_0) and contributes nothing to episode-level ordering. Only
    % the terminal reward can grade how hard the vehicle hit.
    p = testCase.TestData.params;
    w = testCase.TestData.weights;

    speeds = [1.05 2 4 8 12 19 25 40];
    r = arrayfun(@(v) touchdown(-v, p, w), speeds);

    fprintf('\n  impact (m/s) -> reward\n');
    for i = 1:numel(speeds)
        fprintf('    %5.2f  %+7.3f\n', speeds(i), r(i));
    end

    % 1. Strictly monotone: slowing down must never be punished, anywhere.
    verifyTrue(testCase, all(diff(r) < 0), ...
        'Crash reward is not strictly decreasing in impact speed.');

    % 2. Usable slope in the FAR field, where a struggling agent actually lives. The
    %    25 -> 19 m/s step is the one that measured -0.02.
    far = touchdown(-19, p, w) - touchdown(-25, p, w);
    fprintf('  25.0 -> 19.0 m/s is worth %+.3f (was -0.02)\n', far);
    verifyGreaterThan(testCase, far, 0.25, ...
        sprintf(['Braking 25 -> 19 m/s is worth only %+.3f. An agent improving its ' ...
                 'descent gets no feedback until it is nearly perfect.'], far));

    % 3. Usable slope in the NEAR field too, which the previous design did get right and
    %    which must not be sacrificed to buy point 2.
    near = touchdown(-1.05, p, w) - touchdown(-4, p, w);
    fprintf('   4.0 -> 1.05 m/s is worth %+.3f\n', near);
    verifyGreaterThan(testCase, near, 1.0, ...
        'The final approach lost its gradient while fixing the far field.');

    % 4. Every crash still strictly worse than the landing it failed to be.
    verifyGreaterThan(testCase, w.success / w.reward_scale, max(r), ...
        'A crash scores at least as well as a landing.');
end
