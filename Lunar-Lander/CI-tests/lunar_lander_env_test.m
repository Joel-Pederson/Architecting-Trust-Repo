function tests = lunar_lander_env_test
%LUNAR_LANDER_ENV_TEST - Unit tests for the custom RL Environment class
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Dynamically add the entire repository to path
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    testCase.TestData.params = get_sim_params();
end

function testReset(testCase)
    params = testCase.TestData.params;
    env = LunarLanderEnv('DenseBaseline');
    InitialObservation = reset(env);

    % 10 observations: the 8 physics states plus the two barrier values (h_alt, h_fuel)
    % that give the agent sight of the safety envelope.
    verifyEqual(testCase, numel(InitialObservation), 10, ...
        'Observation space should have 10 variables.');

    % Unscale using the SAME constants get_ai_observation normalizes with. Hard-coding a
    % second, different set of constants here is exactly how the visualizer ended up
    % rendering altitudes 6.7x too large.
    unscaled_y = InitialObservation(2) * 3000;
    is_valid_alt = (unscaled_y > 30   && unscaled_y < 70) || ...    % Phase 1: ~50 m
                   (unscaled_y > 420  && unscaled_y < 580) || ...   % Phase 2: ~500 m
                   (unscaled_y > 2300 && unscaled_y < 2700);        % Phase 3: ~2500 m
    verifyTrue(testCase, is_valid_alt, ...
        sprintf('Initial altitude %.1f m matches no curriculum phase.', unscaled_y));

    unscaled_dx = InitialObservation(3) * 100;
    is_valid_vel = (abs(unscaled_dx) < 10) || ...                   % Phase 1
                   (abs(unscaled_dx) > 3 && abs(unscaled_dx) < 20) || ...  % Phase 2: ~10
                   (abs(unscaled_dx) > 5 && abs(unscaled_dx) < 40);        % Phase 3: ~20
    verifyTrue(testCase, is_valid_vel, ...
        sprintf('Initial horizontal velocity %.1f m/s matches no curriculum phase.', unscaled_dx));

    % Fuel starts full, from the central params rather than a literal
    verifyEqual(testCase, env.State(7), params.max_main_fuel, 'RelTol', 1e-9);
    verifyEqual(testCase, env.State(8), params.max_rcs_fuel, 'RelTol', 1e-9);
end

function testDriftDirectionIsRandomized(testCase)
    % Every episode used to begin drifting to the right, so the agent only ever learned
    % to correct one sign of lateral error.
    env = LunarLanderEnv('DenseBaseline');
    signs = zeros(1, 60);
    for k = 1:60
        reset(env);
        signs(k) = sign(env.State(3));
    end
    verifyTrue(testCase, any(signs > 0) && any(signs < 0), ...
        'Initial horizontal drift must occur in both directions across episodes.');
end

function testActionScalingAndIntegration(testCase)
    env = LunarLanderEnv('DenseBaseline');
    reset(env);

    % Start high enough that the guardian has no reason to intervene
    env.State(2) = 2000;
    env.State(4) = 0;   % Isolate gravity
    env.State(3) = 0;   % Eliminate centrifugal lift
    env.State(5) = 0;

    % Command: [-1, 1] means minimum thrust (engine off), max right torque
    step(env, [-1; 1]);

    % One agent step advances control_decimation physics substeps, i.e. agent_dt, NOT dt.
    % With the engine off, vertical acceleration is purely gravity - evaluated AT 2000 m,
    % where inverse-square falloff makes it 0.23% below the surface value. The tolerance
    % below is loose enough to absorb the altitude changing slightly across the five
    % substeps, but far tighter than that 0.23%, so a regression to constant gravity
    % would still fail.
    p = env.params;
    g_at_alt = p.gravity * (p.r_lunar / (p.r_lunar + 2000))^2;
    expected_dy = -g_at_alt * p.agent_dt;
    verifyEqual(testCase, env.State(4), expected_dy, 'RelTol', 1e-4, ...
        'Action scaling failed to map -1 to 0 thrust, or the control decimation is desynced.');
end

function testShapingIsAppliedOncePerAgentStepWithGamma(testCase)
    % Shaping used to be summed per PHYSICS SUBSTEP as Phi(s') - Phi(s), undiscounted.
    % That form is only policy-invariant at gamma = 1, and the residual over a 3000-step
    % episode is of order (1-gamma)*|Phi|*horizon ~ 5 reward - the same size as the
    % landing bonus it is supposed to be neutral against. A per-substep gamma does not
    % fix it either, because gamma^(1/5) applied five times stops telescoping.
    %
    % This pins the corrected contract exactly: one application, at the agent rate, in
    % the discounted form.
    env = LunarLanderEnv('DenseBaseline', 'off');   % guardian off: no veto term
    reset(env);
    env.State = [10; 800; 5; -12; 0.04; 0.01; 6000; 250];

    p = env.params;
    w = get_reward_weights();

    x_before = env.State;
    phi0 = shaping_potential(x_before, p, w);

    action = [0.15; -0.2];
    [~, Reward] = step(env, action);

    % Reconstruct the fuel term, which is the only other contribution on a non-terminal,
    % non-vetoed step. Thrust is decided once per agent step, so it is constant across
    % the substeps.
    % Go through action_to_command rather than restating the map. An earlier version of
    % this test duplicated the formula, which made it a fourth copy of the control
    % interface - and it silently went stale the moment the map was extended to reach
    % full thrust for the powered-descent phase.
    u = action_to_command(action, x_before, p);
    fuel = (w.fuel_main * abs(u(1)) / p.max_main_thrust + ...
            w.fuel_rcs  * abs(u(2)) / p.max_side_torque) * p.dt ...
           * p.control_decimation / w.reward_scale;

    expected_shaping = (shaping_potential(env.State, p, w) - phi0) / w.reward_scale;

    verifyEqual(testCase, Reward - fuel, expected_shaping, 'AbsTol', 1e-9, ...
        ['Shaping is not being applied exactly once per agent step as ' ...
         'Phi(s'') - Phi(s). Applying it per physics substep, or reintroducing the ' ...
         'gamma factor, both break this.']);
end

function testTerminalPotentialIsZero(testCase)
    % Ng et al. require Phi = 0 at an absorbing state. If a terminated state kept its
    % potential, the shaping would charge a fast tilted arrival once and the crash
    % penalty would charge it again, and the two would need tuning against each other
    % forever.
    env = LunarLanderEnv('DenseBaseline', 'off');
    reset(env);
    % Just above the touchdown altitude, descending fast enough to terminate this step.
    env.State = [0; p_touchdown_alt() + 0.4; 0; -6; 0; 0; 6000; 250];

    p = env.params;
    w = get_reward_weights();
    phi0 = shaping_potential(env.State, p, w);

    [~, Reward, IsDone] = step(env, [-1; 0]);
    verifyTrue(testCase, IsDone, 'Test state failed to reach a terminal condition.');

    % With Phi(terminal) = 0 the shaping contribution is exactly -phi0/scale, which is
    % POSITIVE (the potential is always negative). If the implementation instead kept
    % Phi(s_terminal), the contribution would be strongly negative.
    verifyGreaterThan(testCase, Reward - crash_and_fuel_floor(), -phi0 / w.reward_scale - 1, ...
        'Terminal potential does not appear to be zeroed.');
end

function alt = p_touchdown_alt()
    pp = get_sim_params();
    alt = pp.touchdown_alt;
end

function v = crash_and_fuel_floor()
    % Loosest possible bound on the non-shaping terms of a terminal step.
    w = get_reward_weights();
    v = w.crash / w.reward_scale - 1;
end

function testHoverActionIsMassInvariant(testCase)
    % Gravity-compensated thrust: action 0 must command an exact hover whatever the fuel
    % state. Under the old raw-throttle map the same action produced 1.76 m/s^2 per unit
    % at launch mass and 5.26 near-empty - a 3x plant-gain drift the policy had to track
    % while also learning the task.
    p = get_sim_params();
    for fuel = [p.max_main_fuel, 10]
        env = LunarLanderEnv('DenseBaseline', 'off');
        reset(env);
        env.State = [0; 500; 0; 0; 0; 0; fuel; 20];
        step(env, [0; 0]);
        verifyLessThan(testCase, abs(env.State(4)), 0.05, ...
            sprintf(['Action 0 did not hold a hover at %g kg of main fuel; the ' ...
                     'action-to-acceleration map still depends on mass.'], fuel));
    end
end

function testRewardRouting(testCase)
    % Initializing with 'SparseOnly' must route to the sparse logic
    env = LunarLanderEnv('SparseOnly');
    reset(env);
    env.State(2) = 2000;   % Well clear of any terminal condition

    [~, Reward] = step(env, [0; 0]);

    verifyEqual(testCase, Reward, 0, ...
        'Environment failed to route to the SparseOnly reward scheme.');
end

function testGuardianModeToggle(testCase)
    % The A/B study depends on the guardian actually being removable. Drop the lander
    % into a state the guardian must rescue - low and falling fast - and command zero
    % thrust. With the guardian on, thrust is overridden; with it off, it is obeyed.
    x_lethal = [0; 40; 0; -45; 0; 0; 8000; 300];

    env_on = LunarLanderEnv('DenseBaseline', 'on');
    reset(env_on);
    env_on.State = x_lethal;
    [~, ~, ~, logs_on] = step(env_on, [-1; 0]);   % -1 => zero thrust commanded

    env_off = LunarLanderEnv('DenseBaseline', 'off');
    reset(env_off);
    env_off.State = x_lethal;
    [~, ~, ~, logs_off] = step(env_off, [-1; 0]);

    verifyGreaterThan(testCase, logs_on.Control(1), 0, ...
        'Guardian ON must override a zero-thrust command during a lethal descent.');
    verifyEqual(testCase, logs_off.Control(1), 0, 'AbsTol', 1e-9, ...
        'Guardian OFF must pass the zero-thrust command through untouched.');
    verifyTrue(testCase, logs_on.VetoActive, 'Guardian ON should report a veto here.');
    verifyFalse(testCase, logs_off.VetoActive, 'Guardian OFF must never report a veto.');

    verifyError(testCase, @() LunarLanderEnv('DenseBaseline', 'sometimes'), ...
        'LunarLanderEnv:UnknownGuardianMode', ...
        'An unknown guardian mode must be rejected, not silently defaulted.');
end

function testVetoPenaltyIsEdgeTriggered(testCase)
    % The single most damaging bug: the veto penalty was charged every timestep at 50 Hz,
    % costing -5.0 reward per second while the worst crash cost -7.5. Sustained guardian
    % authority must charge the penalty ONCE per engagement, not once per step.
    env = LunarLanderEnv('DenseBaseline', 'on');
    reset(env);

    % Hold the lander in a state that keeps the guardian continuously engaged
    env.State = [0; 30; 0; -40; 0; 0; 8000; 300];

    total_engagements = 0;
    for k = 1:10
        [~, ~, done, logs] = step(env, [-1; 0]);
        total_engagements = logs.VetoCount;
        if done
            break;
        end
    end

    % Across many steps of unbroken guardian authority the engagement count must stay
    % far below the number of physics substeps taken.
    substeps_taken = k * env.params.control_decimation;
    verifyLessThan(testCase, total_engagements, substeps_taken / 2, ...
        sprintf(['Veto penalty is not edge-triggered: %d engagements over %d substeps. ' ...
                 'A level-triggered flag makes crashing the optimal policy.'], ...
                total_engagements, substeps_taken));
    verifyGreaterThan(testCase, total_engagements, 0, ...
        'The guardian should have engaged at least once in a lethal descent.');
end

function testEpisodeReachesTheGround(testCase)
    % Previously every episode terminated at ~10 m via a veto-counter abort that scored
    % the outcome as a crash, leaving the touchdown check at y <= 2.0 as dead code.
    env = LunarLanderEnv('DenseBaseline', 'on');
    reset(env);

    % Phase 1 style start, then fly a steady moderate throttle to the surface
    env.State = [0; 50; 0; -2; 0; 0; 8000; 300];

    done = false;
    for k = 1:env.params.max_agent_steps
        [~, ~, done] = step(env, [-0.1; 0]);
        if done
            break;
        end
    end

    verifyTrue(testCase, done, 'Episode should terminate rather than run to the step cap.');
    verifyNotEqual(testCase, env.Outcome, 'flying', 'A terminated episode must record an outcome.');
    verifyLessThan(testCase, env.State(2), 10.0, ...
        'Episode must terminate near the surface, not abort at 10 m altitude.');
end

function testFuelNeverGoesNegative(testCase)
    % Euler integration can otherwise drive tank mass below zero, silently corrupting
    % every mass and inertia term downstream of it.
    env = LunarLanderEnv('DenseBaseline', 'off');
    reset(env);
    env.State = [0; 3000; 0; 0; 0; 0; 5; 0.1];   % Almost dry, full throttle

    for k = 1:50
        [~, ~, done] = step(env, [1; 1]);
        verifyGreaterThanOrEqual(testCase, env.State(7), 0, 'Main fuel went negative.');
        verifyGreaterThanOrEqual(testCase, env.State(8), 0, 'RCS fuel went negative.');
        if done
            break;
        end
    end
end

function testRewardsAreFinite(testCase)
    % Guards against NaN/Inf leaking in from the barrier maths (division by a_max, etc).
    env = LunarLanderEnv('DenseBaseline', 'on');
    reset(env);
    for k = 1:200
        a = 2 * rand(2, 1) - 1;
        [obs, r, done] = step(env, a);
        verifyTrue(testCase, isfinite(r), 'Reward must always be finite.');
        verifyTrue(testCase, all(isfinite(obs)), 'Observations must always be finite.');
        if done
            break;
        end
    end
end
