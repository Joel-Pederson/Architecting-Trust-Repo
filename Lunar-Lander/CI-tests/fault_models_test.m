function tests = fault_models_test
%FAULT_MODELS_TEST - The fault injection study's conclusions rest on these models.
%
% If a fault does not corrupt what it claims to, the study still runs and still produces a
% table - it just measures something other than what the paper says. These are cheap
% assertions against an expensive silent failure.
%
% They also pin the PERCEPTION BOUNDARY: a sensor fault must alter only the controller's
% view, never the true state, because the sidecar's authority comes from reading the truth
% when the nominal controller cannot.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    p = get_sim_params();
    testCase.TestData.params = p;
    testCase.TestData.x = [12; 300; -3; -8; 0.05; 0.01; p.max_main_fuel; p.max_rcs_fuel];
end

function testEachFaultCorruptsOnlyItsOwnChannel(testCase)
    % Cross-wiring these would change what the study measures without changing that it
    % produces a plausible-looking table.
    x = testCase.TestData.x;

    alt = apply_sensor_fault(x, 'alt_bias', 25);
    verifyEqual(testCase, alt(2), x(2) + 25, 'AbsTol', 1e-12, 'alt_bias must shift altitude.');
    verifyEqual(testCase, alt([1 3:8]), x([1 3:8]), 'alt_bias touched another channel.');

    vel = apply_sensor_fault(x, 'vel_bias', 0.5);
    verifyEqual(testCase, vel(4), x(4) * 0.5, 'AbsTol', 1e-12, ...
        'vel_bias must scale the descent rate.');
    verifyEqual(testCase, vel([1:3 5:8]), x([1:3 5:8]), 'vel_bias touched another channel.');

    % An actuator fault is applied to the command, so the observation passes through.
    thr = apply_sensor_fault(x, 'thrust_loss', 0.4);
    verifyEqual(testCase, thr, x, 'thrust_loss must not alter the perceived state.');

    none = apply_sensor_fault(x, 'none', 99);
    verifyEqual(testCase, none, x, '''none'' must be a passthrough.');
end

function testFaultsAreDirectionallyDangerous(testCase)
    % Sign errors are the easy mistake and they invert the experiment: an altimeter that
    % reads LOW makes the controller brake early, which is safe. The study only means
    % something if the fault pushes toward danger.
    x = testCase.TestData.x;

    biased = apply_sensor_fault(x, 'alt_bias', 10);
    verifyGreaterThan(testCase, biased(2), x(2), ...
        'alt_bias must make the controller believe it is HIGHER than it is.');

    seen = apply_sensor_fault(x, 'vel_bias', 0.7);
    verifyLessThan(testCase, abs(seen(4)), abs(x(4)), ...
        'vel_bias must make the controller UNDER-read its descent rate.');
end

function testAltimeterFreezeGrowsWorseOnApproach(testCase)
    % The harsher perception fault: below the threshold the altimeter stops updating, so
    % the error grows as the ground approaches rather than staying constant.
    p = testCase.TestData.params;
    THRESH = 50;

    high = [0; 120; 0; -5; 0; 0; p.max_main_fuel; p.max_rcs_fuel];
    verifyEqual(testCase, apply_sensor_fault(high, 'alt_freeze', THRESH), high, ...
        'alt_freeze must be inert above its threshold.');

    e50 = freeze_error(p, 50, THRESH);
    e40 = freeze_error(p, 40, THRESH);
    e20 = freeze_error(p, 20, THRESH);
    e05 = freeze_error(p,  5, THRESH);

    verifyEqual(testCase, e50, 0, 'AbsTol', 1e-12);
    verifyGreaterThan(testCase, e20, e40, ...
        'alt_freeze error must GROW as the lander descends.');
    verifyGreaterThan(testCase, e05, e20, ...
        'alt_freeze error must GROW as the lander descends.');
end

function testDelayReturnsAnEarlierState(testCase)
    p = testCase.TestData.params;
    hist = {};
    for k = 1:30
        hist{end+1} = [0; 500 - 10*k; 0; -10; 0; 0; ...
                       p.max_main_fuel; p.max_rcs_fuel]; %#ok<AGROW>
    end
    now_state = [0; 190; 0; -10; 0; 0; p.max_main_fuel; p.max_rcs_fuel];

    seen = apply_sensor_fault(now_state, 'delay', 10, hist);
    verifyEqual(testCase, seen, hist{20}, ...
        'delay must return the state from `magnitude` steps ago.');

    % Empty history must not error; it degrades to no delay.
    verifyEqual(testCase, apply_sensor_fault(now_state, 'delay', 10, {}), now_state, ...
        'delay with no history should pass the current state through.');
end

function testUnknownFaultErrors(testCase)
    % A typo in a fault name must fail loudly rather than silently running an unfaulted
    % cell and reporting it as a fault result.
    verifyError(testCase, ...
        @() apply_sensor_fault(testCase.TestData.x, 'alt_biass', 10), ...
        'applySensorFault:UnknownFault', ...
        'A misspelled fault name must error, not silently pass the state through.');
end

function testAltimeterBiasActuallyDefeatsThePilot(testCase)
    % End to end: the fault must be strong enough to break an otherwise perfect
    % controller, or the recovery result it supports is meaningless.
    p = testCase.TestData.params;
    clean  = fly(p, 'off', 'none',     0,  6);
    faulty = fly(p, 'off', 'alt_bias', 20, 6);
    fprintf('\n  pilot, guardian OFF: clean %.0f%% -> alt_bias 20 m %.0f%%\n', ...
        100*clean, 100*faulty);
    verifyGreaterThan(testCase, clean, 0.8, 'Baseline pilot should land reliably.');
    verifyLessThan(testCase, faulty, 0.3, ...
        'A 20 m altimeter bias should defeat the unprotected pilot.');
end

function e = freeze_error(p, y, thresh)
% How far the frozen altimeter is from the truth at altitude y.
    x = [0; y; 0; -5; 0; 0; p.max_main_fuel; p.max_rcs_fuel];
    seen = apply_sensor_fault(x, 'alt_freeze', thresh);
    e = abs(seen(2) - y);
end


function rate = fly(p, guardian, fault, mag, n)
    env = LunarLanderEnv('DenseBaseline', guardian);
    env.CurriculumWeights = [1 0 0];
    rng(101);
    landed = 0;
    for ep = 1:n
        reset(env);
        for i = 1:p.max_agent_steps
            s = env.State;
            u = scripted_pilot(apply_sensor_fault(s, fault, mag), p);
            [~, ~, done] = step(env, command_to_action(u, s, p));
            if done, break; end
        end
        landed = landed + strcmp(env.Outcome, 'landed');
    end
    rate = landed / n;
end
