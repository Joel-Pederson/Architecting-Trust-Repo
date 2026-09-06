function tests = safety_sidecar_recovery_test
%SAFETY_SIDECAR_RECOVERY_TEST - The paper's central claim, as an executable assertion.
%
% Architecting Trust argues that a runtime barrier bounds an autonomous system whose
% nominal controller has degraded, WITHOUT needing to anticipate how it degraded. That is
% a measurable property, so it belongs in the test suite rather than only in a results
% table that could silently rot as the sidecar is edited.
%
% Measured with run_fault_injection_study: an altimeter reading 10 m high takes the
% classical pilot from 100% landings to 0%, and the sidecar restores it to 100%.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    testCase.TestData.params = get_sim_params();
end

function testSidecarRecoversFromAPerceptionFault(testCase)
    % An altimeter bias is invisible to the pilot: it brakes on a altitude it does not
    % have. The sidecar reads TRUE state, which is the Perception Gatekeeper boundary.
    p = testCase.TestData.params;
    BIAS = 10;   % metres

    off = fly(p, 'off', BIAS, 12);
    on  = fly(p, 'on',  BIAS, 12);

    fprintf('\n  altimeter +%g m : guardian OFF %.0f%% landed (%.2f m/s), ON %.0f%% (%.2f m/s)\n', ...
        BIAS, 100*off.land, off.impact, 100*on.land, on.impact);

    verifyLessThan(testCase, off.land, 0.2, ...
        'The unprotected pilot is supposed to FAIL this fault; if it does not, the fault is too weak to test anything.');
    verifyGreaterThan(testCase, on.land, 0.8, ...
        sprintf(['The sidecar recovered only %.0f%% of landings under a %g m altimeter ' ...
                 'bias. This is the paper''s central claim.'], 100*on.land, BIAS));

    % THE BOUND, not the average. "Mean impact halved" is a performance claim; "no
    % episode exceeded the design limit" is a safety claim, and only the second supports
    % runtime assurance. Measured: worst case under the sidecar stays at 0.66 m/s across
    % an 8x range of fault magnitude, while unprotected it reaches 3.14 m/s.
    fprintf('  worst-case impact: OFF %.2f m/s, ON %.2f m/s (limit %.2f)\n', ...
        off.worst, on.worst, p.max_touchdown_dy);
    verifyLessThan(testCase, on.worst, p.max_touchdown_dy, ...
        sprintf(['Worst-case impact under the sidecar was %.2f m/s, outside the %.2f m/s ' ...
                 'design limit. The barrier must BOUND the outcome, not just improve ' ...
                 'the average.'], on.worst, p.max_touchdown_dy));
end

function testSidecarIsNonIntrusiveWhenTheControllerIsHealthy(testCase)
    % The counterpart claim, and the obvious reviewer objection: a barrier that buys
    % safety by costing capability is a poor trade. With no fault injected, attaching the
    % sidecar must not reduce the landing rate.
    p = testCase.TestData.params;
    off = fly(p, 'off', 0, 12);
    on  = fly(p, 'on',  0, 12);
    fprintf('  no fault       : guardian OFF %.0f%% landed, ON %.0f%%\n', ...
        100*off.land, 100*on.land);
    verifyGreaterThanOrEqual(testCase, on.land, off.land - 0.1, ...
        'Attaching the sidecar cost landings on a healthy controller.');
end

function m = fly(p, guardian, alt_bias, n)
    env = LunarLanderEnv('DenseBaseline', guardian);
    env.CurriculumWeights = [1 0 0];
    rng(101);
    landed = 0; impacts = [];
    for ep = 1:n
        reset(env);
        for i = 1:p.max_agent_steps
            s = env.State;
            s_seen = s; s_seen(2) = s(2) + alt_bias;
            u = scripted_pilot(s_seen, p);
            % Mass comes from TRUE state: the fault is in the altimeter, not the gauge.
            [~,~,done] = step(env, command_to_action(u, s, p));
            if done, break; end
        end
        landed = landed + strcmp(env.Outcome, 'landed');
        if any(strcmp(env.Outcome, {'landed','crashed'}))
            impacts(end+1) = sqrt(env.State(3)^2 + env.State(4)^2); %#ok<AGROW>
        end
    end
    m.land = landed / n;
    m.impact = mean(impacts);
    if isempty(impacts), m.worst = NaN; else, m.worst = max(impacts); end
end
