function tests = entry_points_test
% ENTRY_POINTS_TEST Guards the user-facing entry points and the scenario vocabulary.
%
% These tests exist because of two bugs that shipped past both `checkcode` and the rest of
% the suite. Rewriting run_trained_agent to take scenario NAMES silently dropped its option
% defaults, so every call errored on the first use; and the README-documented workflow can
% rot without any test noticing, because nothing else in CI opens those files.
%
% Neither a lint pass nor a physics test looks at whether the front door opens. This does.

    tests = functiontests(localfunctions);
end


function setupOnce(testCase)
    here = fileparts(fileparts(mfilename('fullpath')));   % Lunar-Lander/
    addpath(genpath(here));
    testCase.TestData.root = here;
end


function testDocumentedEntryPointsExist(testCase)
% Every command the README tells a new user to type must resolve on the path. A renamed or
% deleted script otherwise fails only for the person following the instructions.
    documented = { 'train_pipeline', 'evaluate_final_agent', 'run_trained_agent', ...
                   'demo_sidecar_rescue', 'demo_agent_rescue', 'demo_reel', ...
                   'run_fault_injection_study', 'run_algorithm_trade', ...
                   'generate_demonstrations', 'pretrain_actor_supervised', ...
                   'dagger_refine', 'main_simulation', 'phase_from_name' };
    for k = 1:numel(documented)
        verifyNotEmpty(testCase, which(documented{k}), ...
            sprintf('README documents %s, but it is not on the path.', documented{k}));
    end
end


function testMissingAgentFilesFailInformatively(testCase)
% A fresh clone has no .mat files - they are gitignored. The playback entry points must say
% so and name the remedy, not fail somewhere inside `load`.
    missing = struct('agent_file', 'no_such_agent_file_12345.mat', 'animate', false);

    verifyError(testCase, @() run_trained_agent([], missing), ...
        'runTrainedAgent:NoAgent', ...
        'run_trained_agent must reject a missing agent file by identifier.');

    verifyError(testCase, @() evaluate_final_agent(missing), ...
        'evaluateFinalAgent:NoAgent', ...
        'evaluate_final_agent must reject a missing agent file by identifier.');

    verifyError(testCase, @() demo_agent_rescue('terminal', missing), ...
        'demoAgentRescue:NoAgent', ...
        'demo_agent_rescue must reject a missing agent file by identifier.');
end


function testScenarioNamesResolveToPhases(testCase)
% The scenario vocabulary is the interface a reader actually types. Numbers still work, but
% run_trained_agent('orbit') is the documented form and must keep meaning phase 4.
    verifyEqual(testCase, phase_from_name('touchdown'), 1);
    verifyEqual(testCase, phase_from_name('approach'),  2);
    verifyEqual(testCase, phase_from_name('terminal'),  3);
    verifyEqual(testCase, phase_from_name('orbit'),     4);

    % Case and spacing are not part of the interface.
    verifyEqual(testCase, phase_from_name('ORBIT'),  4);
    verifyEqual(testCase, phase_from_name(' orbit '), 4);
    verifyEqual(testCase, phase_from_name('pdi'),    4);

    % Numeric indices are still accepted, so older scripts keep working.
    for k = 1:4
        verifyEqual(testCase, phase_from_name(k), k);
    end
end


function testUnknownScenarioNamesListTheValidOnes(testCase)
% A typo should print the vocabulary rather than an index error. This is the whole reason
% the lookup exists: '12,4' meant nothing to a reader, and neither does a bare failure.
    verifyError(testCase, @() phase_from_name('descend'), 'phaseFromName:Unknown');

    % And the message itself must carry the vocabulary, not just the identifier.
    msg = '';
    try
        phase_from_name('descend');
    catch ME
        msg = ME.message;
    end
    verifySubstring(testCase, msg, 'orbit', ...
        'The error must list the valid scenario names.');

    verifyError(testCase, @() phase_from_name(9), 'phaseFromName:BadIndex');
end


function testEveryPhaseHasAStepBudget(testCase)
% phase_from_name and params.phase_max_steps have to agree on how many phases exist.
% A four-phase vocabulary against a three-entry budget indexes past the end at run time.
    p = get_sim_params();
    verifyEqual(testCase, numel(p.phase_max_steps), 4, ...
        'phase_from_name defines four scenarios; params must budget four phases.');
    verifyGreaterThan(testCase, p.phase_max_steps(4), p.phase_max_steps(3), ...
        'The powered descent is far longer than the terminal phase and needs a bigger budget.');
end


function verifySubstring(testCase, haystack, needle, msg)
    verifyTrue(testCase, contains(haystack, needle), msg);
end
