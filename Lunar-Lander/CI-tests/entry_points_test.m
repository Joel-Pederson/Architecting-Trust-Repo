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


function testSelectPhaseIsSizedFromParamsNotHardcoded(testCase)
% The regression this function was extracted to prevent. Nine call sites had each written
% their own one-hot and two had hardcoded three phases, so adding the powered descent
% silently made it unselectable in run_fault_injection_study - the script the paper's
% central result comes from.
    p = get_sim_params();
    n = numel(p.phase_max_steps);

    for ph = 1:n
        env = LunarLanderEnv('DenseBaseline', 'off');
        idx = select_phase(env, ph);
        verifyEqual(testCase, idx, ph);
        verifyEqual(testCase, numel(env.CurriculumWeights), n, ...
            'The weight vector must span every configured phase, not a hardcoded three.');
        verifyEqual(testCase, env.CurriculumWeights, double((1:n) == ph), ...
            'select_phase must select exactly the requested phase.');
    end

    % Names work here too, so callers need not resolve them first.
    env = LunarLanderEnv('DenseBaseline', 'off');
    verifyEqual(testCase, select_phase(env, 'orbit'), 4);
    verifyEqual(testCase, env.CurriculumWeights, double((1:n) == 4));

    % And a phase the params do not configure is refused rather than silently producing
    % an all-zero weight vector for the environment to normalise by its own sum.
    env = LunarLanderEnv('DenseBaseline', 'off');
    verifyError(testCase, @() select_phase(env, n + 1), ...
        'phaseFromName:BadIndex');
end


function testEveryPhaseIsReachableByTheFaultStudy(testCase)
% run_fault_injection_study used to hardcode (1:3), so opts.phases = 4 produced a weight
% vector of zeros. Resetting such an environment yields NaN thresholds, which is a silent
% wrong answer rather than an error. Assert every configured phase actually resets.
    p = get_sim_params();
    for ph = 1:numel(p.phase_max_steps)
        env = LunarLanderEnv('DenseBaseline', 'on');
        select_phase(env, ph);
        rng(7);
        reset(env);
        verifyEqual(testCase, env.Phase, ph, ...
            sprintf('Resetting a phase-%d environment must actually start phase %d.', ph, ph));
        verifyTrue(testCase, all(isfinite(env.State)), ...
            'A zero or unnormalised weight vector shows up as a non-finite initial state.');
    end
end


function testPhaseLandingRatesReportsPerPhase(testCase)
% The shared measurement that train_pipeline, dagger_refine and evaluate_final_agent each
% used to implement separately. Flown with an UNTRAINED agent on purpose: no .mat files
% exist in CI, and the contract under test is the shape and per-phase separation of the
% result, not how well anything flies.
    p = get_sim_params();
    env = LunarLanderEnv('DenseBaseline', 'off');
    agent = build_agent('td3', getObservationInfo(env), getActionInfo(env), p.agent_dt);

    stats = phase_landing_rates(agent, p, ...
        struct('n_episodes', 1, 'phases', 1:2, 'seed', 11));

    verifyEqual(testCase, stats.phases, 1:2);
    verifyEqual(testCase, numel(stats.rate), 2, ...
        'One rate per requested phase, never pooled across phases.');
    verifyEqual(testCase, numel(stats.impact), 2);
    verifyEqual(testCase, numel(stats.worst), 2);
    verifyTrue(testCase, all(stats.rate >= 0 & stats.rate <= 1), ...
        'A landing rate is a fraction.');

    % Where an impact was recorded at all, the worst cannot be better than the mean.
    seen = isfinite(stats.impact);
    verifyTrue(testCase, all(stats.worst(seen) >= stats.impact(seen) - 1e-12), ...
        'The worst impact cannot be better than the mean impact.');
end


function testRolloutBudgetsThePhaseItActuallyDrew(testCase)
% THE BUG THIS EXISTS TO CATCH.
%
% rollout_episode used to default its step budget to params.max_agent_steps - the TRAINING
% episode cap of 3000. A Phase 4 powered descent needs ~8,900, so any caller that did not
% pass a budget explicitly had every Phase 4 episode truncated partway down and scored as a
% timeout that never happened.
%
% That was invisible while callers only evaluated phases 1-3, and became a silent WRONG
% ANSWER the moment evaluate_policy started iterating every configured phase: it reported a
% Phase 4 result while guaranteeing Phase 4 could not pass. A behavioural test on landing
% rate would not have caught it - the number looked plausible, it was just impossible.
    p = get_sim_params();
    verifyGreaterThan(testCase, p.phase_max_steps(4), p.max_agent_steps, ...
        'This test only means something while Phase 4 outlasts the training cap.');

    for ph = 1:numel(p.phase_max_steps)
        env = LunarLanderEnv('DenseBaseline', 'off');
        select_phase(env, ph);
        rng(3);
        reset(env);
        % The budget rollout_episode WOULD choose, resolved the same way it resolves it.
        verifyEqual(testCase, p.phase_max_steps(env.Phase), p.phase_max_steps(ph), ...
            'The environment must report the phase that was selected.');
        verifyGreaterThanOrEqual(testCase, p.phase_max_steps(ph), 1, ...
            'Every configured phase needs a positive step budget.');
    end
end


function testEveryEntryPointResolvesPhasesThroughOnePlace(testCase)
% Nine call sites once hand-wrote the curriculum one-hot and two of them went stale. Outside
% the test suite there should now be no hand-written weight vectors at all - the phase
% vocabulary has one implementation and this asserts nobody has quietly added a tenth.
    root = fileparts(fileparts(mfilename('fullpath')));
    offenders = {};

    areas = {fullfile(root, 'core'), fullfile(root, 'RL-training-harness'), ...
             fullfile(root, 'flight-code'), root};
    for a = 1:numel(areas)
        files = dir(fullfile(areas{a}, '**', '*.m'));
        for k = 1:numel(files)
            f = fullfile(files(k).folder, files(k).name);
            if contains(f, [filesep 'CI-tests' filesep]), continue; end
            % LunarLanderEnv declares the property; select_phase is the one sanctioned
            % place that assigns it, and this rule exists to route everyone through it.
            if any(strcmp(files(k).name, {'LunarLanderEnv.m', 'select_phase.m'}))
                continue;
            end
            lines = strsplit(fileread(f), newline);
            code  = regexprep(lines, '%.*$', '');
            hit = ~cellfun(@isempty, regexp(code, 'CurriculumWeights\s*=', 'once'));
            % evaluate_policy legitimately installs the uniform EVALUATION mix.
            hit = hit & cellfun(@isempty, regexp(code, 'eval_curriculum_weights', 'once'));
            if any(hit)
                offenders{end+1} = files(k).name; %#ok<AGROW>
            end
        end
    end

    verifyEmpty(testCase, offenders, sprintf( ...
        ['These files assign CurriculumWeights directly instead of calling select_phase: ' ...
         '%s. Hand-written one-hots are how the phase count went stale twice.'], ...
        strjoin(unique(offenders), ', ')));
end


function verifySubstring(testCase, haystack, needle, msg)
    verifyTrue(testCase, contains(haystack, needle), msg);
end
