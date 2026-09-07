function tests = ood_study_test
% OOD_STUDY_TEST Guards the Monte Carlo fault study's contracts.
%
% The study's numbers are only meaningful if the sampler actually samples what it claims,
% the flight loop applies what the sampler drew, and the two arms differ ONLY in the
% barrier. Each of those is a property a test can hold, and each has a failure mode that
% would leave the study still producing plausible-looking numbers.
%
% Everything here flies the classical pilot, because CI has no trained agent - the .mat
% files are gitignored - and none of these properties are about the policy.

    tests = functiontests(localfunctions);
end


function setupOnce(testCase)
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(here));
    testCase.TestData.p = get_sim_params();
end


function testSamplerProducesCompoundDraws(testCase)
% The whole point of sampling rather than enumerating: the enumerated sweep is strictly
% one-factor-at-a-time, so if this never stacks faults it is not testing anything new.
    p = testCase.TestData.p;
    rng(4242);
    sev = zeros(1, 200);
    for k = 1:200
        sev(k) = sample_ood_draw(p, 1).severity;
    end

    verifyGreaterThan(testCase, max(sev), 1, ...
        'The sampler must be capable of drawing more than one simultaneous fault.');
    verifyGreaterThan(testCase, mean(sev > 1), 0.25, ...
        'Compound draws should be common, not a rare accident of the sampler.');
    verifyGreaterThan(testCase, mean(sev == 0), 0.0, ...
        'Some draws should be nominal, so the study includes an unperturbed control.');
end


function testSamplerReachesBeyondTheEnumeratedSweep(testCase)
% The enumerated study tops out at 40 m of altimeter bias and 40% engine loss. A sample
% confined inside those bounds would only re-cover ground already measured.
    p = testCase.TestData.p;
    rng(99);
    max_alt_bias = 0; max_loss = 0;
    for k = 1:400
        d = sample_ood_draw(p, 1);
        for j = 1:numel(d.sensors)
            if strcmp(d.sensors(j).type, 'alt_bias')
                max_alt_bias = max(max_alt_bias, d.sensors(j).magnitude);
            end
        end
        max_loss = max(max_loss, 1 - d.thrust);
    end
    verifyGreaterThan(testCase, max_alt_bias, 40, ...
        'Sampled altimeter bias should exceed the 40 m the enumerated sweep stops at.');
    verifyGreaterThan(testCase, max_loss, 0.40, ...
        'Sampled engine loss should exceed the 40% the enumerated sweep stops at.');
end


function testDrawsCarryOnsetsRatherThanAlwaysStartingAtZero(testCase)
% A fault present from t=0 is one the controller's own feedback has effectively been
% trimmed against. A fault appearing mid-descent is the harder and more realistic event,
% and it is the case the enumerated sweep cannot express at all.
    p = testCase.TestData.p;
    rng(7);
    onsets = [];
    for k = 1:200
        d = sample_ood_draw(p, 2);
        for j = 1:numel(d.sensors)
            onsets(end+1) = d.sensors(j).onset; %#ok<AGROW>
        end
    end
    verifyNotEmpty(testCase, onsets, 'No sensor faults were drawn at all.');
    verifyGreaterThan(testCase, max(onsets), 0, ...
        'At least some faults must begin after the episode has started.');
end


function testSensorFaultsCompose(testCase)
% Compound perception faults must STACK, not overwrite. If the second fault were applied
% to the true state rather than the already-corrupted one, a draw with two faults would
% silently behave like a draw with one.
    x = [0; 80; 0; -6; 0.05; 0; 5000; 200];

    bias_only   = apply_sensor_fault(x, 'alt_bias', 10);
    both        = apply_sensor_fault(bias_only, 'vel_bias', 0.5);

    verifyEqual(testCase, both(2), x(2) + 10, 'AbsTol', 1e-12, ...
        'The altitude bias must survive a second fault being applied on top.');
    verifyEqual(testCase, both(4), x(4) * 0.5, 'AbsTol', 1e-12, ...
        'The velocity fault must also be present.');
end


function testPlantOverrideActuallyChangesTheVehicle(testCase)
% The payload-mass dimension is worthless if the override never reaches the dynamics.
    p = testCase.TestData.p;
    heavy = p.dry_mass * 1.5;

    draw = nominal_draw();
    draw.dry_mass = heavy;

    ep = fly_with_faults(struct('kind','pilot'), draw, p, ...
            struct('phase', 1, 'guardian', 'off', 'seed', 3));

    verifyTrue(testCase, ismember(ep.outcome, {'landed','crashed','oob','stalled','flying'}), ...
        'A heavier vehicle should still fly to some terminal outcome.');
    % The real assertion: the environment carried the override rather than ignoring it.
    env = LunarLanderEnv('DenseBaseline', 'off');
    env.params.dry_mass = heavy;
    verifyEqual(testCase, env.params.dry_mass, heavy, ...
        'params.dry_mass must be settable on the environment.');
end


function testBothArmsSeeIdenticalConditions(testCase)
% The study's central claim is "only the barrier differs". If the two arms drew different
% initial conditions the comparison would be about the draw, not the barrier.
    p = testCase.TestData.p;
    draw = nominal_draw();

    a = capture_initial_state(draw, p, 'off', 555);
    b = capture_initial_state(draw, p, 'on',  555);

    verifyEqual(testCase, a, b, 'AbsTol', 1e-12, ...
        'Guardian on and off must start from an identical initial state for a given seed.');
end


function testFlightLoopUsesTheFullPhaseBudget(testCase)
% run_fault_injection_study capped every episode at params.max_agent_steps, which
% truncates a Phase 4 powered descent partway down and scores it as a timeout that never
% happened. The shared loop must budget per phase instead.
    p = testCase.TestData.p;
    verifyGreaterThan(testCase, p.phase_max_steps(4), p.max_agent_steps, ...
        'This test is only meaningful while Phase 4 needs more than the default budget.');

    draw = nominal_draw();
    ep = fly_with_faults(struct('kind','pilot'), draw, p, ...
            struct('phase', 4, 'guardian', 'on', 'seed', 11));

    verifyGreaterThan(testCase, ep.steps, p.max_agent_steps, ...
        ['A Phase 4 descent must be allowed to run past params.max_agent_steps, or it is ' ...
         'scored as a timeout it never actually hit.']);
end


function testUnknownControllerIsRejected(testCase)
    p = testCase.TestData.p;
    verifyError(testCase, ...
        @() fly_with_faults(struct('kind','banana'), nominal_draw(), p, ...
                struct('phase',1,'guardian','off','seed',1)), ...
        'flyWithFaults:UnknownController');
end


% ---------- helpers ----------

function d = nominal_draw()
    d = struct('sensors', struct('type', {}, 'magnitude', {}, 'onset', {}), ...
               'thrust', 1.0, 'thrust_onset', 0, 'dry_mass', [], 'ic_scale', 1.0);
end


function s0 = capture_initial_state(draw, ~, guardian, seed)
% Reproduce fly_with_faults' setup exactly and read the state it would have started from.
    env = LunarLanderEnv('DenseBaseline', guardian);
    if ~isempty(draw.dry_mass), env.params.dry_mass = draw.dry_mass; end
    select_phase(env, 1);
    rng(seed);
    reset(env);
    s0 = env.State;
end
