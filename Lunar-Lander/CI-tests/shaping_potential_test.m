function tests = shaping_potential_test
%SHAPING_POTENTIAL_TEST - Executable specification of the reward-shaping potential.
%
% Written after five full training runs (four PPO, one SAC, ~20,000 episodes) produced
% zero landings. The cause was not the algorithm, the budget, or the exploration
% schedule. It was that the potential normalised velocity by 100 m/s, so the entire span
% from a FAILING 2 m/s lateral drift to a PASSING 0.5 m/s was worth 0.06 reward against a
% +5.00 landing bonus - and lateral velocity is the term that decides almost every
% episode. An ideal vertical descent law lands 100% of Phase 1 with zero lateral offset
% and 18.3% with the real init_dx ~ N(0,2), matching P(|dx| < 0.5) for that distribution.
%
% These tests cost milliseconds. The runs that found the problem cost hours.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    testCase.TestData.params  = get_sim_params();
    testCase.TestData.weights = get_reward_weights();
end

function testLateralVelocityHasUsableGradientNearThePad(testCase)
    % THE ONE THAT BIT. Closing a failing lateral drift down to a passing one must be
    % worth something comparable to the landing bonus, not a rounding error.
    p = testCase.TestData.params;
    w = testCase.TestData.weights;

    failing = state_at(2.0, 2.0, -0.5, 0.0);   % 2.0 m/s lateral - four times the limit
    passing = state_at(2.0, 0.5, -0.5, 0.0);   % 0.5 m/s lateral - exactly at the limit

    gradient = (shaping_potential(passing, p, w) - shaping_potential(failing, p, w)) ...
               / w.reward_scale;

    fprintf('\n  lateral 2.0 -> 0.5 m/s at the pad is worth %+.3f reward\n', gradient);
    verifyGreaterThan(testCase, gradient, 1.0, ...
        sprintf(['Closing the lateral drift from 4x the limit to exactly the limit is ' ...
                 'worth only %.3f. Under the descent term alone this was 0.06, which is ' ...
                 'why 20,000 training episodes produced no landings.'], gradient));
end

function testApproachTermIsOffAtAltitude(testCase)
    % The approach term must not influence high-altitude flight, or it would fight the
    % descent guidance during the phase where descending fast is correct.
    p = testCase.TestData.params;
    w = testCase.TestData.weights;

    high_clean  = state_at(p.approach_gate_alt + 1, 0.0, -20, 0.0);
    high_sloppy = state_at(p.approach_gate_alt + 1, 5.0, -20, 0.0);

    % At this altitude the only difference between them should come from the descent
    % term's own speed norm, which is small - not from the touchdown criteria.
    d = abs(shaping_potential(high_clean, p, w) - shaping_potential(high_sloppy, p, w)) ...
        / w.reward_scale;
    verifyLessThan(testCase, d, 0.3, ...
        'The approach term is still active above its gate altitude.');
end

function testFadeInIsContinuous(testCase)
    % A step in the potential is a step in the reward, and the agent will find it. Sweep
    % across the gate altitude and require no jump.
    p = testCase.TestData.params;
    w = testCase.TestData.weights;

    alts = linspace(p.approach_gate_alt + 5, p.approach_gate_alt - 5, 201);
    phis = arrayfun(@(a) shaping_potential(state_at(a, 2.0, -2.0, 0.05), p, w), alts);
    jumps = abs(diff(phis)) / w.reward_scale;

    verifyLessThan(testCase, max(jumps), 0.02, ...
        'The approach term fades in discontinuously; the agent can exploit the step.');
end

function testPotentialCannotDwarfTerminalOutcomes(testCase)
    % The saturation caps exist so shaping cannot outbid the landing. They were 5.0/3.0,
    % which let Phi reach -32 against terminal outcomes of +/-5: flying out of bounds
    % actually cost -37, so "never climb" was the strongest lesson in the landscape, and
    % it is satisfied by not thrusting - which crashes.
    p = testCase.TestData.params;
    w = testCase.TestData.weights;

    runaway = [50000; 40000; 300; 250; pi; 5; 0; 0];   % far outside any legal state
    magnitude = abs(shaping_potential(runaway, p, w)) / w.reward_scale;

    fprintf('  worst-case |Phi| = %.2f scaled (terminal outcomes are +/-%.2f)\n', ...
        magnitude, abs(w.crash) / w.reward_scale);
    verifyLessThan(testCase, magnitude, 3 * abs(w.crash) / w.reward_scale, ...
        sprintf(['Worst-case potential is %.1f against terminal outcomes of %.1f. ' ...
                 'Shaping this large dominates the critic and buries the task.'], ...
                 magnitude, abs(w.crash) / w.reward_scale));
end

function testPotentialImprovesTowardASafeTouchdown(testCase)
    % Basic sanity: the potential must actually rank a good approach above a bad one on
    % every gate term independently.
    p = testCase.TestData.params;
    w = testCase.TestData.weights;
    base = state_at(5.0, 2.0, -3.0, 0.30);

    better = {state_at(5.0, 0.2, -3.0, 0.30), ...   % less lateral
              state_at(5.0, 2.0, -0.5, 0.30), ...   % less vertical
              state_at(5.0, 2.0, -3.0, 0.02)};      % less tilt
    labels = {'lateral', 'vertical', 'tilt'};

    phi_base = shaping_potential(base, p, w);
    for i = 1:numel(better)
        verifyGreaterThan(testCase, shaping_potential(better{i}, p, w), phi_base, ...
            sprintf('Improving %s did not raise the potential.', labels{i}));
    end
end

function x = state_at(y, dx, dy, theta)
    p = get_sim_params();
    x = [0; y; dx; dy; theta; 0; p.max_main_fuel; p.max_rcs_fuel];
end
