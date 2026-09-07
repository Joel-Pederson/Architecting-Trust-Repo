function tests = scripted_pilot_test
%SCRIPTED_PILOT_TEST - Landing-performance floor for the classical reference controller.
%
% The pilot is the paper's non-RL baseline, so a silent regression in it would
% mis-state the comparison the whole trade study rests on. It has regressed silently
% before: an earlier attempt to clamp the lateral demand traded a Phase 3 failure for a
% Phase 1 one, and it was only caught by chance on a single episode.
%
% These thresholds sit well below measured performance (100% on every phase over 30 episodes
% per phase) so ordinary noise cannot fail them - only a real regression can.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    testCase.TestData.params = get_sim_params();
end

function testPilotLandsEveryCurriculumPhase(testCase)
    floors = [0.70, 0.70, 0.50];   % P1, P2, P3
    n = 10;
    fprintf('\n');
    for ph = 1:3
        [rate, v] = fly_phase(ph, n, 'on');
        fprintf('  P%d: landing %5.1f%%  mean impact %.2f m/s (floor %.0f%%)\n', ...
            ph, 100*rate, v, 100*floors(ph));
        verifyGreaterThanOrEqual(testCase, rate, floors(ph), ...
            sprintf(['Scripted pilot landed only %.0f%% of Phase %d. It is the paper''s ' ...
                     'classical baseline; a regression here invalidates the comparison.'], ...
                     100*rate, ph));
    end
end

function testGuardianDoesNotPreventCompetentLandings(testCase)
    % The Operational Sidecar must be non-intrusive to a pilot already flying safely.
    % If attaching it costs landings, the barrier is too aggressive and the paper's
    % central claim is weakened - so this is a safety-case assertion, not a nicety.
    rate_off = fly_phase(1, 10, 'off');
    rate_on  = fly_phase(1, 10, 'on');
    fprintf('  guardian off %.0f%%   guardian on %.0f%%\n', 100*rate_off, 100*rate_on);
    verifyGreaterThanOrEqual(testCase, rate_on, rate_off - 0.1, ...
        'Attaching the sidecar cost the classical pilot landings.');
end

function [rate, mean_impact] = fly_phase(phase, n_episodes, guardian)
    env = LunarLanderEnv('DenseBaseline', guardian);
    select_phase(env, phase);
    p = env.params;

    rng(11);
    landed = 0; impacts = zeros(1, n_episodes);
    for ep = 1:n_episodes
        reset(env);
        for i = 1:p.max_agent_steps
            u = scripted_pilot(env.State, p);
            % Invert the action map so the pilot's physical command survives the trip
            % through the [-1,1] action space.
            [~, ~, done] = step(env, command_to_action(u, env.State, p));
            if done, break; end
        end
        landed = landed + strcmp(env.Outcome, 'landed');
        impacts(ep) = sqrt(env.State(3)^2 + env.State(4)^2);
    end
    rate = landed / n_episodes;
    mean_impact = mean(impacts);
end
