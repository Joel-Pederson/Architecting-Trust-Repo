function tests = action_map_test
%ACTION_MAP_TEST - The agent's control interface must have exactly one definition.
%
% This existed in three places at once: LunarLanderEnv.step (gravity-compensated),
% main_simulation's RL_AGENT branch (the RETIRED raw-throttle map), and two copies of the
% inverse in the fault study and the pilot regression test. The consequence was silent and
% serious: an agent evaluated through main_simulation was flying a different plant than it
% trained on - at full tanks, action 0 commanded 22.5 kN there against the environment's
% 20.3 kN hover, and the mass-invariance the map exists to provide was absent entirely.
%
% Nothing in the test suite could catch that, because each copy was self-consistent.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    testCase.TestData.params = get_sim_params();
end

function testRoundTripIsExact(testCase)
    % command_to_action must invert action_to_command wherever the command is achievable.
    p = testCase.TestData.params;
    states = {full_tanks(p), near_empty(p)};
    labels = {'full tanks', 'near empty'};

    % Thrust actions kept inside [-1, 0.9]: beyond 2x hover the forward map clamps at
    % max_main_thrust and the round trip is legitimately lossy.
    actions = [-1.0 0.0; -0.5 -0.75; 0.0 0.0; 0.25 0.5; 0.9 1.0];

    for s = 1:numel(states)
        for k = 1:size(actions, 1)
            a = actions(k, :)';
            u = action_to_command(a, states{s}, p);
            a_back = command_to_action(u, states{s}, p);
            verifyEqual(testCase, a_back, a, 'AbsTol', 1e-12, ...
                sprintf('Round trip failed at %s for action [%g %g].', ...
                        labels{s}, a(1), a(2)));
        end
    end
end

function testActionZeroIsHoverAtAnyFuelState(testCase)
    % The whole point of gravity compensation: the plant gain must not drift as 8200 kg
    % of propellant burns off. Under the old raw-throttle map the same action produced
    % 1.76 m/s^2 per unit at launch mass and 5.26 near-empty.
    p = testCase.TestData.params;
    for x = {full_tanks(p), near_empty(p)}
        s = x{1};
        u = action_to_command([0; 0], s, p);
        m_total = p.dry_mass + s(7) + s(8);
        net_accel = u(1) / m_total - p.gravity;
        verifyEqual(testCase, net_accel, 0, 'AbsTol', 1e-9, ...
            'Action 0 does not command a hover at this fuel state.');
    end
end

function testEnvironmentAndStandaloneSimAgree(testCase)
    % The regression that motivated this file. Both entry points must resolve an
    % identical action and state to an identical physical command.
    p = testCase.TestData.params;
    env = LunarLanderEnv('DenseBaseline', 'off');
    reset(env);

    action = [0.3; -0.4];
    x = env.State;

    % What main_simulation now computes (it calls action_to_command directly).
    u_standalone = action_to_command(action, x, p);

    % What the environment applies. Guardian is off, so u_actual is the clamped nominal
    % command and LoggedSignals.Control reports it directly.
    [~, ~, ~, logs] = step(env, action);

    verifyEqual(testCase, logs.Control, u_standalone, 'AbsTol', 1e-9, ...
        ['LunarLanderEnv and main_simulation disagree on the action map. An agent ' ...
         'evaluated in one would be flying a different vehicle than it trained in.']);
end

function testCommandToActionClipsToTheActionSpace(testCase)
    % Callers must record the CLIPPED action, because that is what the environment
    % receives. For behaviour cloning, recording the raw quotient would train the network
    % toward commands the plant cannot execute.
    p = testCase.TestData.params;
    s = full_tanks(p);

    huge = [p.max_main_thrust * 10; p.max_side_torque * 10];
    a = command_to_action(huge, s, p);
    verifyLessThanOrEqual(testCase, a, [1; 1]);
    verifyGreaterThanOrEqual(testCase, a, [-1; -1]);

    % Zero thrust is exactly the lower bound, not below it.
    a_zero = command_to_action([0; 0], s, p);
    verifyEqual(testCase, a_zero(1), -1, 'AbsTol', 1e-12);
end

function x = full_tanks(p)
    x = [0; 500; 0; -5; 0; 0; p.max_main_fuel; p.max_rcs_fuel];
end

function x = near_empty(~)
    x = [0; 500; 0; -5; 0; 0; 10; 5];
end
