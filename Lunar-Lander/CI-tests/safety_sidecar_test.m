function tests = test_safety_sidecar
%TEST_SAFETY_SIDECAR - Unit tests for the safety_sidecar_filter wrapper
%
% This test harness constructs function-based tests for safety_sidecar_filter.
    tests = functiontests(localfunctions);
end
function setupOnce(testCase)
% Setup shared parameters used across all tests in this file
    % Dynamically add the entire repository (and all subfolders) to the MATLAB path
    scriptPath = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(scriptPath, '..')));
    
    testCase.TestData.params = struct('dry_mass', 4280, 'gravity', 1.62, ...
        'inertia_dry', 24000, 'inertia_fuel_full', 45000, ...
        'max_main_thrust', 45040, 'max_mass_burn_rate', 15.6, ...
        'max_side_torque', 2000, 'max_main_fuel', 8200, ...
        'max_rcs_fuel', 300, 'max_rcs_burn_rate', 0.5, 'r_lunar', 1737400);
end
function testSuicideBurnOverride(testCase)
    % Scenario: Lander is 1 meter off the ground, falling at 50 m/s.
    % The dead AI commands 0 thrust.
    % Test ensures sidecar vetoes and commands emergency thrust.
    x = [0; 1; 0; -50; 0; 0; 8000; 300];
    u_nominal = [0; 0];
    
    % Invoke safety filter to get commanded controls and veto status
    [u_actual, VetoTriggered, ~, ~] = safety_sidecar_filter(x, u_nominal, testCase.TestData.params);
    
    % Assert the Sidecar correctly panicked and seized control
    verifyTrue(testCase, VetoTriggered, 'Sidecar failed to trigger veto in lethal scenario.');
    
    % Assert the Sidecar commanded absolute maximum thrust to try and save the ship
    verifyEqual(testCase, u_actual(1), testCase.TestData.params.max_main_thrust, 'RelTol', 1e-4, ...
        'Sidecar did not command max thrust during a critical boundary breach.');
end

function testRotationalOverride(testCase)
    % Scenario: Lander is falling fast, close to the ground (30m), but tilted 90 degrees (pi/2).
    % The dead AI commands 0 thrust and 0 torque.
    % Test ensures sidecar vetoes and commands emergency torque to right the ship.
    x = [0; 30; 0; -50; pi/2; 0; 8000; 300];
    u_nominal = [0; 0];
    
    [u_actual, VetoTriggered, ~, ~] = safety_sidecar_filter(x, u_nominal, testCase.TestData.params);
    
    % Assert the Sidecar panicked and seized control
    verifyTrue(testCase, VetoTriggered, 'Sidecar failed to trigger veto in rotational lethal scenario.');
    
    % Assert the Sidecar commanded maximum negative torque to fight the pi/2 tilt
    verifyEqual(testCase, u_actual(2), -testCase.TestData.params.max_side_torque, 'RelTol', 1e-4, ...
        'Sidecar did not command max corrective torque to right the tilted ship.');
end

function testCentrifugalLift(testCase)
    % Scenario: Lander is high up (15000m) and moving horizontally at orbital speed (1700 m/s).
    % Even though it is falling slightly, centrifugal lift should counteract gravity
    % such that a_max > 0 is easily satisfied, and the sidecar does NOT panic.
    x = [0; 15000; 1700; -10; 0; 0; 8000; 300];
    u_nominal = [0; 0];
    
    [~, VetoTriggered, ~, ~] = safety_sidecar_filter(x, u_nominal, testCase.TestData.params);
    
    % Assert the Sidecar did NOT panic because centrifugal lift keeps it safe in orbit
    verifyFalse(testCase, VetoTriggered, 'Sidecar panicked during safe orbital flight due to missing centrifugal lift.');
end

function testTorqueAuthorityRetainedDuringNormalDescent(testCase)
    % THE BUG THAT MADE LANDING IMPOSSIBLE.
    %
    % The rotational barrier previously seized the RCS whenever dy < -0.5 && margin < 50.
    % With a fixed 50 m blending zone, margin ~= y - 1.5 at low speed, so the sidecar held
    % the torque channel continuously below ~51 m altitude. Because it drives theta -> 0
    % and ddx = -T*sin(theta)/m, horizontal velocity FROZE at whatever it was when the
    % barrier engaged. Touchdown requires |dx| <= 0.5 m/s while Phases 2 and 3 begin with
    % 10-20 m/s of drift, so those phases could never be landed - the agent was asked to
    % null drift with the only actuator capable of it taken away.
    %
    % A modest, upright descent must leave the agent's commanded torque untouched.
    x = [0; 40; 5; -2; 0.05; 0; 8000; 300];   % 40 m up, gentle descent, slight tilt
    requested_torque = 1500;
    u_nominal = [20000; requested_torque];

    [u_actual, ~, ~, ~] = safety_sidecar_filter(x, u_nominal, testCase.TestData.params);

    verifyEqual(testCase, u_actual(2), requested_torque, 'RelTol', 1e-6, ...
        ['Sidecar seized torque during an ordinary upright descent. The agent cannot ' ...
         'null lateral drift without it, making a safe touchdown unreachable.']);
end

function testBlendingZoneScalesWithBrakingDistance(testCase)
    % A fixed-width blending zone is wrong at low speed: braking distance on a 2 m/s
    % descent is about 1 m, so a 50 m window meant permanent guardian authority. The zone
    % must scale with the ship's actual stopping distance.
    params = testCase.TestData.params;

    % Slow descent, 40 m up: comfortably safe, so no thrust floor should be imposed.
    x_slow = [0; 40; 0; -2; 0; 0; 8000; 300];
    [u_slow, veto_slow, ~, ~] = safety_sidecar_filter(x_slow, [0; 0], params);
    verifyEqual(testCase, u_slow(1), 0, 'AbsTol', 1e-6, ...
        'Sidecar imposed a thrust floor during a safe slow descent 40 m up.');
    verifyFalse(testCase, veto_slow, 'Sidecar vetoed a manifestly safe state.');

    % Same altitude, fast descent: genuinely unsafe, so it must intervene hard.
    x_fast = [0; 40; 0; -45; 0; 0; 8000; 300];
    [u_fast, veto_fast, ~, ~] = safety_sidecar_filter(x_fast, [0; 0], params);
    verifyEqual(testCase, u_fast(1), params.max_main_thrust, 'RelTol', 1e-4, ...
        'Sidecar failed to command max thrust during a lethal descent at the same altitude.');
    verifyTrue(testCase, veto_fast, 'Sidecar failed to veto a lethal descent.');
end

function testExtremeTiltAlwaysOverridden(testCase)
    % The hard 45 degree structural limit must still apply everywhere in the envelope,
    % including high altitude where the altitude barrier is quiet.
    x = [0; 5000; 0; -5; 1.4; 0; 8000; 300];   % ~80 degrees of tilt
    [u_actual, VetoTriggered, ~, ~] = safety_sidecar_filter(x, [10000; 0], ...
        testCase.TestData.params);

    verifyTrue(testCase, VetoTriggered, 'Sidecar ignored an 80 degree tilt.');
    verifyLessThan(testCase, u_actual(2), 0, ...
        'Sidecar must command negative torque to correct a positive tilt.');
end

function testRCSBingoFuel(testCase)
    % Scenario: Ship is wildly tilted and needs massive corrective torque.
    % However, the RCS fuel tank is completely empty (m_rcs_fuel = 0).
    % The sidecar must mathematically accept that it cannot right the ship and clamp torque to 0.
    x = [0; 1000; 0; -50; pi/2; 0; 8000; 0]; % m_rcs_fuel = 0
    u_nominal = [1000; 2000]; % AI asks for full torque
    
    [u_actual, ~, ~, ~] = safety_sidecar_filter(x, u_nominal, testCase.TestData.params);
    
    % Assert the sidecar forces torque to 0 because the tank is empty
    verifyEqual(testCase, u_actual(2), 0, 'RelTol', 1e-4, ...
        'Sidecar failed to force 0 torque when RCS fuel was depleted.');
end