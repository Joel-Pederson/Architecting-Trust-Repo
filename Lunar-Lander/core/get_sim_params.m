function params = get_sim_params()
    % Lander Parameters: Single source of truth for the physical universe
    % Apollo 11 Specifications - State based off the historical Apollo Powered Descent Initiation (PDI)
    params.dry_mass = 4280;           % kg (Actual empty weight of LEM)
    params.gravity = 1.62;            % m/s^2 (Lunar gravity)
    params.r_lunar = 1737400;         % meters (Radius of the moon)

    % Dynamic Inertia Parameters
    params.inertia_dry = 24000;       % kg*m^2 (Inertia of the empty lander)
    params.inertia_fuel_full = 45000; % kg*m^2 (Inertia contribution of full fuel tanks)

    % Engine & Fuel Parameters
    params.max_main_thrust = 45040;   % Newtons (Actual thrust of the LEM DPS)
    params.max_mass_burn_rate = 15.6; % kg/s (Approximate DPS max flow rate)
    params.max_main_fuel = 8200;      % kg (Total main engine propellant)

    params.max_side_torque = 2000;    % Newton-meters (RCS Maximum Torque)
    params.max_rcs_burn_rate = 0.5;   % kg/s (RCS flow rate at max torque)
    params.max_rcs_fuel = 300;        % kg (Total RCS propellant)

    % --- HARNESS VERSION ---
    % Bumped whenever the semantics of the environment, reward, or control rate change.
    % Tuned hyperparameters saved by tune_hyperparameters are stamped with this value and
    % REFUSED on mismatch. Without that check, a hyperparameter set optimised against an
    % older harness silently overrides the current configuration - which is exactly what
    % happened with an Aug-2025 file whose Gamma had been tuned for a 0.02 s sample time.
    params.harness_version = 8;

    % Simulation Parameters
    params.dt = 0.02;                 % Physics integration step (50Hz) seconds per frame

    % --- CONTROL RATE DECIMATION ---
    % The physics integrator runs at 50Hz for numerical accuracy, but the agent does
    % NOT need to make a decision every 20ms. Holding each action for 5 physics
    % substeps gives the agent a 10Hz decision rate. This matters enormously for RL:
    % a 0.02s agent step makes a 300 second descent 15,000 decisions long, which no
    % practical discount factor can credit-assign across. At 10Hz the same descent is
    % 3,000 decisions, and gamma=0.995 covers a 20 second lookahead instead of 2.
    params.control_decimation = 5;                              % physics substeps per agent decision
    params.agent_dt = params.dt * params.control_decimation;    % 0.1 s -> 10 Hz agent sample time

    % Two distinct step budgets, deliberately named apart. A single `max_steps` field is
    % ambiguous once the agent and the physics run at different rates, and reusing it
    % silently truncated the standalone physics-rate simulation from 900 s to 60 s.
    params.max_agent_steps = 3000;    % RL episode cap: agent decisions (300 s at 10 Hz)

    % PER-PHASE episode budget. A 50 m touchdown does not need the same clock as a
    % powered descent from orbit, and giving every phase the longest budget would make a
    % full-length hover cost nearly as much as flying out of bounds - collapsing the
    % margin that reward_ordering_test exists to protect. Phases 1-3 keep exactly the
    % budget they were measured under, so no existing result changes.
    % Phase 4 raised from 9000 after softening the demonstrator's attitude gains. The RCS
    % delivers 0.033 rad/s^2 against the vehicle's inertia, so a 90 degree slew is
    % torque-limited no matter how it is commanded - a controller smooth enough to be
    % fitted by regression is necessarily slower than one that slams between the stops,
    % and the budget has to cover it. Measured: 8163 steps with the stiff gains, timing
    % out at 9000 with the smooth ones while still doing 270 m/s.
    params.phase_max_steps = [3000, 3000, 3000, 13000];
    params.max_sim_steps   = 45000;   % Standalone main_simulation cap: physics steps (900 s at 50 Hz)

    % --- TOUCHDOWN CRITERIA (Single source of truth) ---
    % Derived from the Apollo LM landing gear design limits. These are shared by the
    % reward schemes, the sidecar, and the CI tests so the definition of a "safe
    % landing" cannot drift between the training loop and the safety case.
    params.touchdown_alt      = 2.0;  % m   - altitude at which ground contact is evaluated
    params.max_touchdown_dy   = 1.0;  % m/s - vertical impact limit (~3 ft/s)
    params.max_touchdown_dx   = 0.5;  % m/s - lateral impact limit (~2 ft/s)
    params.max_touchdown_tilt = 0.1;  % rad - tilt limit at contact (~6 degrees)
    params.max_touchdown_spin = 0.1;  % rad/s - residual spin at contact. NOT part of the
                                      % pass/fail gate (which stays dy/dx/tilt, as before);
                                      % used only to grade how badly a crash missed.

    % --- DISCOUNT FACTOR (Single source of truth) ---
    % Lives here rather than in the agent builders because the environment needs it too:
    % potential-based shaping is only policy-invariant in the form gamma*Phi(s') - Phi(s),
    % so the env must discount with exactly the gamma the agent learns with. Two copies
    % of this number drifting apart would silently break the invariance guarantee.
    % 0.9995 at agent_dt = 0.1 s gives a 2000-step / 200 s horizon.
    %
    % MEASURED, not chosen by rule of thumb. At 0.995 (a 200-step / 20 s horizon) the
    % discounted return of a landing policy on Phase 3 was -0.301 against -0.349 for a
    % policy that simply climbs away: a margin of 0.05 on the phase that makes up a third
    % of the evaluation set. The ordering was correct - landing did win at every gamma
    % tested, and climbing away was always worst - but a 0.05 margin is indistinguishable
    % from noise to a critic, and DDPG duly collapsed to a state-independent max-climb
    % action (thrust spread of 0.0034 across the entire state space).
    %
    % The problem is horizon, not ordering: Phase 3 episodes run 1300-2000 agent steps and
    % all the reward is at the end, so a 200-step horizon cannot see the landing. At 0.999
    % the same comparison is +1.907 against -2.991 (margin 4.90), and at 0.9995 it is
    % +3.673 against -4.037 (margin 7.71).
    %
    % 0.9995 rather than 0.999 because a MEASURED Phase 3 landing takes 1946 agent steps.
    % A 1000-step horizon leaves 0.999^1946 = 14% of that terminal reward visible from the
    % start of the episode; a 2000-step horizon leaves 38%, which a critic can regress on.
    params.gamma_agent = 0.9995;

    % --- SHAPING POTENTIAL BOUNDS ---
    % Saturation caps on the descent-guidance potential, in normalised units
    % (x/1000, y/3000, v/100).
    %
    % These were 5.0 and 3.0, which let Phi reach -32 against terminal outcomes of +/-5.
    % Measured consequence: flying out of bounds cost -37 rather than the -5 the weights
    % advertise, so the single strongest lesson available in the landscape was "never
    % climb" - a lesson satisfied by not thrusting, which crashes. The caps below sit
    % just outside the largest legitimate start (Phase 3 begins at dist 0.83, speed 0.32),
    % so normal flight is never clipped and only genuine runaways are bounded.
    params.shaping_max_dist  = 1.2;
    params.shaping_max_speed = 0.5;
    params.shaping_max_tilt  = 0.5;   % rad; past this the vehicle is tumbling and the
                                      % terminal outcome, not the shaping, should dominate

    % --- FINAL-APPROACH POTENTIAL ---
    % Altitude below which the touchdown criteria start contributing to the potential,
    % faded in linearly so it never appears as a step.
    %
    % WHY THIS EXISTS. The descent potential normalises velocity by 100 m/s, so the entire
    % span from a FAILING 2 m/s lateral drift to a PASSING 0.5 m/s was worth 0.06 reward
    % against a +5.00 landing bonus. The agent received almost no signal about the
    % criterion that actually decides every episode. Measured directly: an ideal vertical
    % descent law lands 100% of Phase 1 with zero lateral offset, and 18.3% once the real
    % init_dx ~ N(0, 2) randomisation is switched on - matching P(|dx| < 0.5) for that
    % distribution almost exactly. Lateral velocity is the binding gate term.
    params.approach_gate_alt = 100.0;  % m
    params.approach_max_miss = 10.0;   % cap on the summed normalised miss, so the
                                       % approach term can never exceed the terminal scale

    % --- POWERED DESCENT INITIATION (PHASE 4) ---
    % Apollo 11 PDI: 15.2 km altitude, ~1697 m/s horizontal, near-zero vertical rate at
    % perilune. This is where autonomous descent actually begins; everything above it is
    % orbital coast, which costs runtime without adding evidence about a runtime barrier.
    %
    % The vehicle can do it: 2963 m/s of delta-v available (Isp 294 s) against ~1900 m/s
    % required, with a thrust-to-weight of 2.18 at PDI mass.
    params.pdi_altitude   = 15200;    % m
    params.pdi_downrange  = 550000;   % m - starts this far short of the pad, closing
                                      %
                                      % SIZED FROM THE VEHICLE, not chosen. Braking
                                      % authority falls through the burn as gravity
                                      % reasserts - 3.52 m/s^2 while weightless at PDI,
                                      % 3.14 near the surface - so the average is ~3.3 and
                                      % the ideal stopping distance is 1697^2/(2*3.3) =
                                      % 436 km. Flying the profile at 85% of available
                                      % thrust, plus slew and cosine losses, puts the real
                                      % requirement near 500 km.
                                      %
                                      % 410 km was exactly marginal and 460 km still
                                      % overshot by 200 km. Apollo's own PDI was ~480 km
                                      % downrange at the same velocity, which is the
                                      % check that this is the vehicle's number rather
                                      % than a fudge.
    params.pdi_velocity   = 1697;     % m/s horizontal
    params.pdi_descent    = -3;       % m/s vertical at perilune
    params.pdi_pitch      = 1.50;     % rad - ALREADY pitched retrograde at PDI, as the
                                      % LM was. Starting upright costs ~20 s of slew at
                                      % 1697 m/s, i.e. ~40 km of downrange, on a profile
                                      % with no margin to give.

    % --- CURRICULUM WEIGHTING ---
    % Probability of drawing each difficulty phase [P1 50m, P2 500m, P3 2500m].
    %
    % A uniform 33/33/33 split gave zero successful landings in 360 random-policy probe
    % episodes, so the +5 touchdown reward never entered the replay buffer at all and the
    % agent could only ever learn from shaping. The probe also showed WHERE success is
    % nearly reachable: with the guardian active, random policies in P1 got within
    % 1.15 m/s of the 1.0 m/s touchdown limit, versus 2.73 m/s in P3.
    %
    % These are FIXED weights, not an adaptive schedule. That matters: parallel workers
    % cannot share an episode counter, which is why the curriculum is randomised per
    % episode in the first place. A fixed reweighting needs no counter and stays
    % correct under async parallel training.
    params.curriculum_weights = [0.60 0.25 0.15];

    % Evaluation deliberately uses the UNIFORM mix, so a training diet weighted toward
    % the easy scenario cannot inflate the reported landing rate. Train on one
    % distribution, report on another, and break results out per phase.
    params.eval_curriculum_weights = [1/3 1/3 1/3];

    % --- FLIGHT BOX (Out of bounds) ---
    % A 500 km lateral boundary was effectively no boundary at all. Scenarios start
    % within +/-100 m carrying up to 20 m/s of drift, so an agent that simply flies away
    % could coast for the full 3000-step episode without ever tripping OOB - accruing an
    % enormous shaping penalty (a TD3 smoke run scored -241) instead of terminating. That
    % forces the critic to represent values spanning [-250, +10], which is precisely the
    % exploding-value problem the reward scaling exists to prevent. 20 km is still an
    % order of magnitude beyond any legitimate descent profile.
    % --- FLIGHT BOX ---
    % Sized for a full Apollo-style powered descent, not just a terminal approach.
    % Braking 1700 m/s of horizontal velocity at ~3.5 m/s^2 takes about 480 s and covers
    % roughly 550 km of downrange, so a 20 km lateral boundary would have triggered
    % out-of-bounds within seconds of PDI. Enlarging it is harmless to the existing
    % curriculum phases, which never travel more than a few hundred metres laterally.
    params.max_abs_x  = 600000;       % m - lateral boundary (downrange arc length)
    params.max_alt    = 30000;        % m - ceiling (PDI starts at 15.2 km)

end
