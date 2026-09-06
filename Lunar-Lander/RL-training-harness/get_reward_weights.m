function weights = get_reward_weights()
    % get_reward_weights: Single source of truth for RL training rewards
    %
    % All values below are RAW. Dense schemes divide the final per-step reward by
    % reward_scale before handing it to the neural network, so the numbers the agent
    % actually sees land in roughly [-9, +5] - asymmetric because failure has to be
    % graded across a much wider range of outcomes than success does.

    weights.crash   = -800;   % ->  -8.0  scaled at worst, graded by how badly it missed
    weights.success =  500;   % -> +5.0  scaled
    weights.oob     = -900;   % ->  -9.0  scaled, and deliberately the worst outcome
    %
    % WHY THE CRASH RANGE IS WIDER THAN THE LANDING BONUS. The penalty has to grade
    % failure across roughly 40 m/s of impact speed while keeping a steep slope in the
    % last 4 m/s, where the landing is actually decided. Those two demands do not both
    % fit in a range of 5.0: giving the near band the slope it needs (about 3.4 over
    % 1x-4x the gate) leaves under 0.35 for everything beyond 4x, which is what produced
    % the measured -0.02 reward for braking Phase 3 from 25 to 19 m/s. Widening the range
    % to 8.0 buys both.
    %
    % Asymmetry against the +5.0 landing bonus is appropriate rather than awkward for a
    % safety argument: crashing a lander is much worse than landing it is good.
    %
    % oob is raised in step so that flying out of the box stays strictly worse than any
    % crash. Otherwise a 25 m/s impact (-6.05) would beat the old -5.0 oob penalty and
    % escaping the flight box would become a way to dodge a bad landing.

    % --- SHAPE OF THE CRASH PENALTY ---
    % Three bands, because no single saturating term covers the whole range. Measured with
    % a controller blended toward free-fall: braking Phase 3 from 25.06 to 19.18 m/s moved
    % the episode reward by -0.02, while the final increment before landing was worth
    % +5.4. The landscape was a cliff, not a slope.
    %
    %   NEAR  - full range over 1x to 4x the touchdown limit. This is where a landing is
    %           won or lost, and it needs the steepest slope. (This band already existed
    %           and worked; the point of the rework is not to lose it.)
    %   FAR   - full range from 4x out to 40x. Previously ABSENT: proximity saturated at
    %           4x, so a 19 m/s and a 25 m/s arrival scored identically and an agent
    %           learning to brake got no feedback until it was nearly perfect.
    %   SEV   - raw impact speed, normalised over 40 m/s rather than 50, keeping the
    %           distinction between a smash and an arrival.
    weights.crash_near_share = 0.40;
    weights.crash_far_share  = 0.45;
    weights.crash_sev_share  = 0.15;
    weights.crash_far_limit  = 40.0;   % multiple of the gate at which FAR saturates
    weights.crash_sev_limit  = 40.0;   % m/s at which SEV saturates

    % --- SIDECAR VETO PENALTY ---
    % Charged ONCE per engagement (on the rising edge), NOT once per timestep.
    %
    % This distinction is the difference between a working reward and a suicidal one.
    % As a per-step charge at 50 Hz, -10 raw meant -5.0 reward per SECOND of guardian
    % activity, while the worst possible crash cost only -7.5. Deliberately crashing
    % became the optimal policy after 1.5 seconds of vetoed flight, and since the
    % guardian was engaged for the whole descent the agent could not avoid the tax by
    % flying well. Per-engagement pricing keeps the intended lesson - "the boundary is
    % expensive, stay away from it" - without making survival cost more than dying.
    weights.sidecar_veto = -10;   % -> -0.1 scaled, per engagement

    % Hard ceiling on the TOTAL veto penalty an episode can accrue.
    %
    % Edge-triggering alone is not enough. If the barrier chatters - engaging, correcting,
    % releasing, re-engaging - the per-event charges still accumulate without bound, and
    % the original inversion creeps back: enough engagements and dying becomes cheaper
    % than being repeatedly rescued. The cap is set well below the magnitude of a
    % hard-impact crash (-5.0 scaled) so that ordering can never invert, whatever the
    % barrier does. Engagements past the cap are still COUNTED as a safety metric; they
    % simply stop being charged.
    weights.sidecar_veto_budget = -200;   % -> -2.0 scaled, per episode

    % Terminating because the ship stalled out near the surface without ever committing
    % to a touchdown. Mildly negative so hovering is not a free way to dodge the landing
    % evaluation, but nowhere near a crash - a guardian-stabilised hover is a SAFE
    % outcome and must not be scored as a failure.
    weights.stall = -50;      % -> -0.5 scaled

    % Running out the episode clock without ever reaching the ground. This is what stops
    % the agent loitering to dodge the touchdown evaluation - the role fuel_main used to
    % play, now stated directly instead of being priced into every gram of propellant.
    % Sized with the reduced fuel cost so a full-length hover totals about -6.7. That has
    % to stay worse than the crashes the agent actually produces, or the 62% timeout
    % failure mode returns: DDPG at 600 episodes was arriving at 4-17 m/s, which scores
    % -3.3 to -4.9, so committing still clearly beats loitering. Above roughly 30 m/s a
    % crash does become worse than hovering - correctly, since at that speed hovering
    % genuinely is the safer outcome.
    weights.timeout = -500;   % -> -5.0 scaled

    % --- SHAPING ---
    % Gain on the DESCENT-guidance potential (see shaping_potential). Total payout across
    % a full 2500 m descent is about +3.3 scaled, against a terminal outcome of +/-5.0, so
    % descent guidance cannot outbid the landing itself.
    weights.shaping = 400;

    % Gain on the FINAL-APPROACH potential, which scores the touchdown criteria against
    % their own limits below params.approach_gate_alt.
    %
    % Sized so the difference between a sloppy arrival and a clean one is comparable to
    % the landing bonus itself. Worked example at the pad: a 2 m/s lateral, 2 m/s vertical,
    % 0.05 rad arrival scores a normalised miss of 6.5 -> -3.25 scaled, while a
    % 0.1 / 0.5 / 0.02 arrival scores 0.9 -> -0.45. The 2.8 of gradient between them is
    % the signal the harness previously did not have: under the descent term alone that
    % same improvement was worth 0.06.
    %
    % Capped by params.approach_max_miss at -5.0 scaled so it can never exceed the
    % magnitude of the terminal outcomes it is meant to lead toward.
    weights.approach = 50;

    % Fuel cost per unit of normalised thrust per second of burn.
    %
    % LOWERED from -3.2, and the anti-loitering job moved to weights.timeout below.
    %
    % At -3.2 this term was doing two incompatible jobs. It was sized to make a 300 s
    % hover cost -4.4 so loitering could not beat committing to a descent - but it taxes
    % THRUST, and thrust is exactly what braking requires. Measured consequence: the
    % terminal reward for braking Phase 3 from 25 to 19 m/s was +0.24, and the fuel burnt
    % to achieve it cost about the same, so the net episode signal was -0.02. The reward
    % was charging the agent for the one behaviour the task needs.
    %
    % Loitering is now punished directly, by an explicit timeout penalty, which is both
    % more honest and leaves the fuel term free to be an efficiency term.
    weights.fuel_main = -1.2;   % per second at full main throttle
    weights.fuel_rcs  = -0.32;  % per second at full RCS deflection

    % Divisor applied to the final reward. Deep networks predict poorly when asked to
    % regress values in the tens of thousands.
    weights.reward_scale = 100;
end
