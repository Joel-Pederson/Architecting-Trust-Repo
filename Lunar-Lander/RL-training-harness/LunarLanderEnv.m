classdef LunarLanderEnv < rl.env.MATLABEnvironment
    % LUNARLANDERENV: Reinforcement Learning Environment Wrapper.
    % This class acts as the bridge between the physical simulation and the AI agent.
    %
    % GUARDIAN MODES (for the A/B study):
    %   'on'  - The Developmental Guardian filters every command during training. This is
    %           the configuration the Architecting Trust framework proposes: safety is a
    %           property of the training loop, not just of deployment.
    %   'off' - The agent flies unfiltered. This is the control arm. Train it, then wrap
    %           the sidecar around it at evaluation time to measure how many crashes the
    %           Operational Sidecar prevents on a policy that never learned about it.
    %
    % The two arms share everything else - physics, rewards, network, hyperparameters -
    % so any difference in landing rate is attributable to the guardian alone.

    properties
        % The current state and simulation parameters
        State           (8,1) double
        params          (1,1) struct
        weights         (1,1) struct

        % State history tracking for rewards and visualization
        State_prev      (8,1) double
        u_prev          (2,1) double

        RewardScheme    char
        GuardianMode    char

        % --- Episode telemetry (read after sim() for the A/B study) ---
        VetoCount       (1,1) double = 0   % number of distinct sidecar engagements
        VetoSteps       (1,1) double = 0   % physics substeps spent under guardian authority
        VetoPenaltyPaid (1,1) double = 0   % raw veto penalty charged so far (capped by budget)
        StepCount       (1,1) double = 0   % agent decisions taken this episode
        Outcome         char               % 'flying' | 'landed' | 'crashed' | 'oob' | 'stalled'
        Phase           (1,1) double = 0   % curriculum phase drawn this episode (1, 2 or 3)

        % Probability of drawing each phase. Defaults to the training weights;
        % evaluate_policy overrides this with the uniform evaluation mix so that a
        % reweighted training diet cannot inflate the reported landing rate.
        % Variable length ON PURPOSE. A fixed (1,3) would reject every existing caller the
        % moment a fourth phase was added - the fault study, the CI tests and the demos
        % all pass [1 0 0] style selectors. Short vectors are zero-padded in reset, so
        % [1 0 0] simply means "phase 1, never the powered descent".
        CurriculumWeights (1,:) double = [1/3 1/3 1/3]

        % Whether the environment adds potential-based shaping on top of the reward
        % scheme's own terms. Set from the scheme in the constructor: 'SparseOnly' is a
        % deliberate sparse-reward control condition, and silently shaping it would have
        % turned the two schemes into the same experiment.
        ShapingEnabled  (1,1) logical = true
    end

    properties (Access = protected)
        IsDone          = false;
        GuardianEnabled = true;
        VetoActive_prev = false;  % for rising-edge detection
        StallSteps      = 0;      % consecutive substeps loitering near the surface
    end

    methods
        function this = LunarLanderEnv(rewardScheme, guardianMode)
            % CONSTRUCTOR: Defines the rules of the universe for the AI

            if nargin < 1 || isempty(rewardScheme)
                rewardScheme = 'DenseBaseline';
            end
            if nargin < 2 || isempty(guardianMode)
                guardianMode = 'on';
            end

            % 1. Define Observation Space (10 Variables)
            % [x; y; dx; dy; theta; dtheta; m_main_fuel; m_rcs_fuel; h_alt; h_fuel]
            obsInfo = rlNumericSpec([10 1]);
            obsInfo.Name = 'LunarLanderStates';

            % 2. Define Action Space (2 Variables)
            % Force the agent to output values between [-1, 1]. Scale these
            % to the actual physics hardware limits inside step() to avoid large numbers.
            actInfo = rlNumericSpec([2 1], 'LowerLimit', [-1; -1], 'UpperLimit', [1; 1]);
            actInfo.Name = 'LanderThrustAndTorque';

            % 3. Initialize the superclass (MATLAB RL Environment)
            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);

            % Now that the object is fully constructed, assign properties
            this.RewardScheme = rewardScheme;
            this.GuardianMode = lower(guardianMode);
            this.ShapingEnabled = strcmp(rewardScheme, 'DenseBaseline');

            switch this.GuardianMode
                case 'on'
                    this.GuardianEnabled = true;
                case 'off'
                    this.GuardianEnabled = false;
                otherwise
                    error('LunarLanderEnv:UnknownGuardianMode', ...
                        'Unknown guardian mode: %s. Use ''on'' or ''off''.', guardianMode);
            end

            % 4. Load our modular configurations (Single Source of Truth)
            this.params = get_sim_params();
            this.weights = get_reward_weights();
            this.CurriculumWeights = this.params.curriculum_weights;
        end

        function [Observation, LoggedSignals] = reset(this)
            % RESET: Called automatically at the start of every new training episode

            % Domain Randomization (Uniform Curriculum Learning)
            % Parallel workers cannot sync a shared episode counter, so instead of a
            % sequential curriculum every episode independently samples a difficulty
            % phase. This fills the replay buffer with a diverse mix of experiences.

            % Weighted draw over the three phases. Uses cumulative probabilities rather
            % than fixed 0.33/0.66 thresholds so the mix is configurable per environment
            % (training uses a P1-weighted diet, evaluation uses the uniform mix).
            % Zero-pad to the full phase count so a 3-element selector like [1 0 0]
            % still means "phase 1", rather than silently indexing past the end.
            n_phases = numel(this.params.phase_max_steps);
            cw = zeros(1, n_phases);
            w = this.CurriculumWeights;
            cw(1:min(numel(w), n_phases)) = w(1:min(numel(w), n_phases));
            cw = cw / sum(cw);
            phase_selector = rand();

            % Drift direction is randomised. Previously every episode began drifting to
            % the right, so the agent only ever learned to correct one sign of error.
            drift_sign = sign(randn());
            if drift_sign == 0, drift_sign = 1; end

            if phase_selector < cw(1)
                % Phase 1: Hover and Touchdown (Easy)
                % Agent starts 50 meters off the ground with near-zero velocity.
                init_x = randn() * 10;
                init_y = 50 + (randn() * 5);
                init_dx = randn() * 2;
                init_dy = -2 + (randn() * 1);
                init_theta = randn() * 0.05;
                init_dtheta = randn() * 0.01;
                this.Phase = 1;

            elseif phase_selector < cw(1) + cw(2)
                % Phase 2: Glide Slope Approach (Medium)
                % Agent starts 500 meters up falling at 10 m/s with 10 m/s drift.
                init_x = randn() * 50;
                init_y = 500 + (randn() * 20);
                init_dx = drift_sign * (10 + (randn() * 2));
                init_dy = -10 + (randn() * 2);
                init_theta = randn() * 0.05;
                init_dtheta = randn() * 0.02;
                this.Phase = 2;

            elseif phase_selector < cw(1) + cw(2) + cw(3)
                % Phase 3: High Altitude Terminal Descent (Hard)
                % Agent starts at 2,500m falling at 25 m/s with 20 m/s drift.
                init_x = randn() * 100;
                init_y = 2500 + (randn() * 50);
                init_dx = drift_sign * (20 + (randn() * 5));
                init_dy = -25 + (randn() * 2);
                init_theta = randn() * 0.1;
                init_dtheta = randn() * 0.05;
                this.Phase = 3;

            else
                % Phase 4: Powered Descent Initiation (Apollo-style)
                %
                % 15.2 km altitude, ~1697 m/s horizontal, near-zero vertical rate at
                % perilune - the point where autonomous descent actually begins. Starts
                % 410 km SHORT of the pad and closes on it, so the drift sign is fixed:
                % unlike phases 1-3 this is not a drift to be nulled, it is orbital
                % velocity to be spent.
                %
                % Note the scale change. Downrange is 4000x that of Phase 3 and horizontal
                % velocity is 85x, which is why the observation normalisers cannot serve
                % both regimes on one linear scale.
                init_x = -this.params.pdi_downrange + (randn() * 2000);
                init_y = this.params.pdi_altitude + (randn() * 200);
                init_dx = this.params.pdi_velocity + (randn() * 20);
                init_dy = this.params.pdi_descent + (randn() * 1);
                init_theta = this.params.pdi_pitch + randn() * 0.05;
                init_dtheta = randn() * 0.01;
                this.Phase = 4;
            end

            init_main_fuel = this.params.max_main_fuel;
            init_rcs_fuel  = this.params.max_rcs_fuel;

            % Set the internal state
            this.State = [init_x; init_y; init_dx; init_dy; init_theta; init_dtheta; init_main_fuel; init_rcs_fuel];

            % Reset historical tracking
            this.u_prev = [0; 0];
            this.State_prev = this.State;
            this.VetoActive_prev = false;
            this.VetoPenaltyPaid = 0;
            this.VetoCount = 0;
            this.VetoSteps = 0;
            this.StallSteps = 0;
            this.StepCount = 0;
            this.Outcome = 'flying';
            this.IsDone = false;

            % Return initial normalized observation to the AI
            Observation = this.normalize_state(this.State);
            LoggedSignals = this.pack_logs([0; 0], false);
        end

        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            % STEP: One agent decision, held across control_decimation physics substeps.
            %
            % The physics integrator needs 50 Hz for accuracy; the agent does not need to
            % re-decide that often. Decoupling the two shortens episodes by 5x, which is
            % what makes the discount factor able to reach the landing at all.

            % 1. SCALE ACTIONS (Neural Net [-1, 1] -> Physics Domain)
            % Gravity-compensated; see action_to_command for the map and why it is not a
            % raw throttle. It lives in core/ because main_simulation needs the identical
            % map - it previously kept its own copy of the retired raw-throttle version,
            % so an agent flown there was driving a different plant than it trained on.
            u_nominal = action_to_command(Action, this.State, this.params);

            T_max   = this.params.max_main_thrust;
            Tau_max = this.params.max_side_torque;
            dt      = this.params.dt;

            Reward = 0;
            IsDone = false;
            VetoActive = false;
            u_actual = u_nominal;

            % --- SHAPING, PART 1 OF 2: capture the potential at the START of the step ---
            % Potential-based shaping is applied ONCE PER AGENT STEP, not per physics
            % substep, and in the discounted form gamma*Phi(s') - Phi(s). Both details
            % matter. Ng et al. (1999) only guarantee policy invariance for the
            % discounted form, and a per-substep discount of gamma^(1/decimation) does not
            % telescope across the five substeps because the intermediate terms stop
            % cancelling. Applying it here, at the agent's own decision rate, is the only
            % place both properties hold.
            if this.ShapingEnabled
                phi_start = shaping_potential(this.State, this.params, this.weights);
            else
                phi_start = 0;
            end

            for k = 1:this.params.control_decimation

                % --- DEVELOPMENTAL GUARDIAN ---
                if this.GuardianEnabled
                    [u_actual, VetoActive, ~, ~] = safety_sidecar_filter(this.State, u_nominal, this.params);
                else
                    % Control arm: hardware clamps only, no safety authority.
                    u_actual = [max(0, min(u_nominal(1), T_max)); ...
                                max(-Tau_max, min(u_nominal(2), Tau_max))];
                    VetoActive = false;
                end

                % Rising-edge detection. The reward is charged once per ENGAGEMENT.
                % Passing the level-triggered flag here is what previously made the
                % guardian cost -5.0 reward per second and turned crashing into the
                % optimal policy.
                VetoEngaged = VetoActive && ~this.VetoActive_prev;
                if VetoEngaged
                    % Always count the engagement - it is a safety metric the A/B study
                    % reports on - but stop CHARGING for it once the episode budget is
                    % exhausted. A chattering barrier must never be able to out-cost a
                    % crash, or the agent relearns that dying is the cheaper option.
                    this.VetoCount = this.VetoCount + 1;

                    budget = abs(this.weights.sidecar_veto_budget);
                    if this.VetoPenaltyPaid >= budget
                        VetoEngaged = false;   % counted, no longer charged
                    else
                        this.VetoPenaltyPaid = this.VetoPenaltyPaid + abs(this.weights.sidecar_veto);
                    end
                end
                if VetoActive
                    this.VetoSteps = this.VetoSteps + 1;
                end
                this.VetoActive_prev = VetoActive;

                % --- 2. THE PHYSICS ENGINE (Euler Integration) ---
                this.State_prev = this.State;
                dxdt = lunar_lander_dynamics(this.State, u_actual, this.params);
                this.State = this.State + dxdt * dt;

                % Wrap angle theta to [-pi, pi] so it never accumulates indefinitely
                this.State(5) = atan2(sin(this.State(5)), cos(this.State(5)));

                % Clamp tanks at empty. Euler stepping can otherwise drive fuel mass
                % slightly negative, which quietly corrupts every mass and inertia term
                % downstream of it.
                this.State(7) = max(0, this.State(7));
                this.State(8) = max(0, this.State(8));

                % --- 3. THE REWARD CALCULATOR ---
                switch this.RewardScheme
                    case 'DenseBaseline'
                        [r, done, outcome] = reward_dense_baseline(this.State, this.State_prev, ...
                            u_actual, this.u_prev, VetoEngaged, this.params, this.weights);
                    case 'SparseOnly'
                        [r, done, outcome] = reward_sparse_only(this.State, this.State_prev, ...
                            u_actual, this.u_prev, VetoEngaged, this.params, this.weights);
                    otherwise
                        error('LunarLanderEnv:UnknownRewardScheme', ...
                            'Unknown reward scheme selected: %s', this.RewardScheme);
                end

                Reward = Reward + r;
                this.u_prev = u_actual;

                % --- 4. STALL DETECTION ---
                % Guards against the agent (or the guardian's hover mode) loitering just
                % above the pad forever to dodge the touchdown evaluation. This is NOT
                % scored as a crash: a guardian-stabilised hover is a safe outcome, and
                % scoring it as a crash was teaching the agent to fear being rescued.
                if this.State(2) < 10.0 && abs(this.State(4)) < 0.2
                    this.StallSteps = this.StallSteps + 1;
                else
                    this.StallSteps = 0;
                end

                if ~done && this.StallSteps >= (5.0 / dt)   % 5 seconds of loitering
                    done = true;
                    outcome = 'stalled';
                    Reward = Reward + (this.weights.stall / this.weights.reward_scale);
                end

                if done
                    IsDone = true;
                    this.Outcome = outcome;
                    break;
                end
            end

            % --- SHAPING, PART 2 OF 2: pay the discounted potential difference ---
            % Phi at an ABSORBING state must be 0 for the invariance theorem to hold. If
            % a terminated state kept its potential, the shaping would double-count the
            % terminal outcome: a fast, tilted arrival would be charged once by Phi and
            % again by the crash penalty, and the two would have to be tuned against each
            % other forever. Zeroing it keeps the terminal reward the single authority on
            % what a landing is worth.
            if IsDone || ~this.ShapingEnabled
                phi_end = 0;
            else
                phi_end = shaping_potential(this.State, this.params, this.weights);
            end
            % UNDISCOUNTED difference, deliberately, and this is a real trade-off.
            %
            % Ng et al.'s exact invariance needs gamma*Phi(s') - Phi(s). Summed
            % undiscounted, that form equals (gamma-1)*sum(Phi_t) - Phi_0, and since Phi
            % is negative everywhere the first term is a POSITIVE drip of (1-gamma)*|Phi|
            % on every single step. Measured on this harness: a 3000-step timeout scored
            % +58.15, better than any landing. The agent's own discounted objective
            % cancels that drip exactly - but MATLAB's training curve, evaluate_policy and
            % every metric in the study sum rewards UNDISCOUNTED, so all of them would
            % have reported loitering as success. Two earlier runs on this project were
            % already misread off a reward curve; a third such trap is not worth the
            % theoretical purity.
            %
            % The undiscounted form telescopes exactly to Phi(terminal) - Phi(s_0) =
            % -Phi(s_0): a bounded, path-independent per-episode constant that leaves the
            % metrics honest. Its cost is a bias in the agent's discounted objective of
            % order (1-gamma)*|Phi|/(1-gamma) = |Phi| ~ 2 reward, bounded and NOT growing
            % with episode length, because gamma^t decays. Bounded bias in the objective
            % beats unbounded corruption of every metric used to judge the result.
            if this.ShapingEnabled
                Reward = Reward + (phi_end - phi_start) / this.weights.reward_scale;
            end

            % --- 5. EPISODE CLOCK ---
            % The environment owns its own step cap and charges for running it out.
            % Loitering used to be punished indirectly, by pricing fuel high enough that a
            % 300 s hover cost more than a crash - but that taxes THRUST, and thrust is
            % what braking requires, so it was charging the agent for the one behaviour
            % the task needs. Penalising the timeout directly leaves the fuel term free to
            % be an efficiency term.
            this.StepCount = this.StepCount + 1;
            if ~IsDone && this.StepCount >= this.episode_step_cap()
                IsDone = true;
                this.Outcome = 'timeout';
                Reward = Reward + (this.weights.timeout / this.weights.reward_scale);
            end

            % --- 6. UPDATE ENVIRONMENT ---
            this.IsDone = IsDone;
            Observation = this.normalize_state(this.State);
            LoggedSignals = this.pack_logs(u_actual, VetoActive);

            % Notify the MATLAB environment that a step has occurred
            notifyEnvUpdated(this);
        end
    end

    methods (Access = private)
        function n = episode_step_cap(this)
        % Episode budget for the phase currently being flown.
        %
        % A 50 m touchdown does not need the clock a 410 km powered descent does, and
        % giving every phase the longest budget would make a full-length hover cost
        % nearly as much as flying out of bounds - collapsing the ordering margin that
        % reward_ordering_test exists to protect. Phases 1-3 keep exactly the budget they
        % were measured under.
            caps = this.params.phase_max_steps;
            if this.Phase >= 1 && this.Phase <= numel(caps)
                n = caps(this.Phase);
            else
                n = this.params.max_agent_steps;
            end
        end

        function norm_state = normalize_state(this, raw_state)
            % NORMALIZE_STATE: Uses the central get_ai_observation function so training
            % and deployment normalization are identical. params is passed explicitly -
            % without it get_ai_observation rebuilds the whole params struct on every
            % single timestep, which is millions of redundant struct constructions per run.
            norm_state = get_ai_observation(raw_state, this.params);
        end

        function logs = pack_logs(this, u_actual, VetoActive)
            % Emit the TRUE physical state alongside the observation. Downstream plotting
            % and analysis should read this rather than trying to invert the observation
            % normalizers, which is how the visualizer ended up rendering altitudes that
            % were 6.7x too large and lateral positions 500x too large.
            logs = struct( ...
                'State',      this.State, ...
                'Control',    u_actual, ...
                'VetoActive', VetoActive, ...
                'VetoCount',  this.VetoCount, ...
                'Outcome',    this.Outcome);
        end
    end
end
