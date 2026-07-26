classdef LunarLanderEnv < rl.env.MATLABEnvironment
    % LUNARLANDERENV: Reinforcement Learning Environment Wrapper. Written with Gemini assistance
    % This class acts as the bridge between the physical simulation and the AI agent

    properties
        % Hardware Limits and Reward Weights
        params
        weights
        State
        
        % Configurable Reward Scheme
        RewardScheme
        
        % Previous action (tracked to calculate smoothness penalties)
        u_prev = [0; 0];
    end
    
    properties (Access = protected)
        % Internal State Tracking
        IsDone = false;
    end
    
    methods
        function this = LunarLanderEnv(rewardScheme)
            % CONSTRUCTOR: Defines the rules of the universe for the AI
            
            % Default to DenseBaseline if not provided
            if nargin < 1
                rewardScheme = 'DenseBaseline';
            end
            
            % 1. Define Observation Space (8 Variables)
            % [x; y; dx; dy; theta; dtheta; m_main_fuel; m_rcs_fuel]
            obsInfo = rlNumericSpec([8 1]);
            obsInfo.Name = 'LunarLanderStates';
            
            % 2. Define Action Space (2 Variables)
            % Force the agent to output values between [-1, 1]. Scale these 
            % to the actual physics hardware limits inside the step() function to avoid ML large numbers.
            actInfo = rlNumericSpec([2 1], 'LowerLimit', [-1; -1], 'UpperLimit', [1; 1]);
            actInfo.Name = 'LanderThrustAndTorque';
            
            % 3. Initialize the superclass (MATLAB RL Environment)
            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);
            
            % Now that the object is fully constructed, assign properties
            this.RewardScheme = rewardScheme;
            
            % 4. Load our modular configurations (Single Source of Truth)
            this.params = get_sim_params();
            this.weights = get_reward_weights();
        end
        
        function [Observation, LoggedSignals] = reset(this)
            % RESET: Called automatically at the start of every new training episode
            
            % Domain Randomization (Uniform Curriculum Learning)
            % Due to use of heavily parallelized training (8 CPU cores), the workers 
            % cannot sync a unified 'Episode Count' with each other. Instead, use 
            % Domain Randomization: every single episode randomly picks a difficulty phase.
            % This fills the Neural Network's Replay Buffer with a diverse mix of experiences
            
            phase_selector = rand();
            
            if phase_selector < 0.33
                % Phase 1: Hover and Touchdown (Easy)
                % Agent starts 50 meters off the ground with near-zero velocity.
                % Teaches the AI fine throttle control and how to actually touch down.
                init_x = randn() * 10;
                init_y = 50 + (randn() * 5);
                init_dx = randn() * 2;
                init_dy = -2 + (randn() * 1);
                init_theta = randn() * 0.05;
                init_dtheta = randn() * 0.01;
                
            elseif phase_selector < 0.66
                % Phase 2: Medium Descent (Medium)
                % Agent starts 2,000 meters up falling at 20 m/s.
                % Teaches the AI how to safely decelerate and manage fuel over medium distances.
                init_x = randn() * 50;
                init_y = 2000 + (randn() * 50);
                init_dx = 50 + (randn() * 10);
                init_dy = -20 + (randn() * 5);
                init_theta = randn() * 0.1;
                init_dtheta = randn() * 0.02;
                
            else
                % Phase 3: Powered Descent Initiation (Hard)
                % Agent starts in orbit at 15,000m going 1,700 m/s.
                % Teaches the AI complex orbital mechanics and massive centrifugal forces.
                init_x = randn() * 100;
                init_y = 15000 + (randn() * 50);
                init_dx = 1700 + (randn() * 10);
                init_dy = -10 + (randn() * 2);
                init_theta = randn() * 0.1;
                init_dtheta = randn() * 0.05;
            end
            
            init_main_fuel = 8200;            % Always start with full main fuel
            init_rcs_fuel = 300;              % Always start with full RCS fuel
            
            % Set the internal state
            this.State = [init_x; init_y; init_dx; init_dy; init_theta; init_dtheta; init_main_fuel; init_rcs_fuel];
            
            % Reset historical tracking
            this.u_prev = [0; 0];
            this.IsDone = false;
            
            % Return initial normalized observation to the AI
            Observation = this.normalize_state(this.State);
            LoggedSignals = [];
        end
        
        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            % STEP: The main loop called by the AI every 0.1 seconds
            
            % Scale neural-network outputs into physical actuator commands
            % 1. SCALE ACTIONS (Neural Net [-1, 1] -> Physics Domain)
            % Thrust: The AI's native range is 2 units wide [-1 to 1]. Add 1 to shift it to [0 to 2], 
            % and divide by 2 to compress it into a [0 to 1] throttle percentage before multiplying by max thrust.
            u_thrust = (Action(1) + 1) / 2 * this.params.max_main_thrust; 
            
            % Torque: Map [-1, 1] directly to [-max_side_torque, max_side_torque] because side thrusters are bidirectional.
            u_torque = Action(2) * this.params.max_side_torque;
            
            u_nominal = [u_thrust; u_torque];
            
            % Bypass the safety sidecar during training so the AI learns from true physical consequences
            u_actual = u_nominal;
            VetoTriggered = false;
            
            % 2. THE PHYSICS ENGINE
            % Calculate derivatives and move time forward by dt (Euler Integration)
            this.State = this.State + dxdt * this.params.dt;
            % Wrap angle theta to [-pi, pi] so it never accumulates indefinitely
            this.State(5) = atan2(sin(this.State(5)), cos(this.State(5)));
            
            % 3. THE REWARD CALCULATOR
            % Determine how well the AI is doing by routing to the selected reward scheme
            switch this.RewardScheme
                case 'DenseBaseline'
                    [Reward, IsDone] = reward_dense_baseline(this.State, u_actual, this.u_prev, VetoTriggered, this.params, this.weights);
                case 'SparseOnly'
                    [Reward, IsDone] = reward_sparse_only(this.State, u_actual, this.u_prev, VetoTriggered, this.params, this.weights);
                otherwise
                    error('Unknown reward scheme selected: %s', this.RewardScheme);
            end
            
            % SCALE REWARD: Shrink the massive physical scores [-400000, 10000] 
            % down to a mathematically stable [-40, 1] range for the neural network.
            Reward = Reward / 10000;
            
            % 4. UPDATE ENVIRONMENT
            this.IsDone = IsDone;
            this.u_prev = u_actual;
            Observation = this.normalize_state(this.State);
            LoggedSignals = [];
            
            % Notify the MATLAB environment that a step has occurred
            notifyEnvUpdated(this);
        end
    end
    
    methods (Access = private)
        function norm_state = normalize_state(~, raw_state)
            % NORMALIZE_STATE: Uses the central get_ai_observation function
            % so training and deployment normalization are identical.
            norm_state = get_ai_observation(raw_state);
        end
    end
end