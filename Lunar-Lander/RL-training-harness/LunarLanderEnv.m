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
            
            % RANDOMIZED initial conditions for robust AI training. This is
            % a light amount of variation, potentially should increase
            % variation in the future.
            % randn() generates a normally distributed random number (bell curve)
            init_x = randn() * 100;           % Start up to ~100m off-center
            init_y = 15000 + (randn() * 50);  % Start around 15,000m (Historical Apollo 11 PDI altitude ~50,000 ft)
            init_dx = 1700 + (randn() * 10);  % Start at historical orbital velocity (1,700 m/s)
            init_dy = -10 + (randn() * 2);    % Start falling around -10 m/s
            init_theta = randn() * 0.1;       % Start slightly tilted (up to ~5.7 deg). Mimics realistic mechanical wobble from detaching from the command module in orbit.
            init_dtheta = randn() * 0.05;     % Start with a slight spin (up to ~2.8 deg/s). Forces the AI to learn to use side torque to stabilize immediately.
            init_main_fuel = 8200;            % Always start with full main fuel (8200 kg). Ensures the AI has a consistent energy budget to solve the randomized physics puzzle.
            init_rcs_fuel = 300;              % Always start with full RCS fuel (300 kg).
            
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
            dxdt = lunar_lander_dynamics(this.State, u_actual, this.params);
            this.State = this.State + dxdt * this.params.dt;
            
            % 3. THE REWARD CALCULATOR
            % Determine how well the AI is doing by routing to the selected reward scheme
            switch this.RewardScheme
                case 'DenseBaseline'
                    [Reward, IsDone] = reward_dense_baseline(this.State, u_actual, this.u_prev, VetoTriggered, this.params);
                case 'SparseOnly'
                    [Reward, IsDone] = reward_sparse_only(this.State, u_actual, this.u_prev, VetoTriggered, this.params);
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
        function norm_state = normalize_state(this, raw_state)
            % NORMALIZE_STATE: Compresses true physics numbers into roughly [-1, 1] bounds
            % so the Neural Network doesn't suffer from vanishing gradients.
            norm_state = [
                raw_state(1) / 500000;
                raw_state(2) / 20000;
                raw_state(3) / 2000;
                raw_state(4) / 150;
                raw_state(5) / pi;
                raw_state(6) / pi;
                raw_state(7) / 8200;
                raw_state(8) / 300
            ];
        end
    end
end