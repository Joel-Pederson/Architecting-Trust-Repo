classdef LunarLanderEnv < rl.env.MATLABEnvironment
    % LUNARLANDERENV: Reinforcement Learning Environment Wrapper. Written
    % with Gemini assistance
    % This class acts as the bridge between the physical simulation and the AI agent.
    
    properties
        % Hardware Limits and Reward Weights
        params
        weights
        State
        
        % Simulation Timestep (0.1 seconds per frame)
        Ts = 0.1; 
        
        % Previous action (tracked to calculate smoothness penalties)
        u_prev = [0; 0];
    end
    
    properties (Access = protected)
        % Internal State Tracking
        IsDone = false;
    end
    
    methods
        function this = LunarLanderEnv()
            % CONSTRUCTOR: Defines the rules of the universe for the AI
            
            % 1. Define Observation Space (7 Variables)
            % [x; y; dx; dy; theta; dtheta; m_fuel]
            obsInfo = rlNumericSpec([7 1]);
            obsInfo.Name = 'LunarLanderStates';
            
            % 2. Define Action Space (2 Variables)
            % We force the AI to output values between [-1, 1]. Scale these 
            % to the actual physics hardware limits inside the step() function to avoid ML large numbers.
            actInfo = rlNumericSpec([2 1], 'LowerLimit', [-1; -1], 'UpperLimit', [1; 1]);
            actInfo.Name = 'LanderThrustAndTorque';
            
            % 3. Initialize the parent class
            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);
            
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
            init_y = 5000 + (randn() * 50);   % Start around 5000m, varying slightly
            init_dx = randn() * 10;           % Start drifting sideways up to 10 m/s
            init_dy = -10 + (randn() * 2);    % Start falling around -10 m/s
            init_theta = randn() * 0.1;       % Start slightly tilted (up to ~5.7 deg). Mimics realistic mechanical wobble from detaching from the command module in orbit.
            init_dtheta = randn() * 0.05;     % Start with a slight spin (up to ~2.8 deg/s). Forces the AI to learn to use side torque to stabilize immediately.
            init_fuel = 1000;                 % Always start with full fuel (1000 kg). Ensures the AI has a consistent energy budget to solve the randomized physics puzzle.
            
            % Set the internal state
            this.State = [init_x; init_y; init_dx; init_dy; init_theta; init_dtheta; init_fuel];
            
            % Reset historical tracking
            this.u_prev = [0; 0];
            this.IsDone = false;
            
            % Return initial observation to the AI
            Observation = this.State;
            LoggedSignals = [];
        end
        
        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            % STEP: The main loop called by the AI every 0.1 seconds
            
            % 1. SCALE ACTIONS (Neural Net [-1, 1] -> Physics Domain)
            % Thrust: Map [-1, 1] to [0, max_main_thrust]
            u_thrust = (Action(1) + 1) / 2 * this.params.max_main_thrust; 
            % Torque: Map [-1, 1] to [-max_side_torque, max_side_torque]
            u_torque = Action(2) * this.params.max_side_torque;
            
            u_nominal = [u_thrust; u_torque];
            
            % 2. THE SAFETY SIDECAR
            % Intercept the AI's command. If it's about to crash, veto it.
            [u_actual, VetoTriggered, ~, ~] = safety_sidecar_filter(this.State, u_nominal, this.params);
            
            % 3. THE PHYSICS ENGINE
            % Calculate derivatives and move time forward by Ts (Euler Integration)
            dxdt = lunar_lander_dynamics(this.State, u_actual, this.params);
            this.State = this.State + dxdt * this.Ts;
            
            % 4. THE REWARD CALCULATOR
            % Determine how well the AI is doing and if the simulation is over
            [Reward, IsDone] = calculate_reward(this.State, u_actual, this.u_prev, VetoTriggered, this.params);
            
            % 5. UPDATE ENVIRONMENT
            this.IsDone = IsDone;
            this.u_prev = u_actual;
            Observation = this.State;
            LoggedSignals = [];
            
            % Notify the MATLAB environment that a step has occurred
            notifyEnvUpdated(this);
        end
    end
end