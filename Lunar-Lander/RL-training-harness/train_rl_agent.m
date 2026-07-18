function trainStats = train_rl_agent(agent_type, reward_scheme)
% TRAIN_RL_AGENT Master orchestrator for Reinforcement Learning
%
% Inputs:
%   agent_type    - String specifying the type of agent to train (e.g., 'ddpg', 'ppo')
%   reward_scheme - (Optional) String specifying the reward scheme (e.g., 'DenseBaseline', 'SparseOnly')
%
% Outputs:
%   trainStats - Struct containing training performance data

    if nargin < 2
        reward_scheme = 'DenseBaseline';
    end

    % --- Reinforcement Learning Agent Setup & Training ---
    % Dynamically add the entire repository (and all subfolders) to the MATLAB path
    currentFolder = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(currentFolder, '..');
    addpath(genpath(repoRoot));
    
    % --- 1. Load the Environment ---
    % Initialize the wrapper with the desired reward scheme
    env = LunarLanderEnv(reward_scheme);
    
    % --- 2. Build the Agent ---
    obsInfo = getObservationInfo(env); 
    actInfo = getActionInfo(env);
    
    switch lower(agent_type)
        case 'ddpg'
            agent = build_ddpg_agent(obsInfo, actInfo, env.params.dt);
        case 'ppo'
            error('PPO agent builder is not yet implemented. Please resolve the pending GitHub issue.');
        otherwise
            error('Unknown agent type: %s. Supported types: ''ddpg''', agent_type);
    end
    
    % --- 3. Configure Training ---
    trainOpts = get_training_options();

    % --- 4. Execution ---
    disp('Starting AI Training...');
    trainStats = train(agent, env, trainOpts);
    
    % Save the final trained agent to disk with a unique name for A/B testing!
    agent_filename = sprintf('trained_lunar_agent_%s_%s.mat', lower(agent_type), reward_scheme);
    save(fullfile(repoRoot, agent_filename), 'agent');
    fprintf('Successfully saved: %s\n', agent_filename);
    
    % --- 5. Post-Training Visualization ---
    disp('Training Complete! Simulating the best agent...');
    
    simOpts = rlSimulationOptions('MaxSteps', trainOpts.MaxStepsPerEpisode);
    experience = sim(env, agent, simOpts);
    
    % Extract telemetry from the simulation experience
    obs_data = experience.Observation.LunarLanderStates.Data;
    act_data = experience.Action.LanderThrustAndTorque.Data;
    
    % Squeeze the 3D arrays into 1D vectors
    x      = squeeze(obs_data(1,:,:));
    y      = squeeze(obs_data(2,:,:));
    dy     = squeeze(obs_data(4,:,:));
    theta  = squeeze(obs_data(5,:,:));
    fuel   = squeeze(obs_data(7,:,:));
    
    % Scale the AI's neural network [-1, 1] thrust output back into physical Newtons for the plot
    raw_thrust = squeeze(act_data(1,:,:));
    thrust_history = (raw_thrust + 1) / 2 * env.params.max_main_thrust;
    
    % Create time vector
    t = (0:length(x)-1) * env.params.dt;
    
    % Generate dummy veto array since it isn't tracked in the observation states
    veto_history = zeros(size(t));
    
    % Launch the Advanced Visualizer
    disp('Launching Advanced Visualizer...');
    animate_lunar_lander(t, x, y, dy, theta, thrust_history, fuel, veto_history, env.params);
end