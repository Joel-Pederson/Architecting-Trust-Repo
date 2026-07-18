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
end