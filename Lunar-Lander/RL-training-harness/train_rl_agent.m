% --- Reinforcement Learning Agent Setup & Training ---
% Dynamically add the entire repository (and all subfolders) to the MATLAB path
currentFolder = fileparts(mfilename('fullpath'));
repoRoot = fullfile(currentFolder, '..');
addpath(genpath(repoRoot));

% --- 1. Load the Environment ---
% Initialize the wrapper
env = LunarLanderEnv();
obsInfo = getObservationInfo(env); % Tells AI it has 7 sensors
actInfo = getActionInfo(env);      % Tells AI it has 2 joysticks [-1, 1]

% --- 2. Build the Critic Network (The Judge) ---
% The Critic takes TWO inputs (The State and The Action) and merges them

% Path 1: Process the 7 State sensors
statePath = [
    featureInputLayer(obsInfo.Dimension(1), 'Name', 'State')
    fullyConnectedLayer(64, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(64, 'Name', 'CriticStateFC2')];

% Path 2: Process the 2 Action joysticks
actionPath = [
    featureInputLayer(actInfo.Dimension(1), 'Name', 'Action')
    fullyConnectedLayer(64, 'Name', 'CriticActionFC1')];

% Path 3: Merge them together to predict the final Score (Q-Value)
commonPath = [
    additionLayer(2, 'Name', 'add') % Adds the State and Action math together
    reluLayer('Name', 'CriticCommonRelu')
    fullyConnectedLayer(1, 'Name', 'QValue')]; % Outputs a single number: Predicted Score

% Assemble the Critic Network
criticNet = layerGraph();
criticNet = addLayers(criticNet, statePath);
criticNet = addLayers(criticNet, actionPath);
criticNet = addLayers(criticNet, commonPath);
criticNet = connectLayers(criticNet, 'CriticStateFC2', 'add/in1');
criticNet = connectLayers(criticNet, 'CriticActionFC1', 'add/in2');

% Tell MATLAB this network represents a Q-Value Critic
critic = rlQValueRepresentation(criticNet, obsInfo, actInfo, ...
    'Observation', {'State'}, 'Action', {'Action'});

% --- 3. Build the Actor Network (The Pilot) ---
% The Actor takes ONE input (The State) and outputs Actions.

actorNet = [
    featureInputLayer(obsInfo.Dimension(1), 'Name', 'State')
    fullyConnectedLayer(64, 'Name', 'ActorFC1')
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(64, 'Name', 'ActorFC2')
    reluLayer('Name', 'ActorRelu2')
    fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'ActionOutput') % Outputs 2 numbers
    tanhLayer('Name', 'ActionTanh')]; % CRITICAL: Squashes outputs to exactly [-1, 1]

% Tell MATLAB this network represents a Deterministic Actor
actor = rlDeterministicActorRepresentation(actorNet, obsInfo, actInfo, ...
    'Observation', {'State'}, 'Action', {'ActionOutput'});

% --- 4. Configure the DDPG Agent ---

% Set the agent's clock to match our physics engine exactly (dt = 0.02)
agentOpts = rlDDPGAgentOptions('SampleTime', env.params.dt);

% Add some exploratory noise so the AI wiggles the joysticks to learn
agentOpts.ExplorationModel.Variance = 0.3; % 30% random wiggle
agentOpts.ExplorationModel.VarianceDecayRate = 1e-4; % Slowly turn off wiggle as it gets smarter

% Combine the Pilot and the Judge into a single Agent
agent = rlDDPGAgent(actor, critic, agentOpts);

% --- 5. Training Settings & Execution ---

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 2000, ...               % Try to land 2,000 times
    'MaxStepsPerEpisode', 2000, ...        % Max 40 seconds per flight (2000 * 0.02s)
    'ScoreExceeds', 9000, ...              % Stop early if it scores >9,000 points
    'StopTrainingCriteria', 'AverageReward', ... 
    'StopTrainingValue', 9000, ...         
    'SaveAgentCriteria', 'EpisodeReward', ... % Save a backup if it has a great landing
    'SaveAgentValue', 5000, ...
    'Plots', 'training-progress');         % Show us the beautiful graph!

disp('Starting AI Training...');

trainStats = train(agent, env, trainOpts);