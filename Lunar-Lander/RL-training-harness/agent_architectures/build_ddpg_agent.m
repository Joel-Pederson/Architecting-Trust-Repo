function agent = build_ddpg_agent(obsInfo, actInfo, dt)
% BUILD_DDPG_AGENT Constructs a Deep Deterministic Policy Gradient Agent
%
% Inputs:
%   obsInfo - Observation specification from the environment
%   actInfo - Action specification from the environment
%   dt      - Simulation time step (for sample time)
%
% Outputs:
%   agent   - Fully configured rlDDPGAgent object

    % --- 1. Build the Critic Network ---
    % The Critic takes two inputs (The State and The Action) and merges them
    
    % Path 1: Process the State sensors
    statePath = [
        featureInputLayer(obsInfo.Dimension(1), 'Name', 'State')
        fullyConnectedLayer(64, 'Name', 'CriticStateFC1')
        reluLayer('Name', 'CriticRelu1')
        fullyConnectedLayer(64, 'Name', 'CriticStateFC2')];
    
    % Path 2: Process the Action joysticks
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
    
    % --- 2. Build the Actor Network ---
    % The Actor takes one input (The State) and outputs Actions
    
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
    
    % --- 3. Configure the DDPG Agent ---
    
    % Set the agent's clock to match our physics engine exactly
    agentOpts = rlDDPGAgentOptions('SampleTime', dt);
    
    % Add some exploratory noise so the AI wiggles the joysticks to learn
    agentOpts.ExplorationModel.Variance = 0.3; % 30% random wiggle
    agentOpts.ExplorationModel.VarianceDecayRate = 1e-4; % Slowly turn off wiggle as it gets smarter
    
    % Combine the Pilot and the Judge into a single Agent
    agent = rlDDPGAgent(actor, critic, agentOpts);

end
