function agent = build_sac_agent(obsInfo, actInfo, dt, hyperparams)
% BUILD_SAC_AGENT Soft Actor-Critic agent.
%
% SAC optimises reward PLUS policy entropy, which changes the character of exploration
% entirely. DDPG and TD3 explore by bolting hand-tuned Gaussian noise onto a
% deterministic policy and decaying it on a schedule - and getting that schedule wrong is
% exactly what killed the original harness, where noise decayed to nothing within ~30
% episodes. SAC instead learns how stochastic to be, and tunes the entropy weight
% automatically against a target entropy. There is no decay schedule to get wrong.
%
% That property makes it the most likely architecture to survive this particular reward
% landscape, where the interesting behaviour (a slow controlled descent that never trips
% the barrier) is a narrow region the agent has to keep probing to find.
%
% Inputs:
%   obsInfo, actInfo - specifications from the environment
%   dt               - AGENT sample time (params.agent_dt), not the physics step
%   hyperparams      - (Optional) struct of tuning parameters

    if nargin < 4 || isempty(hyperparams)
        hyperparams = load_hyperparams('sac');
    end

    nets = build_shared_networks(obsInfo, actInfo);

    % --- 1. Twin Critics (SAC uses the same min-of-two trick as TD3) ---
    critic1 = rlQValueFunction(nets.qCritic, obsInfo, actInfo, ...
        'ObservationInputNames', 'State', 'ActionInputNames', 'Action');
    critic2 = rlQValueFunction(nets.qCritic2, obsInfo, actInfo, ...
        'ObservationInputNames', 'State', 'ActionInputNames', 'Action');

    % --- 2. Stochastic Actor ---
    % Outputs a mean and standard deviation per action dimension. SAC applies its own
    % bounded (tanh-squashed) transform internally, which is why the shared Gaussian
    % actor deliberately leaves its mean head unsquashed.
    actor = rlContinuousGaussianActor(nets.gaussActor, obsInfo, actInfo, ...
        'ActionMeanOutputNames', 'ActorMean', ...
        'ActionStandardDeviationOutputNames', 'ActorStd');

    % --- 3. Agent options ---
    agentOpts = rlSACAgentOptions('SampleTime', dt);

    agentOpts.DiscountFactor = hyperparams.Gamma;
    agentOpts.ActorOptimizerOptions.LearnRate = hyperparams.ActorLR;
    agentOpts.ActorOptimizerOptions.GradientThreshold = 1.0;

    for i = 1:numel(agentOpts.CriticOptimizerOptions)
        agentOpts.CriticOptimizerOptions(i).LearnRate = hyperparams.CriticLR;
        agentOpts.CriticOptimizerOptions(i).GradientThreshold = 1.0;
    end

    agentOpts.ExperienceBufferLength = 1e6;
    agentOpts.MiniBatchSize = 128;

    % Automatic entropy tuning. TargetEntropy is left at the toolbox default
    % (-numel(action)), the standard heuristic, so the agent anneals its own exploration.
    agentOpts.EntropyWeightOptions.EntropyWeight = hyperparams.EntropyWeight;
    agentOpts.EntropyWeightOptions.LearnRate = hyperparams.EntropyLearnRate;

    agent = rlSACAgent(actor, [critic1 critic2], agentOpts);
end
