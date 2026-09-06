function agent = build_td3_agent(obsInfo, actInfo, dt, hyperparams)
% BUILD_TD3_AGENT Twin Delayed Deep Deterministic policy gradient agent.
%
% TD3 is DDPG with three corrections, all of which matter for this problem:
%
%   1. Twin critics. Q is taken as the MINIMUM of two independently initialised critics,
%      which suppresses the systematic overestimation that makes DDPG diverge. On a task
%      with a large terminal reward (+/-5 at touchdown) and a long bootstrapped horizon,
%      overestimation compounds badly.
%   2. Delayed policy updates. The actor updates every other critic update, so the policy
%      chases a value estimate that has had time to settle.
%   3. Target policy smoothing. Noise is added to the target action, which stops the
%      critic from exploiting sharp peaks in its own approximation.
%
% Same networks, same sample time, same reward as every other arm of the trade study -
% only the learning rule differs.
%
% Inputs:
%   obsInfo, actInfo - specifications from the environment
%   dt               - AGENT sample time (params.agent_dt), not the physics step
%   hyperparams      - (Optional) struct of tuning parameters

    if nargin < 4 || isempty(hyperparams)
        hyperparams = load_hyperparams('td3');
    end

    nets = build_shared_networks(obsInfo, actInfo);

    % --- 1. Twin Critics ---
    % Both must be built from independently initialised networks. Handing TD3 two copies
    % of the same weights defeats the entire point: identical critics produce identical
    % estimates, min() becomes a no-op, and it degenerates back into DDPG.
    critic1 = rlQValueFunction(nets.qCritic, obsInfo, actInfo, ...
        'ObservationInputNames', 'State', 'ActionInputNames', 'Action');
    critic2 = rlQValueFunction(nets.qCritic2, obsInfo, actInfo, ...
        'ObservationInputNames', 'State', 'ActionInputNames', 'Action');

    % --- 2. Actor ---
    actor = rlContinuousDeterministicActor(nets.detActor, obsInfo, actInfo);

    % --- 3. Agent options ---
    agentOpts = rlTD3AgentOptions('SampleTime', dt);

    agentOpts.DiscountFactor = hyperparams.Gamma;
    agentOpts.ActorOptimizerOptions.LearnRate = hyperparams.ActorLR;
    agentOpts.ActorOptimizerOptions.GradientThreshold = 1.0;

    % CriticOptimizerOptions is a 1x2 array, one entry per twin
    for i = 1:numel(agentOpts.CriticOptimizerOptions)
        agentOpts.CriticOptimizerOptions(i).LearnRate = hyperparams.CriticLR;
        agentOpts.CriticOptimizerOptions(i).GradientThreshold = 1.0;
    end

    agentOpts.ExperienceBufferLength = 1e6;
    agentOpts.MiniBatchSize = 128;

    % KEEP whatever is already in the buffer when train() is called.
    %
    % The toolbox default is TRUE, which empties the replay buffer before the first
    % update. That silently destroys demonstration seeding: generate_demonstrations
    % writes ~140,000 expert transitions into the buffer, train() throws all of them
    % away, and the run looks exactly like ordinary from-scratch training that failed.
    % Nothing errors and nothing warns.
    %
    % Setting it false has no effect on a from-scratch run, where the buffer is empty
    % anyway, so this is safe for every existing caller.
    agentOpts.ResetExperienceBufferBeforeTraining = false;

    % Exploration noise, decayed per STEP (see build_ddpg_agent for why 1e-6)
    agentOpts.ExplorationModel.StandardDeviation = sqrt(hyperparams.NoiseVariance);
    agentOpts.ExplorationModel.StandardDeviationDecayRate = 1e-6;
    agentOpts.ExplorationModel.StandardDeviationMin = 0.01;

    % Target policy smoothing
    agentOpts.TargetPolicySmoothModel.StandardDeviation = hyperparams.TargetPolicyNoise;
    agentOpts.TargetPolicySmoothModel.LowerLimit = -hyperparams.TargetPolicyNoiseClip;
    agentOpts.TargetPolicySmoothModel.UpperLimit =  hyperparams.TargetPolicyNoiseClip;

    % Delayed actor updates
    agentOpts.PolicyUpdateFrequency = hyperparams.PolicyUpdateFrequency;

    agent = rlTD3Agent(actor, [critic1 critic2], agentOpts);
end
