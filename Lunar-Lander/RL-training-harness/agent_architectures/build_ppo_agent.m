function agent = build_ppo_agent(obsInfo, actInfo, dt, hyperparams)
% BUILD_PPO_AGENT Proximal Policy Optimization agent.
%
% Resolves the previously unimplemented PPO branch in train_rl_agent.
%
% PPO is the one ON-POLICY architecture in the trade study, and that changes what it
% needs from the harness in ways worth being explicit about:
%
%   - No replay buffer. It learns from freshly gathered trajectories and discards them,
%    so ExperienceBufferLength and warm-start options do not apply. get_training_options
%    detects this and switches to sync parallelism accordingly.
%   - Its exploration comes from the stochastic policy itself, regulated by an entropy
%     loss term rather than an injected noise schedule.
%   - The clipped surrogate objective bounds how far the policy can move per update,
%     which makes it markedly more stable than DDPG but slower in sample terms.
%
% It is included because on-policy methods are far less sensitive to the reward-scale and
% bootstrapping pathologies that broke the original harness, which makes PPO a useful
% control: if PPO learns and the off-policy methods do not, the remaining problem is in
% value estimation rather than in the environment.
%
% Inputs:
%   obsInfo, actInfo - specifications from the environment
%   dt               - AGENT sample time (params.agent_dt), not the physics step
%   hyperparams      - (Optional) struct of tuning parameters

    if nargin < 4 || isempty(hyperparams)
        hyperparams = load_hyperparams('ppo');
    end

    nets = build_shared_networks(obsInfo, actInfo);

    % --- 1. Value Critic V(s) ---
    % PPO estimates a state value, not a state-action value: it needs the advantage of
    % the actions it actually took, not a Q surface over all possible actions.
    critic = rlValueFunction(nets.vCritic, obsInfo);

    % --- 2. Stochastic Actor ---
    actor = rlContinuousGaussianActor(nets.gaussActor, obsInfo, actInfo, ...
        'ActionMeanOutputNames', 'ActorMean', ...
        'ActionStandardDeviationOutputNames', 'ActorStd');

    % --- 3. Agent options ---
    agentOpts = rlPPOAgentOptions('SampleTime', dt);

    agentOpts.DiscountFactor = hyperparams.Gamma;
    agentOpts.ActorOptimizerOptions.LearnRate = hyperparams.ActorLR;
    agentOpts.ActorOptimizerOptions.GradientThreshold = 1.0;
    agentOpts.CriticOptimizerOptions.LearnRate = hyperparams.CriticLR;
    agentOpts.CriticOptimizerOptions.GradientThreshold = 1.0;

    % Trajectory length gathered before each policy update
    agentOpts.ExperienceHorizon = hyperparams.ExperienceHorizon;
    agentOpts.MiniBatchSize = 128;
    agentOpts.NumEpoch = hyperparams.NumEpoch;

    % Clipped surrogate objective - the mechanism that keeps updates conservative
    agentOpts.ClipFactor = hyperparams.ClipFactor;

    % Entropy bonus. Without it the Gaussian collapses to near-deterministic early and
    % PPO stops exploring, which on this task means never discovering a slow descent.
    agentOpts.EntropyLossWeight = hyperparams.EntropyLossWeight;

    agent = rlPPOAgent(actor, critic, agentOpts);
end
