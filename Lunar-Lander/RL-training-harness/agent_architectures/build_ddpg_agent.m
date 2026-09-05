function agent = build_ddpg_agent(obsInfo, actInfo, dt, hyperparams)
% BUILD_DDPG_AGENT Constructs a Deep Deterministic Policy Gradient Agent
%
% DDPG is the original continuous-control actor-critic. It is included in the trade study
% as the baseline, not because it is expected to win: a single critic trained with a max
% operator systematically overestimates Q-values, and DDPG is notoriously sensitive to
% hyperparameters. build_td3_agent is the direct fix for that failure mode.
%
% Inputs:
%   obsInfo     - Observation specification from the environment
%   actInfo     - Action specification from the environment
%   dt          - AGENT sample time (params.agent_dt = 0.1 s), NOT the physics step.
%                 The environment holds each action across control_decimation substeps.
%   hyperparams - (Optional) Struct containing tuning parameters
%
% Outputs:
%   agent       - Fully configured rlDDPGAgent object

    if nargin < 4 || isempty(hyperparams)
        hyperparams = load_hyperparams('ddpg');
    end

    nets = build_shared_networks(obsInfo, actInfo);

    % --- 1. Critic ---
    critic = rlQValueFunction(nets.qCritic, obsInfo, actInfo, ...
        'ObservationInputNames', 'State', 'ActionInputNames', 'Action');

    % --- 2. Actor ---
    actor = rlContinuousDeterministicActor(nets.detActor, obsInfo, actInfo);

    % --- 3. Agent options ---
    agentOpts = rlDDPGAgentOptions('SampleTime', dt);

    agentOpts.DiscountFactor = hyperparams.Gamma;
    agentOpts.ActorOptimizerOptions.LearnRate = hyperparams.ActorLR;
    agentOpts.CriticOptimizerOptions.LearnRate = hyperparams.CriticLR;

    % Prevent exploding gradients
    agentOpts.ActorOptimizerOptions.GradientThreshold = 1.0;
    agentOpts.CriticOptimizerOptions.GradientThreshold = 1.0;

    % --- REPLAY BUFFER ---
    % MATLAB defaults to 10,000 transitions. Episodes here run to 3,000 agent steps, so
    % the default held roughly three episodes: the agent was effectively learning
    % on-policy from a sliding window and continuously forgetting every landing it had
    % ever seen. Off-policy methods need orders of magnitude more.
    agentOpts.ExperienceBufferLength = 1e6;
    agentOpts.MiniBatchSize = 128;

    % --- EXPLORATION NOISE ---
    % VarianceDecayRate is applied PER STEP, not per episode. At 1e-4 with thousand-step
    % episodes, variance fell to ~1e-4 of its initial value within about 30 episodes:
    % exploration was dead long before there was anything worth exploiting. 1e-6 gives a
    % half-life of roughly 1,000 episodes, matching the training budget.
    agentOpts.NoiseOptions.StandardDeviation = sqrt(hyperparams.NoiseVariance);
    agentOpts.NoiseOptions.StandardDeviationDecayRate = 1e-6;
    agentOpts.NoiseOptions.StandardDeviationMin = 0.01;

    agent = rlDDPGAgent(actor, critic, agentOpts);
end
