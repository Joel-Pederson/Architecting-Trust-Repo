function trainOpts = get_training_options(agent_type, max_episodes)
% GET_TRAINING_OPTIONS Returns the training configuration for a given architecture.
%
% Centralizes training options so they can be managed in one place or replaced with
% Bayesian optimization configurations.
%
% The on-policy / off-policy distinction is not cosmetic. PPO discards its experience
% after each update, so it must gather trajectories synchronously across workers; the
% async 'Experiences' mode used for the off-policy agents feeds a shared replay buffer,
% which PPO does not have. Sending it async experiences produces stale-gradient training
% that silently underperforms rather than erroring.
%
% Inputs:
%   agent_type   - (Optional) 'ddpg' | 'td3' | 'sac' | 'ppo'. Default 'ddpg'.
%   max_episodes - (Optional) episode budget. Default 5000.
%
% Outputs:
%   trainOpts - rlTrainingOptions object

    if nargin < 1 || isempty(agent_type),   agent_type = 'ddpg'; end
    if nargin < 2 || isempty(max_episodes), max_episodes = 5000; end

    params = get_sim_params();
    on_policy = strcmpi(agent_type, 'ppo');

    trainOpts = rlTrainingOptions(...
        'MaxEpisodes', max_episodes, ...
        'MaxStepsPerEpisode', params.max_agent_steps, ...   % 3000 decisions = 300 s at 10 Hz
        'ScoreAveragingWindowLength', 50, ...               % Default of 5 is too noisy to read
        'StopTrainingCriteria', 'AverageReward', ...
        'StopTrainingValue', 6, ...
        'SaveAgentCriteria', 'EpisodeReward', ...
        'SaveAgentValue', 6, ...
        'UseParallel', true, ...
        'Plots', 'training-progress');

    if on_policy
        % PPO: workers must return complete, current trajectories.
        trainOpts.ParallelizationOptions.Mode = 'sync';
        trainOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';
    else
        % Off-policy: async workers stream experience into the shared replay buffer.
        %
        % Apple Silicon note - because M-series chips use unified memory, shipping large
        % gradient arrays between workers thrashes the cache. Having workers compute
        % experiences and return them in chunks minimizes thread locks.
        trainOpts.ParallelizationOptions.Mode = 'async';
        trainOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';
        trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 128;
    end
end
