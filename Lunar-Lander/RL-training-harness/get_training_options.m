function trainOpts = get_training_options()
% GET_TRAINING_OPTIONS Returns the hyperparameter configuration for RL training
%
% This function centralizes the training options so they can be easily 
% managed or replaced with Bayesian optimization configurations in the future.
%
% Outputs:
%   trainOpts - rlTrainingOptions object

    trainOpts = rlTrainingOptions(...
        'MaxEpisodes', 2000, ...               % Try to land 2,000 times
        'MaxStepsPerEpisode', 45000, ...       % Max 900 seconds per flight (45000 * 0.02s) to support orbital descents
        'StopTrainingCriteria', 'AverageReward', ... 
        'StopTrainingValue', 7500, ...         
        'SaveAgentCriteria', 'EpisodeReward', ... % Save a backup if it has a great landing
        'SaveAgentValue', 7500, ...
        'UseParallel', true, ...               % Distribute episodes across multiple CPU cores
        'Plots', 'training-progress');         
    
    % Configure async parallel execution for maximum speed
    trainOpts.ParallelizationOptions.Mode = 'async';
    
    % --- Apple Silicon Optimizations ---
    % Because the M series chips use Unified Memory, sending massive gradient arrays between
    % CPU workers can bottleneck the cache. It is much faster to have the workers
    % calculate 'Experiences' (State, Action, Reward) and send those back to the
    % main orchestrator thread in chunks of 64 or 128 steps to minimize thread locks.
    trainOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';
    trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 128;

end
