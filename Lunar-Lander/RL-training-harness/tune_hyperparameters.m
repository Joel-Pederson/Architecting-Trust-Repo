function tune_hyperparameters()
% TUNE_HYPERPARAMETERS Master orchestrator for Bayesian Optimization
%
% This script uses MATLAB's bayesopt to intelligently search for the absolute
% best combination of DDPG hyperparameters (Learning Rates, Gamma, Noise)
% by running numerous "mini" training sessions and minimizing the negative reward.

    disp('=== Starting Bayesian Optimization Hyperparameter Tuning ===');

    % --- 0. Initialize UI Progress Dialog ---
    fig = uifigure('Name', 'RL Tuning', 'Position', [500 500 400 200]);
    d = uiprogressdlg(fig, 'Title', 'Bayesian Optimization', ...
        'Message', 'Initializing workers...', 'Cancelable', 'on');

    % --- 1. Define Optimization Variables ---
    % Actor Learning Rate (Log scale since it can range over orders of magnitude)
    actorLR = optimizableVariable('ActorLR', [1e-5, 1e-3], 'Transform', 'log');
    
    % Critic Learning Rate
    criticLR = optimizableVariable('CriticLR', [1e-4, 1e-2], 'Transform', 'log');
    
    % Discount Factor (Gamma) - How much it cares about long-term survival
    gamma = optimizableVariable('Gamma', [0.90, 0.999], 'Transform', 'none');
    
    % Exploration Noise Variance - How randomly it pushes the joystick
    noiseVar = optimizableVariable('NoiseVariance', [0.1, 0.6], 'Transform', 'none');
    
    vars = [actorLR, criticLR, gamma, noiseVar];

    % --- 2. Run Bayesian Optimization ---
    % Pass custom objective function to bayesopt.
    results = bayesopt(@training_objective, vars, ...
        'MaxObjectiveEvaluations', 30, ... % Run 30 different mini-sessions
        'UseParallel', true, ...           % Evaluate multiple configurations simultaneously across CPU cores!
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'OutputFcn', @update_progress, ...
        'IsObjectiveDeterministic', false); % RL training has random noise
        
    % Close UI
    if isvalid(fig)
        close(fig);
    end
        
    % --- 3. Display and Save Results ---
    bestParams = bestPoint(results);
    disp(' ');
    disp('=== OPTIMAL HYPERPARAMETERS FOUND ===');
    disp(bestParams);
    
    % Save to disk automatically so the harness can pull them in
    currentFolder = fileparts(mfilename('fullpath'));
    resultsFolder = fullfile(currentFolder, 'tuning_results');
    if ~exist(resultsFolder, 'dir')
        mkdir(resultsFolder);
    end
    savePath = fullfile(resultsFolder, 'optimal_ddpg_hyperparams.mat');
    
    optimal_hp = struct();
    optimal_hp.ActorLR = bestParams.ActorLR;
    optimal_hp.CriticLR = bestParams.CriticLR;
    optimal_hp.Gamma = bestParams.Gamma;
    optimal_hp.NoiseVariance = bestParams.NoiseVariance;
    
    save(savePath, 'optimal_hp');
    disp(['Optimal hyperparameters automatically saved to: ', savePath]);
    disp('The train_rl_agent harness will now natively pull these in!');

    % --- Nested Progress Function ---
    function stop = update_progress(results, state)
        stop = false;
        if d.CancelRequested
            stop = true;
            disp('Optimization cancelled by user.');
            return;
        end
        if strcmp(state, 'iteration')
            completed = results.NumObjectiveEvaluations;
            d.Value = completed / 30;
            d.Message = sprintf('Completed %d of 30 evaluations...', completed);
        end
    end

% --- Objective Function ---
function negReward = training_objective(params)
    % Dynamically add paths to ensure all dependencies are foundf
    currentFolder = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(currentFolder, '..');
    addpath(genpath(repoRoot));
    
    % 1. Initialize Environment
    env = LunarLanderEnv('DenseBaseline');
    obsInfo = getObservationInfo(env); 
    actInfo = getActionInfo(env);
    
    % 2. Map bayesopt variables to our hyperparams struct
    hp = struct();
    hp.ActorLR = params.ActorLR;
    hp.CriticLR = params.CriticLR;
    hp.Gamma = params.Gamma;
    hp.NoiseVariance = params.NoiseVariance;
    
    % 3. Build Agent with these specific hyperparameters
    agent = build_ddpg_agent(obsInfo, actInfo, env.params.dt, hp);
    
    % 4. Configure Mini-Training Session
    trainOpts = get_training_options();
    % Override max episodes to 300 for statistically significant testing
    % (Requires enough episodes for the agent to demonstrate learning capability)
    trainOpts.MaxEpisodes = 300; 
    % Disable UI plots so the screen doesn't get flooded, but enable console printing
    trainOpts.Plots = 'none';
    trainOpts.Verbose = true;
    % CRITICAL: Turn OFF inner parallelization. 
    % 1 core per agent, running multiple agents simultaneously via bayesopt.
    trainOpts.UseParallel = false;
    
    % 5. Run Training
    disp(['Evaluating Params: ActorLR=', num2str(hp.ActorLR), ', CriticLR=', num2str(hp.CriticLR), ', Gamma=', num2str(hp.Gamma)]);
    trainStats = train(agent, env, trainOpts);
    
    % 6. Extract Final Performance Metric
    % Take the average reward of the final 10 episodes to see where it plateaued
    if length(trainStats.EpisodeReward) >= 10
        final_avg_reward = mean(trainStats.EpisodeReward(end-9:end));
    else
        final_avg_reward = mean(trainStats.EpisodeReward);
    end
    
    % Bayesian optimization MINIMIZES the objective function.
    % Objective is to MAXIMIZE reward, so return negative reward.
    negReward = -final_avg_reward;
    disp(['Result (Negative Reward): ', num2str(negReward)]);
end

end % End of tune_hyperparameters
