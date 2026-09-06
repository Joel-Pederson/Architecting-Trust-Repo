function [trainStats, agent] = train_rl_agent(agent_type, reward_scheme, guardian_mode, do_visualize)
% TRAIN_RL_AGENT Master orchestrator for Reinforcement Learning
%
% Inputs:
%   agent_type    - String specifying the agent architecture (e.g. 'ddpg')
%   reward_scheme - (Optional) 'DenseBaseline' (default) or 'SparseOnly'
%   guardian_mode - (Optional) 'on' (default) trains with the Developmental Guardian
%                   filtering every command; 'off' trains the unprotected control arm
%                   for the A/B study.
%   do_visualize  - (Optional) true (default) to animate the trained policy afterwards
%
% Outputs:
%   trainStats - Struct containing training performance data
%   agent      - The trained agent

    if nargin < 2 || isempty(reward_scheme),  reward_scheme = 'DenseBaseline'; end
    if nargin < 3 || isempty(guardian_mode),  guardian_mode = 'on';            end
    if nargin < 4 || isempty(do_visualize),   do_visualize  = true;            end

    % Dynamically add the entire repository (and all subfolders) to the MATLAB path
    currentFolder = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(currentFolder, '..');
    addpath(genpath(repoRoot));

    % --- 1. Load the Environment ---
    env = LunarLanderEnv(reward_scheme, guardian_mode);

    % --- 2. Build the Agent ---
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);

    % Pass the AGENT sample time (0.1 s), not the physics step (0.02 s).
    agent = build_agent(agent_type, obsInfo, actInfo, env.params.agent_dt);

    % --- 3. Configure Training ---
    trainOpts = get_training_options(agent_type);

    % --- 4. Execution ---
    fprintf('Starting AI Training  [agent=%s  reward=%s  guardian=%s]\n', ...
        lower(agent_type), reward_scheme, guardian_mode);
    trainStats = train(agent, env, trainOpts);

    % Save the final trained agent with a unique name for A/B testing
    agent_filename = sprintf('trained_lunar_agent_%s_%s_guardian_%s.mat', ...
        lower(agent_type), reward_scheme, guardian_mode);
    save(fullfile(repoRoot, agent_filename), 'agent');
    fprintf('Successfully saved: %s\n', agent_filename);

    % --- 5. Post-Training Visualization ---
    if do_visualize
        disp('Training Complete! Simulating the trained agent...');
        simulate_and_animate(env, agent);
    end
end


function simulate_and_animate(env, agent)
% Runs one episode and renders it.
%
% Telemetry comes from rollout_episode, which carries the TRUE physics state in real
% units. The previous version reconstructed telemetry by multiplying the normalized
% observation by 500000 / 20000 / 150 while get_ai_observation actually normalizes by
% 1000 / 3000 / 100 - so plotted lateral position was 500x too large and altitude 6.7x
% too large. With axis equal on the tracking view that flattened every trajectory onto
% the ground, which is why the lander appeared to start just above the surface.

    ep = rollout_episode(env, agent);

    fprintf('Episode outcome: %s | reward: %.2f | sidecar engagements: %d | steps: %d\n', ...
        ep.outcome, ep.reward, ep.veto_count, ep.steps);

    disp('Launching Advanced Visualizer...');
    animate_lunar_lander(ep.t, ep.states(1, :), ep.states(2, :), ep.states(4, :), ...
        ep.states(5, :), ep.controls(1, :), ep.states(7, :), double(ep.veto), env.params);
end
