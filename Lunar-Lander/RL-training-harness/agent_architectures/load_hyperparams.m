function hp = load_hyperparams(agent_type)
% LOAD_HYPERPARAMS Returns tuning parameters for an architecture.
%
% Prefers a saved Bayesian-optimization result from tune_hyperparameters if one exists on
% disk, otherwise falls back to hand-picked defaults.
%
% ON GAMMA: read every discount factor here against the AGENT sample time (0.1 s), not
% the physics step (0.02 s). The effective horizon is 1/(1-gamma) STEPS:
%     gamma = 0.99   ->  100 steps ->  10 s
%     gamma = 0.995  ->  200 steps ->  20 s
%     gamma = 0.999  -> 1000 steps -> 100 s
% The original harness paired gamma=0.99 with a 0.02 s sample time, giving a 2 SECOND
% lookahead on descents lasting minutes. The touchdown reward was discounted to roughly
% 1e-13 by the time it reached the start of an episode - the agent was structurally
% incapable of perceiving the goal it was being trained on.
%
% Inputs:
%   agent_type - 'ddpg' | 'td3' | 'sac' | 'ppo'
%
% Outputs:
%   hp - struct of hyperparameters for that architecture

    agent_type = lower(agent_type);

    currentFolder = fileparts(mfilename('fullpath'));
    hyperparamFile = fullfile(currentFolder, '..', 'tuning_results', ...
        sprintf('optimal_%s_hyperparams.mat', agent_type));

    if isfile(hyperparamFile)
        data = load(hyperparamFile);
        candidate = data.optimal_hp;

        % --- PROVENANCE CHECK ---
        % Tuned hyperparameters are only valid for the harness they were tuned against.
        % A Gamma optimised when the agent ran at 50 Hz means something entirely
        % different once the agent runs at 10 Hz, and loading it silently reverts the
        % discount-horizon fix while printing a reassuring "loaded tuned parameters"
        % message. This happened for real: an Aug-2025 file carried Gamma = 0.9926,
        % which is a 13.5 s horizon here versus the 2.7 s it was tuned for.
        params = get_sim_params();
        [valid, reason] = check_provenance(candidate, params);

        if valid
            hp = candidate;
            fprintf('Loaded tuned hyperparameters for %s from disk.\n', upper(agent_type));
            return;
        end

        warning('loadHyperparams:StaleTuning', ...
            ['Ignoring tuned hyperparameters in %s\n' ...
             '  Reason: %s\n' ...
             '  Falling back to defaults. Re-run tune_hyperparameters to regenerate.'], ...
            hyperparamFile, reason);
    end

    % Shared defaults
    hp = struct();

    % Gamma is read from get_sim_params rather than written here. The environment applies
    % potential-based shaping as gamma*Phi(s') - Phi(s), so it needs the SAME gamma the
    % agent discounts with; two independent copies drifting apart would quietly void the
    % policy-invariance guarantee while everything still appeared to run.
    hp.Gamma    = get_sim_params().gamma_agent;
    hp.ActorLR  = 1e-4;
    hp.CriticLR = 1e-3;

    switch agent_type
        case 'ddpg'
            hp.NoiseVariance = 0.3;

        case 'td3'
            % Slightly higher critic rate is safe here: twin critics suppress the
            % overestimation that makes an aggressive critic dangerous in DDPG.
            hp.CriticLR = 1e-3;
            hp.NoiseVariance = 0.2;          % exploration noise
            hp.TargetPolicyNoise = 0.1;      % smoothing noise added to target actions
            hp.TargetPolicyNoiseClip = 0.3;
            hp.PolicyUpdateFrequency = 2;    % delayed actor updates - the "D" in TD3

        case 'sac'
            % SAC tunes its own exploration through the entropy term, so there is no
            % noise schedule to hand-tune. Actor and critic rates are matched, as is
            % conventional for SAC.
            hp.ActorLR  = 3e-4;
            hp.CriticLR = 3e-4;
            % Target entropy defaults to -numel(action) inside the toolbox, which is the
            % standard heuristic and appropriate for a 2-DOF action space.
            hp.EntropyWeight = 1.0;
            hp.EntropyLearnRate = 3e-4;

        case 'ppo'
            % On-policy: no replay buffer, and it needs a much larger effective batch.
            % PPO tolerates a higher actor rate than the off-policy methods because its
            % clipped objective bounds how far the policy can move per update.
            hp.ActorLR  = 3e-4;
            hp.CriticLR = 1e-3;
            hp.ClipFactor = 0.2;
            hp.EntropyLossWeight = 0.01;     % keeps the Gaussian from collapsing early
            hp.ExperienceHorizon = 512;      % agent steps gathered before each update
            hp.NumEpoch = 3;

        otherwise
            error('load_hyperparams: unknown agent type ''%s''.', agent_type);
    end

    fprintf('Using default hyperparameters for %s.\n', upper(agent_type));
end


function [valid, reason] = check_provenance(hp, params)
% Validates that a saved hyperparameter set was tuned against the current harness.

    valid = false;

    if ~isfield(hp, 'tuned_for')
        reason = 'file predates provenance stamping (no tuned_for field)';
        return;
    end

    tf = hp.tuned_for;

    if ~isfield(tf, 'harness_version') || tf.harness_version ~= params.harness_version
        if isfield(tf, 'harness_version')
            found = tf.harness_version;
        else
            found = NaN;
        end
        reason = sprintf('harness version mismatch (tuned for %g, current is %g)', ...
            found, params.harness_version);
        return;
    end

    if ~isfield(tf, 'agent_dt') || abs(tf.agent_dt - params.agent_dt) > 1e-12
        if isfield(tf, 'agent_dt')
            found = tf.agent_dt;
        else
            found = NaN;
        end
        reason = sprintf('agent sample time mismatch (tuned at %g s, current is %g s)', ...
            found, params.agent_dt);
        return;
    end

    valid = true;
    reason = '';
end
