function [agent, info] = pretrain_actor_supervised(agent, demos, params, opts)
% PRETRAIN_ACTOR_SUPERVISED Fits a deterministic actor to demonstrations by regression.
%
% Plain supervised behaviour cloning: minimise ||actor(obs) - a_expert||^2 over the
% demonstration set, then install the fitted weights into the agent's actor.
%
% --- WHY THIS RATHER THAN THE BC REGULARISER ALONE ---
% rlBehaviorCloningRegularizerOptions adds a cloning term to TD3's actor loss, but that
% loss is dominated by the CRITIC gradient, and early in offline training the critic is
% uninformative. Measured on this harness: 40 epochs (20,000 updates) of trainFromData
% with BC weight 2.5 left the policy at 0% landings and 70% timeouts, while the critic's
% Q estimate was still climbing. The cloning signal was there but not in control.
%
% Direct regression removes the interaction entirely. It also answers a question the
% regulariser cannot: whether the network can REPRESENT the pilot at all. If this fails to
% clone, the problem is capacity or features, not the RL setup - worth knowing before
% spending more compute either way.
%
% --- DETERMINISTIC ACTORS ONLY ---
% Requires an actor whose output is already the action, i.e. TD3/DDPG (their network ends
% in tanh, spanning [-1,1] exactly). SAC's Gaussian actor leaves ActorMean UNSQUASHED and
% applies tanh internally, so its regression target would have to be atanh(a*); this
% function rejects it rather than fitting the wrong thing silently.
%
% Inputs:
%   agent  - rlTD3Agent or rlDDPGAgent
%   demos  - output of generate_demonstrations, or a path to demonstrations.mat
%   params - get_sim_params
%   opts   - (Optional) struct: .max_epochs (60), .mini_batch (256), .learn_rate (1e-3),
%            .val_frac (0.1), .verbose (true), .balance_phases (true)
%
% Outputs:
%   agent - same handle, actor replaced with the fitted network
%   info  - struct with .rmse_train, .rmse_val, .n_samples

    if nargin < 4, opts = struct(); end
    if ~isfield(opts,'max_epochs'), opts.max_epochs = 60;   end
    if ~isfield(opts,'mini_batch'), opts.mini_batch = 256;  end
    if ~isfield(opts,'learn_rate'), opts.learn_rate = 1e-3; end
    if ~isfield(opts,'val_frac'),   opts.val_frac   = 0.1;  end
    if ~isfield(opts,'verbose'),    opts.verbose    = true; end
    if ~isfield(opts,'balance_phases'), opts.balance_phases = true; end

    if ischar(demos) || isstring(demos)
        loaded = load(demos, 'demos');
        demos = loaded.demos;
    end

    actor = getActor(agent);
    if ~isa(actor, 'rl.function.rlContinuousDeterministicActor')
        error('pretrainActorSupervised:NotDeterministic', ...
            ['Supervised cloning here requires a deterministic actor (TD3/DDPG). This ' ...
             'agent has a %s. A Gaussian actor''s mean head is unsquashed, so its ' ...
             'regression target must be atanh(a*), not a*.'], class(actor));
    end

    % --- 1. Flatten to a regression problem ---
    % Observations are normalised here from the raw states in the dataset, matching what
    % the agent sees at run time.
    n_total = sum(arrayfun(@(e) size(e.actions, 2), demos.episodes));
    X = zeros(n_total, numel(get_ai_observation(demos.episodes(1).states(:,1), params)));
    T = zeros(n_total, 2);
    P = zeros(n_total, 1);          % phase of each sample, for balancing
    k = 0;
    for i = 1:numel(demos.episodes)
        e = demos.episodes(i);
        for t = 1:size(e.actions, 2)
            k = k + 1;
            X(k,:) = get_ai_observation(e.states(:,t), params)';
            T(k,:) = e.actions(:,t)';
            P(k)   = e.phase;
        end
    end

    % --- PHASE BALANCING ---
    % Episode LENGTH varies by two orders of magnitude across the curriculum, so equal
    % episode counts produce wildly unequal sample counts: measured on the four-phase set,
    % Phase 1 is 7.0% of transitions and Phase 4 is 40.8%. MSE weights every sample
    % equally, so the fit is dominated by long cruise and braking segments while the
    % terminal manoeuvre - the part the touchdown gate actually measures - is a rounding
    % error in the loss.
    %
    % That is the most likely explanation for the previous clone scoring 45% on Phase 1
    % against 100% and 86% on Phases 2 and 3: its WORST regime was the easiest one, which
    % only makes sense if it barely trained on it.
    %
    % Resample each phase to the mean count so every regime carries equal weight.
    if opts.balance_phases
        phases = unique(P(P > 0))';
        counts = arrayfun(@(q) nnz(P == q), phases);
        target = round(mean(counts));
        rng(11);
        idx = [];
        for q = phases
            pool = find(P == q);
            % With replacement only where a phase is short of target.
            take = pool(randi(numel(pool), target, 1));
            idx = [idx; take]; %#ok<AGROW>
        end
        X = X(idx, :);
        T = T(idx, :);
        n_total = numel(idx);
        if opts.verbose
            fprintf('  phase-balanced: %d samples, %d per phase (was %s)\n', ...
                n_total, target, mat2str(counts));
        end
    end

    % Held-out split. Without it there is no way to distinguish "cloned the pilot" from
    % "memorised the trajectories", and the two behave very differently off-distribution.
    rng(7);
    perm = randperm(n_total);
    n_val = max(1, round(opts.val_frac * n_total));
    val_idx = perm(1:n_val);
    trn_idx = perm(n_val+1:end);

    % --- 2. Fit ---
    net = getModel(actor);
    trainOpts = trainingOptions('adam', ...
        'MaxEpochs', opts.max_epochs, ...
        'MiniBatchSize', opts.mini_batch, ...
        'InitialLearnRate', opts.learn_rate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', max(1, floor(opts.max_epochs/3)), ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {X(val_idx,:), T(val_idx,:)}, ...
        'ValidationFrequency', 200, ...
        'Plots', 'none', ...
        'Verbose', opts.verbose, ...
        'VerboseFrequency', 200);

    net = trainnet(X(trn_idx,:), T(trn_idx,:), net, 'mse', trainOpts);

    % --- 3. Install ---
    % setModel requires the layer names to be unchanged, which holds because the network
    % came from this same actor.
    actor = setModel(actor, net);
    agent = setActor(agent, actor);

    % --- 4. Report fit quality in ACTION UNITS ---
    info = struct();
    info.n_samples  = n_total;
    info.rmse_train = action_rmse(net, X(trn_idx,:), T(trn_idx,:));
    info.rmse_val   = action_rmse(net, X(val_idx,:), T(val_idx,:));
    if opts.verbose
        fprintf(['Cloned actor on %d samples: RMSE train %.4f, val %.4f ' ...
                 '(action range is 2.0)\n'], n_total, info.rmse_train, info.rmse_val);
    end
end


function r = action_rmse(net, X, T)
    P = predict(net, X);
    r = sqrt(mean((P(:) - T(:)).^2));
end
