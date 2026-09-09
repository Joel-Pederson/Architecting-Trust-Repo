function tests = agent_weights_test
% AGENT_WEIGHTS_TEST The committed policy, and the portable export of it, stay in sync.
%
% cloned_agent_4phase.mat is committed because every published figure was measured on THAT
% network - retraining draws a different one - so it is evidence rather than a build output.
% Its portable companion exists because the .mat holds a serialised rlTD3Agent, which needs
% the Reinforcement Learning Toolbox to read and is coupled to the MATLAB release that
% wrote it.
%
% Both of those only mean anything if the export actually reproduces the agent. A weights
% file that had silently drifted from the network would still load, still contain plausible
% numbers, and be wrong in a way nothing else here would notice.
%
% Committing the agent also lets CI exercise the REAL policy for the first time; several
% other test files still fly the classical pilot or a freshly initialised network because
% they were written when no trained agent existed in a clone.

    tests = functiontests(localfunctions);
end


function setupOnce(testCase)
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(here));
    testCase.TestData.root = here;
    testCase.TestData.p    = get_sim_params();
end


function testBothArtefactsAreCommitted(testCase)
    root = testCase.TestData.root;
    verifyTrue(testCase, isfile(fullfile(root, 'cloned_agent_4phase.mat')), ...
        'The trained policy must be committed; the published figures are measured on it.');
    verifyTrue(testCase, isfile(fullfile(root, 'cloned_agent_4phase_weights.mat')), ...
        'The portable weights must be committed; regenerate with export_agent_weights.');
end


function testWeightsFileNeedsNoToolboxToRead(testCase)
% The entire point of the export. If a toolbox object leaks into the file, a reader without
% the Reinforcement Learning Toolbox - or without MATLAB at all, via scipy.io.loadmat -
% cannot open it, and the artefact stops being portable while still looking fine here.
    W = load(fullfile(testCase.TestData.root, 'cloned_agent_4phase_weights.mat'));

    verifyTrue(testCase, isfield(W, 'weights') && isfield(W, 'layers') && isfield(W, 'note'), ...
        'The export should carry weights, layer topology and a usage note.');

    names = fieldnames(W.weights);
    for k = 1:numel(names)
        v = W.weights.(names{k});
        verifyTrue(testCase, isnumeric(v) && isreal(v), sprintf( ...
            'weights.%s is %s; every entry must be a plain real numeric array.', ...
            names{k}, class(v)));
    end
end


function testExportedWeightsReproduceTheAgent(testCase)
% THE ASSERTION THIS FILE EXISTS FOR.
%
% Runs the forward pass from the exported arrays alone - no agent object, no toolbox
% inference - and compares against the real policy over a wide state sweep.
%
% Tolerance is 1e-4 on an action in [-1,1]. The network stores and computes in SINGLE, so
% an independent double-precision forward pass agrees to about 2e-6: that residual is
% accumulation order inside a 256-wide dot product, not a difference in the weights, which
% convert single-to-double losslessly. 1e-4 sits far above that and far below anything
% physically meaningful - one action unit is a full lunar g of net acceleration, so 1e-4 is
% under 2e-4 m/s^2.
    root = testCase.TestData.root;
    p = testCase.TestData.p;

    W = load(fullfile(root, 'cloned_agent_4phase_weights.mat'));
    d = load(fullfile(root, 'cloned_agent_4phase.mat'), 'agent');
    agent = d.agent;
    if isprop(agent, 'UseExplorationPolicy'), agent.UseExplorationPolicy = false; end

    rng(11);
    worst = 0;
    for k = 1:60
        % Deliberately spans the whole envelope, including past 90 degrees of tilt and
        % near-empty tanks, so the comparison is not confined to the nominal regime.
        x = [ (rand*2-1)*5e5; 10^(rand*4 + 0.3); (rand*2-1)*1700; (rand*2-1)*60; ...
              (rand*2-1)*pi;  (rand*2-1)*0.5;    rand*8200;        rand*300 ];
        obs = get_ai_observation(x, p);

        ref = getAction(agent, {obs});
        if iscell(ref), ref = ref{1}; end
        ref = reshape(double(ref), [], 1);

        worst = max(worst, max(abs(forward_from_weights(W.weights, obs(:)) - ref)));
    end

    verifyLessThan(testCase, worst, 1e-4, sprintf( ...
        ['The exported weights no longer reproduce the agent (worst difference %.3e). ' ...
         'Either the agent was retrained without re-running export_agent_weights, or the ' ...
         'network topology changed and forward_from_weights below is stale.'], worst));
end


function testTopologyMatchesTheDocumentedContrast(testCase)
% The paper's central table contrasts 448 lines of auditable C against the policy's
% parameter count. If the network is ever resized, that number moves and the claim goes
% stale, so it is asserted here rather than only written in prose.
    W = load(fullfile(testCase.TestData.root, 'cloned_agent_4phase_weights.mat'));
    total = 0;
    names = fieldnames(W.weights);
    for k = 1:numel(names)
        total = total + numel(W.weights.(names{k}));
    end
    verifyEqual(testCase, total, 69122, sprintf( ...
        ['The actor now has %d learnable parameters, not the 69,122 quoted in the README ' ...
         'and the flight-code contrast table. Update both, or the claim is stale.'], total));
end


% ---------- helpers ----------

function a = forward_from_weights(w, obs)
% The policy's forward pass, written from nothing but the exported arrays.
%
% Deliberately hand-written rather than driven by the layer table: this is the calculation a
% reader in another language would have to implement, so if it cannot be expressed in four
% lines here, the export is not as usable as it claims to be.
    h = max(0, w.ActorFC1_Weights * obs + w.ActorFC1_Bias);
    h = max(0, w.ActorFC2_Weights * h   + w.ActorFC2_Bias);
    a = tanh(w.ActionOutput_Weights * h + w.ActionOutput_Bias);
end
