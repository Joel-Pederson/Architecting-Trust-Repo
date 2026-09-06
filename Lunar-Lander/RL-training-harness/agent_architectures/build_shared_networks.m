function nets = build_shared_networks(obsInfo, actInfo, hidden_size)
% BUILD_SHARED_NETWORKS Common actor/critic topologies for every agent architecture.
%
% Every builder in agent_architectures/ draws its networks from here so the trade study
% compares ALGORITHMS rather than accidental differences in network width or depth. If
% DDPG used a 256x256 trunk and SAC a 400x300 one, any performance gap would be
% uninterpretable - which is the usual way algorithm comparisons go wrong.
%
% All networks are returned as dlnetwork objects, the modern representation. The
% layerGraph / rlQValueRepresentation / rlDeterministicActorRepresentation API this
% replaced was deprecated in R2022a and is not reliable on current MATLAB releases.
%
% Inputs:
%   obsInfo - observation specification from the environment
%   actInfo - action specification from the environment
%
% Outputs:
%   nets - struct of dlnetwork objects:
%     .qCritic          Q(s,a) -> scalar. Used by DDPG and TD3 (TD3 needs two copies).
%     .qCritic2         An independently initialised twin of qCritic, for TD3/SAC.
%     .vCritic          V(s) -> scalar. Used by PPO.
%     .detActor         s -> tanh-bounded action. Used by DDPG and TD3.
%     .gaussActor       s -> (mean, log std). Used by SAC and PPO.
%
% Note on the deterministic actor: the tanh output layer bounds actions to [-1, 1], and
% the environment maps that range onto physical thrust and torque limits. The Gaussian
% actors deliberately do NOT tanh their mean - SAC applies its own bounded transform, and
% PPO relies on the action spec - so squashing here would double-bound them.

    nObs = obsInfo.Dimension(1);
    nAct = actInfo.Dimension(1);

    % Capacity is now an argument. The four-phase curriculum spans a 50 m touchdown and a
    % 550 km powered descent, and at 256 units different random initialisations
    % specialised in different regimes - one clone scoring 100/100/75/12 across the phases
    % and another 50/0/0/38. That split across inits is the signature of a network being
    % asked to hold more regimes than it comfortably fits.
    if nargin < 3 || isempty(hidden_size)
        hidden_size = 256;      % unchanged default: existing callers are unaffected
    end
    hidden = hidden_size;   % Width shared by every architecture in the trade study

    % --- Q-VALUE CRITIC: Q(s, a) -> scalar ---
    nets.qCritic  = make_q_critic(nObs, nAct, hidden);
    nets.qCritic2 = make_q_critic(nObs, nAct, hidden);  % Independent init: twins must differ

    % --- VALUE CRITIC: V(s) -> scalar (PPO) ---
    vNet = [
        featureInputLayer(nObs, 'Name', 'State')
        fullyConnectedLayer(hidden, 'Name', 'VFC1')
        reluLayer('Name', 'VRelu1')
        fullyConnectedLayer(hidden, 'Name', 'VFC2')
        reluLayer('Name', 'VRelu2')
        fullyConnectedLayer(1, 'Name', 'Value')];
    nets.vCritic = initialize(dlnetwork(vNet));

    % --- DETERMINISTIC ACTOR: s -> action (DDPG, TD3) ---
    aNet = [
        featureInputLayer(nObs, 'Name', 'State')
        fullyConnectedLayer(hidden, 'Name', 'ActorFC1')
        reluLayer('Name', 'ActorRelu1')
        fullyConnectedLayer(hidden, 'Name', 'ActorFC2')
        reluLayer('Name', 'ActorRelu2')
        fullyConnectedLayer(nAct, 'Name', 'ActionOutput')
        tanhLayer('Name', 'ActionTanh')];   % Squashes outputs to exactly [-1, 1]
    nets.detActor = initialize(dlnetwork(aNet));

    % --- GAUSSIAN ACTOR: s -> (mean, log std) (SAC, PPO) ---
    % A shared trunk splits into two heads. The standard-deviation head uses softplus to
    % keep it strictly positive without the numerical cliff of an exponential.
    trunk = [
        featureInputLayer(nObs, 'Name', 'State')
        fullyConnectedLayer(hidden, 'Name', 'GActorFC1')
        reluLayer('Name', 'GActorRelu1')
        fullyConnectedLayer(hidden, 'Name', 'GActorFC2')
        reluLayer('Name', 'GActorRelu2')];

    meanHead = fullyConnectedLayer(nAct, 'Name', 'ActorMean');

    stdHead = [
        fullyConnectedLayer(nAct, 'Name', 'ActorStdFC')
        softplusLayer('Name', 'ActorStd')];

    gNet = dlnetwork();
    gNet = addLayers(gNet, trunk);
    gNet = addLayers(gNet, meanHead);
    gNet = addLayers(gNet, stdHead);
    gNet = connectLayers(gNet, 'GActorRelu2', 'ActorMean');
    gNet = connectLayers(gNet, 'GActorRelu2', 'ActorStdFC');
    nets.gaussActor = initialize(gNet);

    nets.hidden_size = hidden;
end


function net = make_q_critic(nObs, nAct, hidden)
% Q(s,a) -> scalar. Two input paths (state, action) merged by addition, the standard
% DDPG/TD3 critic topology. Called twice for the twin-critic architectures, which relies
% on each call producing an INDEPENDENT random initialisation - handing TD3 or SAC two
% copies of identical weights makes min(Q1,Q2) a no-op and silently degrades them to DDPG.

    statePath = [
        featureInputLayer(nObs, 'Name', 'State')
        fullyConnectedLayer(hidden, 'Name', 'CriticStateFC1')
        reluLayer('Name', 'CriticRelu1')
        fullyConnectedLayer(hidden, 'Name', 'CriticStateFC2')];

    actionPath = [
        featureInputLayer(nAct, 'Name', 'Action')
        fullyConnectedLayer(hidden, 'Name', 'CriticActionFC1')];

    commonPath = [
        additionLayer(2, 'Name', 'add')
        reluLayer('Name', 'CriticCommonRelu')
        fullyConnectedLayer(1, 'Name', 'QValue')];

    net = dlnetwork();
    net = addLayers(net, statePath);
    net = addLayers(net, actionPath);
    net = addLayers(net, commonPath);
    net = connectLayers(net, 'CriticStateFC2', 'add/in1');
    net = connectLayers(net, 'CriticActionFC1', 'add/in2');
    net = initialize(net);
end
