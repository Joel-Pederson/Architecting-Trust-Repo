function agent = build_agent(agent_type, obsInfo, actInfo, dt, hyperparams)
% BUILD_AGENT Dispatcher for the modular agent architectures.
%
% Single entry point so callers never need to know which builder exists. Adding a new
% architecture is one new build_<name>_agent.m file plus one case here - nothing in
% train_rl_agent, tune_hyperparameters, or the study runners has to change.
%
% Inputs:
%   agent_type  - 'ddpg' | 'td3' | 'sac' | 'ppo'
%   obsInfo     - observation specification from the environment
%   actInfo     - action specification from the environment
%   dt          - AGENT sample time (params.agent_dt = 0.1 s), NOT params.dt.
%                 Passing the physics step here collapses the discount horizon by 5x and
%                 is the single easiest way to silently break training.
%   hyperparams - (Optional) struct; each builder falls back to load_hyperparams
%
% Outputs:
%   agent - configured RL agent

    if nargin < 5
        hyperparams = [];
    end

    switch lower(agent_type)
        case 'ddpg'
            agent = build_ddpg_agent(obsInfo, actInfo, dt, hyperparams);
        case 'td3'
            agent = build_td3_agent(obsInfo, actInfo, dt, hyperparams);
        case 'sac'
            agent = build_sac_agent(obsInfo, actInfo, dt, hyperparams);
        case 'ppo'
            agent = build_ppo_agent(obsInfo, actInfo, dt, hyperparams);
        otherwise
            error('buildAgent:UnknownAgentType', ...
                'build_agent: unknown agent type ''%s''. Supported: ddpg, td3, sac, ppo.', ...
                agent_type);
    end
end
