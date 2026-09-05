function agent = seed_agent_from_demonstrations(agent, demos, params, opts)
% SEED_AGENT_FROM_DEMONSTRATIONS Pre-fills an off-policy agent's replay buffer.
%
% Converts recorded expert trajectories into the toolbox's experience format and installs
% them as the agent's replay memory, so learning starts from a buffer that already
% contains successful landings instead of one that may never see a single success.
%
% --- WHY OFF-POLICY ONLY ---
% trainFromData and buffer seeding accept DDPG / TD3 / SAC / DQN / MBPO. PPO and the other
% on-policy agents are rejected by the toolbox: they discard experience after each update
% by construction, so a seeded buffer would be meaningless to them.
%
% --- THE EXPERIENCE FORMAT ---
% rlReplayMemory.validateExperience hard-requires Observation, Action and NextObservation
% to be CELL arrays, one cell per observation channel - `{obs}`, not `obs`. This is easy
% to get wrong: MathWorks' own Example 5 in trainFromData.m omits the braces and would
% error if run. Experiences must also be passed as a COLUMN struct array; a struct array
% built in a loop is a row.
%
% Inputs:
%   agent  - an off-policy agent (rlTD3Agent / rlSACAgent / rlDDPGAgent)
%   demos  - output of generate_demonstrations, or a path to demonstrations.mat
%   params - get_sim_params
%   opts   - (Optional) struct:
%              .bc_weight  behaviour-cloning regulariser weight (default 2.5). Set to 0
%                          to seed the buffer without adding the BC term to the actor loss.
%              .verbose    default true
%
% Outputs:
%   agent - the same handle, with ExperienceBuffer populated and, unless bc_weight is 0,
%           BatchDataRegularizerOptions set.
%
% NOTE: RL agents are HANDLE objects. This mutates the agent passed in and returns the
% same handle. Callers running more than one experimental arm must copy() first.

    if nargin < 4, opts = struct(); end
    if ~isfield(opts,'bc_weight'), opts.bc_weight = 2.5;  end
    if ~isfield(opts,'verbose'),   opts.verbose   = true; end

    if ischar(demos) || isstring(demos)
        loaded = load(demos, 'demos');
        demos = loaded.demos;
    end

    obsInfo = getObservationInfo(agent);
    actInfo = getActionInfo(agent);

    % --- 1. Flatten trajectories into transitions ---
    % Observations are normalised HERE, from the raw physics states in the dataset.
    % generate_demonstrations deliberately stores raw states because
    % get_ai_observation's normalisation constants are unowned literals; normalising at
    % load keeps the dataset valid if they ever change.
    n_total = sum(arrayfun(@(e) size(e.actions, 2), demos.episodes));
    exp_struct = struct('Observation', cell(n_total,1), 'Action', cell(n_total,1), ...
                        'Reward', cell(n_total,1), 'NextObservation', cell(n_total,1), ...
                        'IsDone', cell(n_total,1));

    k = 0;
    for i = 1:numel(demos.episodes)
        e = demos.episodes(i);
        n = size(e.actions, 2);
        for t = 1:n
            k = k + 1;
            exp_struct(k).Observation     = {get_ai_observation(e.states(:,t),   params)};
            exp_struct(k).Action          = {e.actions(:,t)};
            exp_struct(k).Reward          = e.rewards(t);
            exp_struct(k).NextObservation = {get_ai_observation(e.states(:,t+1), params)};
            % Terminal only on the true final transition of a landed episode. Marking
            % every episode end as terminal would be right here (all kept episodes land),
            % but stating the condition explicitly keeps it correct if the filter changes.
            exp_struct(k).IsDone = double(t == n && ~strcmp(e.outcome, 'timeout'));
        end
    end

    % --- 2. Install as the agent's replay memory ---
    buffer_len = max(n_total, agent.AgentOptions.ExperienceBufferLength);
    buf = rlReplayMemory(obsInfo, actInfo, buffer_len);
    append(buf, exp_struct(:));           % column, per the toolbox's (:,1) signature
    agent.ExperienceBuffer = buf;         % public settable, validated against the specs

    % --- 3. Behaviour-cloning regulariser ---
    % Adds a term to the actor loss pulling it toward the demonstrated actions. Without
    % it, seeding only helps the critic: the actor is still free to walk away from the
    % demonstrated policy as soon as the critic's estimates drift.
    if opts.bc_weight > 0
        agent.AgentOptions.BatchDataRegularizerOptions = ...
            rlBehaviorCloningRegularizerOptions( ...
                'BehaviorCloningRegularizerWeight', opts.bc_weight);
    end

    if opts.verbose
        fprintf(['Seeded %d transitions from %d demonstration episodes ' ...
                 '(buffer %d, BC weight %.2f)\n'], ...
                n_total, numel(demos.episodes), buffer_len, opts.bc_weight);
        if agent.AgentOptions.ResetExperienceBufferBeforeTraining
            warning('seedAgent:BufferWillBeCleared', ...
                ['ResetExperienceBufferBeforeTraining is TRUE. train() will discard ' ...
                 'every one of these %d transitions before its first update.'], n_total);
        end
    end
end
