function stats = phase_landing_rates(agent, params, opts)
% PHASE_LANDING_RATES Flies a policy phase by phase and reports how often it lands.
%
%   stats = phase_landing_rates(agent, p)
%   stats = phase_landing_rates(agent, p, struct('n_episodes', 30, 'guardian', 'off'))
%
% The single measurement this project selects on. train_pipeline, dagger_refine and
% evaluate_final_agent each carried a near-copy, which is how one experiment ends up
% quoted at two sample sizes in two places.
%
% Always PER PHASE, never pooled: a Phase 4 descent runs ~8,900 agent steps against Phase
% 1's ~450, so pooling hides which regime works. The seed is applied per phase so two arms
% of a comparison face identical initial conditions - what makes "only the barrier
% differs" true.
%
% Inputs:
%   agent  - anything rollout_episode accepts (a trained agent)
%   params - get_sim_params
%   opts   - (Optional) struct:
%              .n_episodes episodes per phase (default 8, a screening budget)
%              .seed       RNG seed applied per phase (default 42)
%              .guardian   'on' (default) or 'off'
%              .phases     phases to measure (default all configured)
%
% Outputs:
%   stats - struct with, one element per requested phase:
%             .phases  the phase indices measured
%             .rate    landing rate in [0,1]
%             .impact  mean touchdown speed of episodes that reached the ground
%             .worst   worst touchdown speed of those episodes
%
% Touchdown speed is Euclidean, hypot(dx, dy). Episodes that time out or leave the box
% never touch down, so they contribute to .rate but not to .impact or .worst - counting
% them as zero impact would flatter a policy that never arrives at all.
%
% See also ROLLOUT_EPISODE, SELECT_PHASE, EVALUATE_FINAL_AGENT, TRAIN_PIPELINE.

    if nargin < 3, opts = struct(); end
    if ~isfield(opts,'n_episodes'), opts.n_episodes = 8;    end
    if ~isfield(opts,'seed'),       opts.seed       = 42;   end
    if ~isfield(opts,'guardian'),   opts.guardian   = 'on'; end
    if ~isfield(opts,'phases'),     opts.phases     = 1:numel(params.phase_max_steps); end

    if isprop(agent, 'UseExplorationPolicy'), agent.UseExplorationPolicy = false; end

    n = numel(opts.phases);
    stats = struct('phases', opts.phases, ...
                   'rate',   zeros(1, n), ...
                   'impact', nan(1, n), ...
                   'worst',  nan(1, n));

    for k = 1:n
        env = LunarLanderEnv('DenseBaseline', opts.guardian);
        select_phase(env, opts.phases(k));

        rng(opts.seed);
        landed = 0;
        impacts = [];
        for e = 1:opts.n_episodes
            % Roll to the LONGEST configured budget, not params.max_agent_steps: that
            % default truncates a Phase 4 descent partway down and scores it as a timeout
            % that never happened.
            ep = rollout_episode(env, agent, max(params.phase_max_steps));
            landed = landed + strcmp(ep.outcome, 'landed');
            if any(strcmp(ep.outcome, {'landed', 'crashed'}))
                impacts(end+1) = hypot(ep.touchdown_dx, ep.touchdown_dy); %#ok<AGROW>
            end
        end

        stats.rate(k) = landed / opts.n_episodes;
        if ~isempty(impacts)
            stats.impact(k) = mean(impacts);
            stats.worst(k)  = max(impacts);
        end
    end
end
