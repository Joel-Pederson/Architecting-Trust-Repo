function info = export_agent_weights(opts)
% EXPORT_AGENT_WEIGHTS Writes the policy's weights in a form that outlives MATLAB.
%
%   export_agent_weights()
%   export_agent_weights(struct('agent_file','my_agent.mat'))
%
% --- WHY THIS EXISTS ALONGSIDE THE AGENT ITSELF ---
% cloned_agent_4phase.mat is committed so the published figures can be re-measured against
% the exact network that produced them. But it holds a SERIALISED rlTD3Agent: reading it
% needs the Reinforcement Learning Toolbox, and it is coupled to the MATLAB release that
% wrote it. As a long-term artefact that is fragile in a way the generated C is not.
%
% This writes the same policy as plain numeric arrays - weights, biases, layer order - in a
% v7 MAT-file, which loads with no toolbox at all and which scipy.io.loadmat reads directly.
% Anyone can then reimplement the forward pass in any language and check the numbers.
%
% That symmetry is the point. The paper's argument is a small verifiable component bounding
% a large unverifiable one, and both halves are now inspectable in portable form: 448 lines
% of C on one side, and on the other, every weight as a number a reader can look at.
%
% --- WHAT IS DELIBERATELY NOT IN HERE ---
% The observation normalisation and the action mapping are NOT copied into this file. They
% live in core/get_ai_observation.m and core/action_to_command.m, and duplicating them here
% would create exactly the failure this codebase has produced six times: a fact written in
% two places, where one copy goes stale. The file carries a note naming them instead.
%
% Inputs (optional struct):
%   opts.agent_file - default 'cloned_agent_4phase.mat'
%   opts.out_file   - default 'cloned_agent_4phase_weights.mat'
%
% Outputs:
%   info - struct with .n_parameters, .n_layers, .bytes
%
% See also GENERATE_FLIGHT_CODE, GET_AI_OBSERVATION, ACTION_TO_COMMAND.

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'agent_file'), opts.agent_file = 'cloned_agent_4phase.mat'; end
    if ~isfield(opts,'out_file'),   opts.out_file   = 'cloned_agent_4phase_weights.mat'; end

    here = fileparts(mfilename('fullpath'));
    addpath(genpath(here));

    src = opts.agent_file;
    if ~isfile(src), src = fullfile(here, opts.agent_file); end
    if ~isfile(src)
        error('exportAgentWeights:NoAgent', ...
            'Agent file not found: %s. Build one with train_pipeline().', opts.agent_file);
    end

    loaded = load(src, 'agent');
    net = getModel(getActor(loaded.agent));

    % --- topology, in execution order ---
    layers = struct('name', {}, 'type', {}, 'size', {});
    for k = 1:numel(net.Layers)
        L = net.Layers(k);
        entry = struct('name', char(L.Name), 'type', class(L), 'size', []);
        if isprop(L, 'OutputSize'), entry.size = double(L.OutputSize); end
        layers(end+1) = entry; %#ok<AGROW>
    end

    % --- learnables as plain arrays ---
    % Extracted from the dlnetwork's Learnables table and stored under sanitised field
    % names, so the result is a struct of doubles with no toolbox objects anywhere in it.
    lp = net.Learnables;
    weights = struct();
    n_params = 0;
    for k = 1:height(lp)
        field = matlab.lang.makeValidName(sprintf('%s_%s', ...
            char(lp.Layer(k)), char(lp.Parameter(k))));
        v = double(extractdata(lp.Value{k}));
        weights.(field) = v;
        n_params = n_params + numel(v);
    end

    note = [ ...
        'Actor weights for the Architecting Trust lunar lander policy. Plain arrays: no ' ...
        'MATLAB toolbox is needed to read this file, and scipy.io.loadmat reads it directly. ' ...
        'The forward pass maps a 10-element observation to a 2-element action in [-1,1]. ' ...
        'The observation is built by core/get_ai_observation.m (signed-log normalisation, ' ...
        'including the two barrier margins h_alt and h_fuel) and the action is mapped to ' ...
        'physical thrust and torque by core/action_to_command.m. Those two files are the ' ...
        'authoritative definition of the interface and are deliberately not duplicated here.'];

    out = opts.out_file;
    if ~isfile(out), out = fullfile(here, opts.out_file); end

    % -v7, not -v7.3. The HDF5 format needs a reader; v7 is what scipy.io.loadmat expects,
    % and portability is the entire reason this file exists.
    save(out, 'weights', 'layers', 'note', '-v7');

    d = dir(out);
    info = struct('n_parameters', n_params, 'n_layers', numel(layers), 'bytes', d.bytes);
    fprintf('Wrote %s\n  %d learnable parameters across %d layers, %.0f KB\n', ...
        out, info.n_parameters, info.n_layers, info.bytes/1024);
    for k = 1:numel(layers)
        if isempty(layers(k).size), sz = '-'; else, sz = mat2str(layers(k).size); end
        fprintf('    %-20s %-26s %s\n', layers(k).name, layers(k).type, sz);
    end
end
