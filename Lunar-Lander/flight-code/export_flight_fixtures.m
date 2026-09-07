function info = export_flight_fixtures(opts)
% EXPORT_FLIGHT_FIXTURES Records the episode the equivalence check runs against.
%
%   export_flight_fixtures()
%
% Writes two CSVs into flight-code/fixtures:
%
%   telemetry.csv        one row per control step: the 8 true states, then the 2-element
%                        command the (faulty) pilot requested at that state
%   matlab_reference.csv what core/safety_sidecar_filter.m did with each of those rows:
%                        the 2-element command it allowed, the veto flag, h_alt, h_fuel
%
% The generated C is then fed telemetry.csv and diffed against matlab_reference.csv by
% flight-code/harness. That is the claim the artefact rests on: the C is not merely
% compilable, it makes the same decisions.
%
% --- WHY THIS EPISODE ---
% The 10 m altimeter bias on the classical pilot, seed 101 - the same run behind
% demo_sidecar_rescue and the fault-injection study. It is chosen because the barrier
% ACTUALLY FIRES here. An equivalence check flown on a healthy controller would agree
% perfectly while exercising none of the code that matters: the barrier stays inactive,
% so every row would compare "pass-through equals pass-through".
%
% --- WHY %.17g ---
% Full round-trip precision for a double. At the default 15 significant digits the states
% themselves change slightly on the way through the file, and the resulting C-vs-MATLAB
% residual (~1e-10) is that text conversion rather than anything the compiler did. Writing
% all 17 digits removes the confound, so any remaining difference is genuinely attributable
% to floating-point contraction in the generated code.
%
% Inputs (optional struct):
%   opts.steps    - maximum control steps to record (default 400)
%   opts.alt_bias - altimeter over-read in metres (default 10)
%   opts.seed     - default 101, matching demo_sidecar_rescue
%
% Outputs:
%   info - struct with .steps and .vetoes
%
% See also GENERATE_FLIGHT_CODE, DEMO_SIDECAR_RESCUE.

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'steps'),    opts.steps    = 400; end
    if ~isfield(opts,'alt_bias'), opts.alt_bias = 10;  end
    if ~isfield(opts,'seed'),     opts.seed     = 101; end

    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);
    addpath(genpath(root));
    p = get_sim_params();

    fixtures = fullfile(here, 'fixtures');
    if ~isfolder(fixtures), mkdir(fixtures); end

    env = LunarLanderEnv('DenseBaseline', 'on');
    env.CurriculumWeights = [1 0 0];
    rng(opts.seed);
    reset(env);

    X = zeros(8, opts.steps);
    U = zeros(2, opts.steps);
    R = zeros(5, opts.steps);
    n = 0;

    for i = 1:opts.steps
        s_true = env.State;

        % The pilot reads a corrupted altimeter; the barrier reads truth. That split is
        % the Perception Gatekeeper boundary, and it is what makes the barrier fire.
        s_seen = apply_sensor_fault(s_true, 'alt_bias', opts.alt_bias);
        u = scripted_pilot(s_seen, p);

        [u_allowed, veto, h_alt, h_fuel] = safety_sidecar_filter(s_true, u, p);

        n = i;
        X(:,i) = s_true;
        U(:,i) = u;
        R(:,i) = [u_allowed(:); double(veto); h_alt; h_fuel];

        [~, ~, done] = step(env, command_to_action(u, s_true, p));
        if done, break; end
    end

    write_csv(fullfile(fixtures, 'telemetry.csv'),        [X(:,1:n); U(:,1:n)]);
    write_csv(fullfile(fixtures, 'matlab_reference.csv'), R(:,1:n));

    info = struct('steps', n, 'vetoes', nnz(R(3,1:n)));
    fprintf('Wrote %d steps to flight-code/fixtures (%d vetoed by the barrier).\n', ...
        info.steps, info.vetoes);
    if info.vetoes == 0
        warning('exportFlightFixtures:NoVetoes', ...
            ['The barrier never fired in this episode, so the equivalence check would ' ...
             'exercise only the pass-through path. Raise opts.alt_bias.']);
    end
end


function write_csv(path, M)
% One row per control step, full double precision.
    fid = fopen(path, 'w');
    assert(fid > 0, 'Could not write %s', path);
    fmt = [repmat('%.17g,', 1, size(M,1) - 1), '%.17g\n'];
    fprintf(fid, fmt, M);
    fclose(fid);
end
