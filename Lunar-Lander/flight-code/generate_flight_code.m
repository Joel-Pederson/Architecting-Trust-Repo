function info = generate_flight_code(opts)
% GENERATE_FLIGHT_CODE Emits the Safety Sidecar barrier as standalone portable C.
%
%   generate_flight_code()                       % regenerate flight-code/src
%   generate_flight_code(struct('verify', true)) % regenerate, then build and diff
%
% The barrier is the only part of this system that has to be trusted, so it is the only
% part that gets compiled to flight code. That asymmetry is the architecture's whole
% claim: a small deterministic component bounds a large learned one, and "small" should
% be a number a reader can check rather than an adjective.
%
% --- WHAT COMES OUT ---
% Nine files, no dynamic allocation, no IEEE special-case handling, and a single entry
% point whose signature carries no configuration at all:
%
%     void safety_sidecar_filter(const double x[8], const double u_nominal[2],
%                                double u_actual[2], boolean_T *VetoTriggered,
%                                double *h_alt, double *h_fuel);
%
% params is passed as coder.Constant, so every physical constant is compiled into the
% source as a literal. The generated function cannot be misconfigured at run time because
% there is nothing left to configure.
%
% --- WHY THE BARRIER CONTAINS NO Inf OR NaN ---
% It used to. Three sentinels (margin = +/-Inf, t_stop = Inf) forced
% cfg.SupportNonFinite, which makes Coder emit rt_nonfinite.c, rtGetInf.c, rtGetNaN.c and
% the IEEE special-case paths that go with them - 18 files instead of 9. Flight-software
% review generally objects to non-finite values in the first place, so the sentinels were
% replaced with finite equivalents and verified bit-identical across 200,000 randomised
% states spanning the regimes the infinities lived in. CI-tests/flight_code_test.m keeps
% them out.
%
% --- WHY rtwtypes.h IS REWRITTEN ---
% Coder's rtwtypes.h ends in #include "tmwtypes.h", a MathWorks header outside this repo,
% so this function substitutes a self-contained equivalent. Without it the artefact cannot
% be compiled by a reader who does not own MATLAB, which defeats the point of shipping it.
% Generation also asks for CBuiltIn types, so the code a reader audits says `double`
% rather than `real_T`.
%
% --- LICENSING, WORTH KNOWING BEFORE YOU PUBLISH ---
% Files generated under an academic MATLAB licence carry an "Academic License ... Not for
% government, commercial, or other organizational use" header. That header is preserved
% verbatim in the committed sources. It is appropriate for a paper artefact and it does
% constrain reuse; do not present the generated C as production-deployable code.
%
% Inputs (optional struct):
%   opts.verify  - after generating, build the harness and run the equivalence check
%                  against the committed fixtures (default false)
%   opts.outdir  - destination for the generated sources (default flight-code/src)
%
% Outputs:
%   info - struct with .files, .barrier_lines, .total_lines
%
% See also SAFETY_SIDECAR_FILTER, EXPORT_FLIGHT_FIXTURES, FLIGHT_CODE_TEST.

    if nargin < 1, opts = struct(); end
    if ~isfield(opts,'verify'), opts.verify = false; end

    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);
    addpath(genpath(root));

    if ~isfield(opts,'outdir'), opts.outdir = fullfile(here, 'src'); end

    p = get_sim_params();

    % Generate into a scratch folder first, so a failed run cannot leave the committed
    % sources half-written.
    scratch = fullfile(tempdir, sprintf('sidecar_cg_%s', datestr(now, 'yyyymmddHHMMSS'))); %#ok<TNOW1,DATST>
    cleanup = onCleanup(@() rmdir_quiet(scratch));

    cfg = coder.config('lib');
    cfg.GenCodeOnly         = true;
    cfg.GenerateReport      = false;
    cfg.TargetLang          = 'C';
    cfg.SupportNonFinite    = false;   % the barrier has no Inf/NaN; keep it that way
    cfg.GenerateExampleMain = 'DoNotGenerate';
    cfg.EnableOpenMP        = false;
    cfg.DataTypeReplacement = 'CBuiltIn';   % plain double/int in the artefact a reader audits

    fprintf('Generating C from core/safety_sidecar_filter.m ...\n');
    codegen('safety_sidecar_filter', '-config', cfg, ...
            '-args', {zeros(8,1), zeros(2,1), coder.Constant(p)}, ...
            '-d', scratch, '-o', 'sidecar');

    % --- install ---
    if isfolder(opts.outdir), rmdir(opts.outdir, 's'); end
    mkdir(opts.outdir);

    produced = [dir(fullfile(scratch,'*.c')); dir(fullfile(scratch,'*.h'))];
    for k = 1:numel(produced)
        copyfile(fullfile(scratch, produced(k).name), fullfile(opts.outdir, produced(k).name));
    end

    write_portable_rtwtypes(fullfile(opts.outdir, 'rtwtypes.h'));

    % --- report ---
    files = [dir(fullfile(opts.outdir,'*.c')); dir(fullfile(opts.outdir,'*.h'))];
    total = 0;
    for k = 1:numel(files)
        total = total + count_lines(fullfile(opts.outdir, files(k).name));
    end
    barrier = count_lines(fullfile(opts.outdir, 'safety_sidecar_filter.c'));

    info = struct('files', numel(files), 'barrier_lines', barrier, 'total_lines', total);
    fprintf('\n  %d files, %d lines total, %d in the barrier itself\n', ...
        info.files, info.total_lines, info.barrier_lines);
    fprintf('  installed to %s\n', opts.outdir);

    assert_no_tmwtypes(opts.outdir);

    if opts.verify
        fprintf('\nBuilding harness and checking equivalence ...\n');
        [status, out] = system(sprintf('cd "%s" && make --no-print-directory check', ...
                                       fullfile(here, 'harness')));
        fprintf('%s\n', out);
        if status ~= 0
            error('generateFlightCode:EquivalenceFailed', ...
                  'The generated C did not reproduce the MATLAB reference.');
        end
    end
end


function write_portable_rtwtypes(path)
% Self-contained replacement for Coder's rtwtypes.h, which ends in an #include of the
% MathWorks header tmwtypes.h. The generated barrier uses exactly three typedefs; this
% supplies them from <stdint.h> so the artefact builds with nothing but a C99 compiler.
    txt = [ ...
"/*", newline, ...
" * rtwtypes.h - portable replacement, written by flight-code/generate_flight_code.m.", newline, ...
" *", newline, ...
" * MATLAB Coder's own rtwtypes.h ends in #include ""tmwtypes.h"", a MathWorks header", newline, ...
" * that lives outside this repository. Generation uses plain C built-in types, so the", newline, ...
" * barrier itself needs only boolean_T; the other two are defined for completeness. That", newline, ...
" * lets a reader with a C compiler and no MATLAB licence build and audit this code.", newline, ...
" *", newline, ...
" * Regenerate with: generate_flight_code", newline, ...
" */", newline, newline, ...
"#ifndef RTWTYPES_H", newline, ...
"#define RTWTYPES_H", newline, newline, ...
"#include <stdint.h>", newline, newline, ...
"typedef double        real_T;", newline, ...
"typedef unsigned char boolean_T;", newline, ...
"typedef int32_t       int32_T;", newline, newline, ...
"#ifndef true", newline, ...
"#define true  1", newline, ...
"#endif", newline, ...
"#ifndef false", newline, ...
"#define false 0", newline, ...
"#endif", newline, newline, ...
"#endif /* RTWTYPES_H */", newline ];
    fid = fopen(path, 'w');
    assert(fid > 0, 'Could not write %s', path);
    fwrite(fid, strjoin(txt, ''));
    fclose(fid);
end


function assert_no_tmwtypes(dirpath)
% The whole point of the artefact is that it builds without MATLAB. Catch a regression
% here rather than in someone else's build.
%
% Matches the #include DIRECTIVE rather than the bare word: the replacement header this
% function writes explains itself, and that explanation naturally names the header it
% replaced. A substring search flags its own documentation.
    pattern = '^\s*#\s*include\s*[""<]tmwtypes\.h["">]';
    files = [dir(fullfile(dirpath,'*.c')); dir(fullfile(dirpath,'*.h'))];
    for k = 1:numel(files)
        lines = strsplit(fileread(fullfile(dirpath, files(k).name)), newline);
        if any(~cellfun(@isempty, regexp(lines, pattern, 'once')))
            error('generateFlightCode:MatlabHeaderLeaked', ...
                ['%s still includes tmwtypes.h, so the generated code cannot be built ' ...
                 'without a MATLAB installation.'], files(k).name);
        end
    end
end


function n = count_lines(path)
    n = numel(strsplit(fileread(path), newline));
end


function rmdir_quiet(d)
    if isfolder(d)
        try, rmdir(d, 's'); catch, end %#ok<CTCH,NOCOM>
    end
end
