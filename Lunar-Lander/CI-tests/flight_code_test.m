function tests = flight_code_test
% FLIGHT_CODE_TEST Guards the properties that make the generated C worth shipping.
%
% The committed artefact in flight-code/src is only meaningful if three things stay true:
% it builds without MATLAB, it makes the same decisions as the MATLAB barrier, and the
% barrier stays free of the constructs that would drag IEEE special-case handling back in.
%
% None of these are checked by regenerating the code - CI has no MATLAB Coder licence, and
% regenerating would test the generator rather than the artefact anyone actually reads.
% These tests run against the committed sources instead, which is what a reader gets.

    tests = functiontests(localfunctions);
end


function setupOnce(testCase)
    here = fileparts(fileparts(mfilename('fullpath')));   % Lunar-Lander/
    addpath(genpath(here));
    testCase.TestData.root    = here;
    testCase.TestData.src     = fullfile(here, 'flight-code', 'src');
    testCase.TestData.harness = fullfile(here, 'flight-code', 'harness');
end


function testGeneratedSourcesAreCommitted(testCase)
% A fresh clone must carry the C, because the whole point is that it can be read and built
% by someone who does not own MATLAB.
    required = { 'safety_sidecar_filter.c', 'safety_sidecar_filter.h', ...
                 'safety_sidecar_filter_initialize.c', 'safety_sidecar_filter_terminate.c', ...
                 'rtwtypes.h' };
    for k = 1:numel(required)
        f = fullfile(testCase.TestData.src, required{k});
        verifyTrue(testCase, isfile(f), ...
            sprintf('flight-code/src/%s is missing; run generate_flight_code.', required{k}));
    end
end


function testEntryPointCarriesNoConfiguration(testCase)
% params is passed to the code generator as a constant, so every physical value is
% compiled in as a literal. If the signature grows a params argument back, the artefact
% has become misconfigurable at run time and the claim in the README is no longer true.
    hdr = fileread(fullfile(testCase.TestData.src, 'safety_sidecar_filter.h'));
    verifySubstring(testCase, hdr, 'const double x[8]', ...
        'The generated entry point should take the 8-element state directly.');
    verifySubstring(testCase, hdr, 'const double u_nominal[2]', ...
        'The generated entry point should take the 2-element nominal command.');
    verifyFalse(testCase, contains(hdr, 'params'), ...
        ['The generated barrier should carry no configuration argument - params is ' ...
         'compiled in as constants via coder.Constant.']);
end


function testArtefactBuildsWithoutMatlabHeaders(testCase)
% Coder's own rtwtypes.h ends in #include "tmwtypes.h", which lives inside a MATLAB
% installation. generate_flight_code substitutes a self-contained replacement; if that
% substitution regresses, the artefact silently stops building for everyone else.
    files = [dir(fullfile(testCase.TestData.src,'*.c')); ...
             dir(fullfile(testCase.TestData.src,'*.h'))];
    verifyNotEmpty(testCase, files, 'No generated sources found.');

    % Match the include DIRECTIVE, not the bare word: the replacement header explains what
    % it replaced, and a substring search would flag its own documentation.
    pattern = '^\s*#\s*include\s*["<]tmwtypes\.h[">]';
    for k = 1:numel(files)
        lines = strsplit(fileread(fullfile(testCase.TestData.src, files(k).name)), newline);
        hit = any(~cellfun(@isempty, regexp(lines, pattern, 'once')));
        verifyFalse(testCase, hit, sprintf( ...
            '%s includes tmwtypes.h, so the artefact needs a MATLAB installation to build.', ...
            files(k).name));
    end
end


function testBarrierStaysFreeOfNonFiniteValues(testCase)
% THE REGRESSION THIS FILE EXISTS FOR.
%
% The barrier used Inf as a sentinel in three places. That forces cfg.SupportNonFinite,
% which makes MATLAB Coder emit rt_nonfinite.c, rtGetInf.c and rtGetNaN.c and the IEEE
% special-case paths with them - 18 files instead of 9 - and flight-software review
% generally objects to non-finite values in the first place. Reintroducing one would not
% break any behavioural test; it would quietly double the artefact.
    src = fileread(fullfile(testCase.TestData.root, 'core', 'safety_sidecar_filter.m'));
    lines = strsplit(src, newline);

    % Code only: a comment may legitimately discuss Inf, and this file's header does.
    code = regexprep(lines, '%.*$', '');
    offending = find(~cellfun(@isempty, regexp(code, '(^|[^\w.])(Inf|NaN|inf|nan)\s*[;,)\]]', 'once')));

    verifyEmpty(testCase, offending, sprintf( ...
        ['safety_sidecar_filter.m assigns a non-finite value on line(s) %s. Use a finite ' ...
         'sentinel where the value is only compared, or restructure where it feeds ' ...
         'arithmetic - realmax multiplied by anything overflows back to Inf.'], ...
        mat2str(offending)));
end


function testGeneratedCodeAllocatesNothing(testCase)
% Static footprint is half the claim in the README table. A malloc appearing here would
% mean the barrier can fail at run time for reasons unrelated to flight.
    src = fileread(fullfile(testCase.TestData.src, 'safety_sidecar_filter.c'));
    for token = {'malloc', 'calloc', 'realloc', 'free('}
        verifyFalse(testCase, contains(src, token{1}), sprintf( ...
            'The generated barrier calls %s; it is documented as allocation-free.', token{1}));
    end
end


function testGeneratedCodeReproducesTheMatlabBarrier(testCase)
% The claim the artefact rests on. Builds the committed C with the system compiler and
% diffs it against the recorded MATLAB outputs.
%
% Skipped rather than failed where no compiler exists, because a missing toolchain is an
% environment fact, not a defect in this repository.
    harness = testCase.TestData.harness;
    fixtures = fullfile(testCase.TestData.root, 'flight-code', 'fixtures');

    assumeTrue(testCase, isfile(fullfile(fixtures, 'telemetry.csv')) && ...
                         isfile(fullfile(fixtures, 'matlab_reference.csv')), ...
        'Fixtures missing; run export_flight_fixtures.');

    [no_cc, ~] = system('command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1');
    assumeEqual(testCase, no_cc, 0, 'No C compiler on PATH; skipping the build.');

    % Build and compare as separate steps. Folding them together once reported a LINK
    % error as "veto decisions diverged", which sent the reader looking for a numerical
    % problem that did not exist. A diagnostic that names the wrong failure is worse than
    % no diagnostic.
    [build_status, build_out] = system(sprintf( ...
        'cd "%s" && make --no-print-directory clean >/dev/null 2>&1 && make --no-print-directory all 2>&1', ...
        harness));
    verifyEqual(testCase, build_status, 0, sprintf( ...
        ['The committed C failed to BUILD - this is a toolchain or portability problem, ' ...
         'not a behavioural one:\n%s'], build_out));

    [status, out] = system(sprintf('cd "%s" && make --no-print-directory check 2>&1', harness));

    verifyEqual(testCase, status, 0, sprintf( ...
        'The equivalence harness reported a mismatch:\n%s', out));
    verifySubstring(testCase, out, 'reproduces the MATLAB barrier', ...
        sprintf('Unexpected harness output:\n%s', out));
    verifySubstring(testCase, out, 'veto decisions differing : 0', ...
        sprintf('Veto decisions diverged between C and MATLAB:\n%s', out));
end


function verifySubstring(testCase, haystack, needle, msg)
    verifyTrue(testCase, contains(haystack, needle), msg);
end
