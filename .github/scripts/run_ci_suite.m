function run_ci_suite(root)
%RUN_CI_SUITE Run the MATLAB test suite and publish failures as CI annotations.
%
% GitHub Actions job logs are readable only by repository admins, but the annotations a
% job emits are public. A bare `assertSuccess` therefore fails the build with no
% externally visible reason, which makes a red CI run expensive to diagnose for anyone
% without admin rights - including from a clone or a fork.
%
% This wrapper is behaviour-preserving with respect to the build result: the job still
% fails if and only if a test fails. It only adds visibility, by echoing the failure
% diagnostics as `::error::` workflow commands before it exits.
%
% Inputs:
%   root - (Optional) folder to scan for tests, default pwd.

    if nargin < 1 || isempty(root), root = pwd; end

    addpath(genpath(root));

    % Capture the console transcript as well as the result objects. The transcript is the
    % only place the full assertion text reliably appears across MATLAB releases; the
    % TestResult Details schema has changed between versions.
    transcript = '';
    try
        transcript = evalc('results = runtests(root, ''IncludeSubfolders'', true);');
    catch ME
        emit_error('Test suite did not run to completion', ...
                   sprintf('%s\n%s', ME.identifier, ME.message));
        fprintf('%s\n', transcript);
        rethrow(ME);
    end

    % Preserve the normal log for anyone who can read it.
    fprintf('%s\n', transcript);

    failed = results([results.Failed]);
    fprintf('===SUMMARY=== Total: %d  Passed: %d  Failed: %d  Incomplete: %d\n', ...
        numel(results), nnz([results.Passed]), nnz([results.Failed]), ...
        nnz([results.Incomplete]));

    if ~isempty(failed)
        % One annotation per failing test, so the checks UI lists them individually.
        for k = 1:numel(failed)
            emit_error(failed(k).Name, diagnostic_for(failed(k)));
        end
        % Plus the transcript tail, which carries the assertion values themselves.
        emit_error('MATLAB test transcript (tail)', tail_of(transcript, 6000));
    end

    assertSuccess(results);
end


function d = diagnostic_for(result)
% Best-effort extraction of a per-test diagnostic across MATLAB releases.
    d = '';
    try
        recs = result.Details.DiagnosticRecord;
        parts = cell(1, numel(recs));
        for k = 1:numel(recs)
            parts{k} = sprintf('%s\n%s', recs(k).Event, recs(k).Report);
        end
        d = strjoin(parts, newline);
    catch
        % Older/newer schema: fall back to the transcript annotation below.
    end
    if isempty(strtrim(d))
        d = sprintf('Test failed (duration %.2fs). See the transcript annotation.', ...
                    result.Duration);
    end
end


function t = tail_of(str, n)
    if numel(str) > n
        t = ['...(truncated)...' newline str(end-n+1:end)];
    else
        t = str;
    end
end


function emit_error(title, message)
% Emit a GitHub Actions error annotation. Workflow commands are single-line, so newlines
% and the delimiter characters must be percent-encoded or the payload is silently cut.
    fprintf('::error title=%s::%s\n', escape_property(title), escape_data(message));
end


function s = escape_data(s)
    s = strrep(s, '%',  '%25');
    s = strrep(s, sprintf('\r'), '%0D');
    s = strrep(s, sprintf('\n'), '%0A');
end


function s = escape_property(s)
    s = escape_data(s);
    s = strrep(s, ':', '%3A');
    s = strrep(s, ',', '%2C');
end
