function [idx, label] = phase_from_name(name)
% PHASE_FROM_NAME Resolves a readable scenario name to a curriculum phase index.
%
% One definition, so every entry point accepts the same vocabulary and nobody has to
% remember whether the powered descent is phase 3 or 4.
%
%   'touchdown'  1   50 m, 2 m/s        final touchdown only
%   'approach'   2   500 m, 10 m/s      glide slope approach
%   'terminal'   3   2500 m, 25 m/s     full terminal descent
%   'orbit'      4   15.2 km, 1697 m/s  powered descent from orbit, 550 km downrange
%
% Numeric indices are still accepted, so existing scripts keep working. Aliases are
% included for the names people actually reach for.
%
% Inputs:
%   name - char/string name, or a numeric phase index
%
% Outputs:
%   idx   - phase index 1-4
%   label - canonical human-readable description

    aliases = { ...
        1, 'touchdown', '50 m touchdown',           {'touchdown','hover','land','p1','phase1'}; ...
        2, 'approach',  '500 m glide slope',        {'approach','glide','glideslope','p2','phase2'}; ...
        3, 'terminal',  '2.5 km terminal descent',  {'terminal','descent','p3','phase3'}; ...
        4, 'orbit',     'powered descent from orbit (15.2 km, 1697 m/s, 550 km)', ...
                                                    {'orbit','pdi','powered','poweredddescent','powered_descent','p4','phase4'} };

    if isnumeric(name)
        idx = name;
        if idx < 1 || idx > size(aliases,1)
            error('phaseFromName:BadIndex', 'Phase index %d is outside 1-%d.', idx, size(aliases,1));
        end
        label = aliases{idx,3};
        return;
    end

    key = strrep(strtrim(char(name)), ' ', '');
    for r = 1:size(aliases,1)
        if any(strcmpi(key, aliases{r,4}))
            idx   = aliases{r,1};
            label = aliases{r,3};
            return;
        end
    end

    names = strjoin(cellfun(@(c) sprintf('''%s''', c), aliases(:,2)', 'UniformOutput', false), ', ');
    error('phaseFromName:Unknown', ...
        'Unknown scenario "%s". Use one of: %s (or a phase number 1-4).', char(name), names);
end
