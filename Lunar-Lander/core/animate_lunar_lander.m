function animate_lunar_lander(t, x, y, dy, theta, thrust, fuel, veto, params)
    % animate_lunar_lander Visualizes the 2D flight path of the Lunar Lander
    
    % --- 1. Set up the Figures ---
    % FIGURE 1: Phase Space Plot (Separate Window)
    fig_phase = figure('Name', 'Lunar Lander Phase Space', 'Position', [50, 200, 500, 500]);
    ax_phase = axes('Parent', fig_phase);
    hold(ax_phase, 'on');
    grid(ax_phase, 'on');
    box(ax_phase, 'on');
    title(ax_phase, 'Phase Space (Glide Slope)', 'FontSize', 14);
    xlabel(ax_phase, 'Descent Rate (m/s)');
    ylabel(ax_phase, 'Vertical Altitude (m)');
    
    phase_trail = plot(ax_phase, dy(1), y(1), 'm-', 'LineWidth', 2);
    phase_current = plot(ax_phase, dy(1), y(1), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 8);
    % Axes fit the actual flight envelope. Hard-coding a 15,000 m ceiling squashed every
    % realistic descent profile into a sliver at the bottom of the plot.
    xlim(ax_phase, [min(min(dy), -1) * 1.1, max(max(dy), 1) * 1.1]);
    ylim(ax_phase, [0, max(y) * 1.1 + 1]);
    
    % FIGURE 2: Main Dual-Cam & Dashboard Visualizer
    fig_main = figure('Name', 'Lunar Lander Advanced Visualizer', 'Position', [600, 100, 1000, 700]);
    
    % Subplot 1: Global View (Rows 1-5, Column 1)
    ax_global = subplot(7, 2, [1, 3, 5, 7, 9]);
    hold(ax_global, 'on');
    grid(ax_global, 'on');
    title(ax_global, 'Global Descent View', 'FontSize', 14);
    xlabel(ax_global, 'Horizontal Position (m)');
    ylabel(ax_global, 'Vertical Altitude (m)');
    
    % Subplot 2: Tracking Camera (Rows 1-5, Column 2)
    ax_track = subplot(7, 2, [2, 4, 6, 8, 10]);
    hold(ax_track, 'on');
    grid(ax_track, 'on');
    title(ax_track, 'Tracking Camera', 'FontSize', 14);
    xlabel(ax_track, 'Horizontal Position (m)');
    ylabel(ax_track, 'Vertical Altitude (m)');
    
    % Subplot 5: Systems Dashboard (Row 6, Columns 1 & 2)
    ax_dash = subplot(7, 2, [11, 12]);
    hold(ax_dash, 'on');
    grid(ax_dash, 'on');
    set(ax_dash, 'YColor', 'none', 'YTick', [], 'Layer', 'top'); % Hide Y axis, put grid on top of bars
    xlim(ax_dash, [0, 100]);
    ylim(ax_dash, [0, 2]);
    title(ax_dash, 'Systems Dashboard', 'FontSize', 12, 'FontWeight', 'bold');
    
    % --- 2. Define Lander Geometry ---
    W = 7; 
    H = 4.3;
    
    body_x = [-W/2, W/2, W/2, 0, -W/2];
    body_y = [-H/2, -H/2, H/2, H/2 + 2, H/2];
    
    nozzle_W = 2;
    nozzle_H = 1.5;
    nozzle_x = [-nozzle_W/2, nozzle_W/2, 0];
    nozzle_y = [-H/2 - nozzle_H, -H/2 - nozzle_H, -H/2];
    
    % Function to draw static environment
    function draw_environment(ax_target)
        plot(ax_target, [-100000, 100000], [0, 0], 'Color', [0.5 0.5 0.5], 'LineWidth', 2);
        target_W = 20;
        plot(ax_target, [-target_W/2, target_W/2], [0, 0], 'g-', 'LineWidth', 6);
        plot(ax_target, [-target_W/2, -target_W/2], [0, 10], 'k-', 'LineWidth', 1.5); 
        patch(ax_target, [-target_W/2, -target_W/2, -target_W/2-4], [10, 7, 8.5], 'r', 'EdgeColor', 'k'); 
        plot(ax_target, [target_W/2, target_W/2], [0, 10], 'k-', 'LineWidth', 1.5); 
        patch(ax_target, [target_W/2, target_W/2, target_W/2+4], [10, 7, 8.5], 'r', 'EdgeColor', 'k'); 
        box(ax_target, 'on');
    end
    
    draw_environment(ax_global);
    draw_environment(ax_track);
    
    % Initialize graphics objects for Global View
    trail_line = plot(ax_global, x(1), y(1), 'm-', 'LineWidth', 2);
    body_patch_g = patch(ax_global, 'XData', body_x, 'YData', body_y, 'FaceColor', [0.8 0.7 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
    nozzle_patch_g = patch(ax_global, 'XData', nozzle_x, 'YData', nozzle_y, 'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'k', 'LineWidth', 1.5);
    flame_patch_g = patch(ax_global, 'XData', [], 'YData', [], 'FaceColor', [1 0.5 0], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    
    % Initialize graphics objects for Tracking Cam
    body_patch_t = patch(ax_track, 'XData', body_x, 'YData', body_y, 'FaceColor', [0.8 0.7 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
    nozzle_patch_t = patch(ax_track, 'XData', nozzle_x, 'YData', nozzle_y, 'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'k', 'LineWidth', 1.5);
    flame_patch_t = patch(ax_track, 'XData', [], 'YData', [], 'FaceColor', [1 0.5 0], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    
    % Dashboard Horizontal Bars
    box(ax_dash, 'on');
    
    % Thrust Bar (Top)
    rectangle(ax_dash, 'Position', [12, 1.15, 85, 0.7], 'FaceColor', [0.2 0.2 0.2]);
    thrust_bar = rectangle(ax_dash, 'Position', [12, 1.15, 0, 0.7], 'FaceColor', 'c', 'EdgeColor', 'none');
    text(ax_dash, 10, 1.5, 'THRUST %', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
    
    % Fuel Bar (Bottom)
    rectangle(ax_dash, 'Position', [12, 0.15, 85, 0.7], 'FaceColor', [0.2 0.2 0.2]);
    fuel_bar = rectangle(ax_dash, 'Position', [12, 0.15, 85, 0.7], 'FaceColor', 'g', 'EdgeColor', 'none');
    text(ax_dash, 10, 0.5, 'FUEL', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
    
    % Telemetry HUD in Tracking Cam (Top Left)
    txt_telemetry = text(ax_track, 0.02, 0.95, '', 'Units', 'normalized', 'FontSize', 12, 'FontName', 'Helvetica Neue', 'BackgroundColor', 'w', 'EdgeColor', 'k', 'VerticalAlignment', 'top');
    
    % Subplot 6: Landing Status (Row 7, Column 1)
    ax_status = subplot(7, 2, 13);
    set(ax_status, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none', 'Color', 'none');
    text(ax_status, 0.65, 0.5, 'Landing Status: ', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
    txt_status_val = text(ax_status, 0.66, 0.5, ' PENDING ', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k', 'BackgroundColor', 'y', 'EdgeColor', 'k', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    
    % Subplot 7: Safety Veto Alarm (Row 7, Column 2)
    ax_veto = subplot(7, 2, 14);
    set(ax_veto, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none', 'Color', 'none');
    text(ax_veto, 0.65, 0.5, 'Safety Sidecar: ', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
    txt_veto_val = text(ax_veto, 0.66, 0.5, ' INACTIVE ', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w', 'BackgroundColor', [0 0.8 0], 'EdgeColor', 'k', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    
    % Set initial views
    axis(ax_track, 'equal');
    
    % Scale Global View to fit the entire trajectory. Logic lives in
    % core/compute_view_limits.m so it can be regression-tested.
    [min_x, max_x, min_y, max_y] = compute_view_limits(x, y);
    xlim(ax_global, [min_x, max_x]);
    ylim(ax_global, [min_y, max_y]);
    
    % Create a Replay Button in the top right of the main visualizer figure
    uicontrol('Parent', fig_main, ...
              'Style', 'pushbutton', ...
              'String', '▶ REPLAY ANIMATION', ...
              'Units', 'normalized', ...
              'Position', [0.76, 0.93, 0.22, 0.05], ...
              'FontSize', 11, ...
              'FontWeight', 'bold', ...
              'BackgroundColor', [0.1 0.5 0.8], ...
              'ForegroundColor', 'w', ...
              'Callback', @(~,~) run_replay());

    is_animating = false;

    % Playback pacing, derived from the data rather than fixed.
    %
    % skip_frames was hardcoded to 2, which suited telemetry logged at the 50 Hz PHYSICS
    % rate: a 900 s run is 45,000 samples, so every second frame still gave a long
    % animation. Episodes rolled at the 10 Hz AGENT rate are ~300 samples for a 30 s
    % flight, and at skip 2 that renders in a fraction of a second - the figure appears
    % already finished and REPLAY looks like it does nothing.
    %
    % Render a bounded number of frames spread over a fixed wall-clock duration, so
    % playback looks the same whether the source is 300 samples or 45,000.
    % TARGET_DURATION sets the PAUSE budget, not the total: rendering costs roughly as
    % much again, so a 5 s budget plays back in about 10-13 s. Measured on both a 179
    % frame Phase 1 landing and a 1706 frame Phase 3 descent, which is the point - the
    % two now take a comparable time to watch despite a 10x difference in sample count.
    TARGET_FRAMES   = 300;    % upper bound on rendered frames
    TARGET_DURATION = 5.0;    % seconds of PAUSE spread across the replay

    skip_frames = max(1, round(numel(t) / TARGET_FRAMES));
    n_rendered  = numel(1:skip_frames:numel(t));
    frame_pause = TARGET_DURATION / max(1, n_rendered);
    
    % Run animation once automatically on launch
    run_replay();
    
    % --- 3. Animation Helper Function ---
    function run_replay()
        if is_animating
            return; % Prevent overlapping animation loops if clicked while running
        end
        is_animating = true;
        % Guarantee the flag is cleared even if the loop below throws. Without this, one
        % error anywhere in the render path leaves is_animating stuck true, and every
        % subsequent REPLAY click returns immediately at the guard above - the button
        % appears dead for the life of the figure, with no error shown, which is a very
        % confusing thing to debug.
        reset_flag = onCleanup(@() set_animating(false));

        % Reset trailing paths and status for replay
        if isvalid(ax_global) && isvalid(ax_phase)
            trail_line.XData = x(1);
            trail_line.YData = y(1);
            phase_trail.XData = dy(1);
            phase_trail.YData = y(1);
            txt_status_val.String = ' PENDING ';
            txt_status_val.BackgroundColor = 'y';
            txt_status_val.Color = 'k';
        end
        
        % --- Glyph scaling for the GLOBAL view ---
        % The global axes are NOT axis-equal: a Phase 1 landing spans ~20 m across and
        % ~65 m up, so a single scale factor draws a 7 m x 4.3 m lander at 35% of the
        % plot width and 6% of its height - hugely stretched. The tracking camera does not
        % have this problem because it is axis-equal.
        %
        % Fix: scale x and y independently so the glyph's PIXEL aspect matches its true
        % one. Writing upp for data-units-per-pixel, the on-screen shape is
        %     screen = diag(gscale_x/upp_x, gscale_y/upp_y) * R * shape
        % so setting gscale_x/upp_x == gscale_y/upp_y makes that a UNIFORM scale of the
        % rotated true shape: undistorted, and correctly rotated with it.
        %
        % Computed per replay rather than per frame so a window resize is picked up on
        % the next run without paying for getpixelposition on every frame.
        GLYPH_FRAC = 0.07;                       % of visible height
        ax_px  = getpixelposition(ax_global);
        upp_x  = diff(xlim(ax_global)) / max(ax_px(3), 1);
        upp_y  = diff(ylim(ax_global)) / max(ax_px(4), 1);
        gscale_y = max(1, (GLYPH_FRAC * diff(ylim(ax_global))) / H);
        gscale_x = gscale_y * (upp_x / upp_y);

        for i = 1:skip_frames:length(t)
            if ~isvalid(fig_main) || ~isvalid(fig_phase)
                is_animating = false;
                break; 
            end
            
            curr_x = x(i);
            curr_y = y(i);
            curr_dy = dy(i);
            curr_theta = theta(i);
            curr_thrust = thrust(i);
            curr_fuel = fuel(i);
            is_veto = veto(i);
            
            % Dynamic dx calculation
            if i > 1
                curr_dx = (x(i) - x(i-1)) / (t(i) - t(i-1));
            else
                curr_dx = 0;
            end
            
            % 1. Phase Space Update
            phase_trail.XData = dy(1:i);
            phase_trail.YData = y(1:i);
            phase_current.XData = curr_dy;
            phase_current.YData = curr_y;
            
            % 2. Geometry Update
            R = [cos(curr_theta), -sin(curr_theta); 
                 sin(curr_theta),  cos(curr_theta)];
                 
            b_coords = R * [body_x; body_y];
            n_coords = R * [nozzle_x; nozzle_y];
            
            % Global View patches. gscale_x and gscale_y are computed once per replay
            % (see below) and are deliberately DIFFERENT, to cancel the axes' aspect.
            body_patch_g.XData = (b_coords(1,:) * gscale_x) + curr_x;
            body_patch_g.YData = (b_coords(2,:) * gscale_y) + curr_y;
            nozzle_patch_g.XData = (n_coords(1,:) * gscale_x) + curr_x;
            nozzle_patch_g.YData = (n_coords(2,:) * gscale_y) + curr_y;
            
            % Tracking View patches
            body_patch_t.XData = b_coords(1,:) + curr_x;
            body_patch_t.YData = b_coords(2,:) + curr_y;
            nozzle_patch_t.XData = n_coords(1,:) + curr_x;
            nozzle_patch_t.YData = n_coords(2,:) + curr_y;
            
            % Flame Update & Safety Alarm
            if is_veto
                txt_veto_val.String = ' ACTIVE ';
                txt_veto_val.BackgroundColor = [0.8 0 0]; % Red
                flame_color = [0 0.5 1]; 
            else
                txt_veto_val.String = ' INACTIVE ';
                txt_veto_val.BackgroundColor = [0 0.8 0]; % Green
                flame_color = [1 0.5 0]; 
            end
            flame_patch_g.FaceColor = flame_color;
            flame_patch_t.FaceColor = flame_color;
            
            if curr_thrust > 0
                flame_L = 15 * (curr_thrust / params.max_main_thrust);
                flame_x_base = [-nozzle_W/2, nozzle_W/2, 0];
                flame_y_base = [-H/2 - nozzle_H, -H/2 - nozzle_H, -H/2 - nozzle_H - flame_L];
                flame_y_base(3) = flame_y_base(3) * (0.9 + 0.2*rand()); % flicker
                
                f_coords = R * [flame_x_base; flame_y_base];
                
                flame_patch_g.XData = (f_coords(1,:) * gscale_x) + curr_x;
                flame_patch_g.YData = (f_coords(2,:) * gscale_y) + curr_y;
                flame_patch_t.XData = f_coords(1,:) + curr_x;
                flame_patch_t.YData = f_coords(2,:) + curr_y;
            else
                flame_patch_g.XData = [];
                flame_patch_g.YData = [];
                flame_patch_t.XData = [];
                flame_patch_t.YData = [];
            end
            
            % Trail update
            trail_line.XData = x(1:i);
            trail_line.YData = y(1:i);
            
            % Tracking Camera bounds
            cam_width = 100;
            cam_height = 100;
            min_y = max(-10, curr_y - cam_height/2);
            max_y = min_y + cam_height;
            if min_y <= -10
                min_y = -10;
                max_y = -10 + cam_height;
            end
            xlim(ax_track, [curr_x - cam_width/2, curr_x + cam_width/2]);
            ylim(ax_track, [min_y, max_y]);
            
            % Update Dashboard UI
            thrust_pct = curr_thrust / params.max_main_thrust;
            thrust_w = 85 * thrust_pct;
            if thrust_w > 0
                thrust_bar.Position = [12, 1.15, thrust_w, 0.7];
                thrust_bar.Visible = 'on';
            else
                thrust_bar.Visible = 'off';
            end
            
            fuel_pct = curr_fuel / 8200; 
            fuel_w = 85 * fuel_pct;
            if fuel_w > 0
                fuel_bar.Position = [12, 0.15, fuel_w, 0.7];
                fuel_bar.Visible = 'on';
            else
                fuel_bar.Visible = 'off';
            end
            
            if fuel_pct < 0.2
                fuel_bar.FaceColor = 'r';
            elseif fuel_pct < 0.5
                fuel_bar.FaceColor = 'y';
            else
                fuel_bar.FaceColor = 'g';
            end
            
            % Update Landing Status Dynamically
            if curr_y <= 2
                % Apollo 11 strict landing tolerances
                is_safe = (abs(curr_dy) <= 3) && (abs(curr_dx) <= 3) && (abs(curr_theta) <= 0.2);
                if is_safe
                    txt_status_val.String = ' SAFE ';
                    txt_status_val.BackgroundColor = [0 0.8 0]; % Green
                    txt_status_val.Color = 'w';
                else
                    txt_status_val.String = ' CRASH ';
                    txt_status_val.BackgroundColor = [0.8 0 0]; % Red
                    txt_status_val.Color = 'w';
                end
            else
                txt_status_val.String = ' PENDING ';
                txt_status_val.BackgroundColor = 'y'; % Yellow
                txt_status_val.Color = 'k';
            end
            
            % Update Telemetry Text
            telemetry_str = sprintf('T: %6.1f s\nAlt: %6.1f m\nX: %6.1f m\nTheta: %6.1f deg', ...
                t(i), curr_y, curr_x, rad2deg(curr_theta));
            txt_telemetry.String = telemetry_str;
            
            % Plain drawnow, not limitrate: limitrate DROPS frames to keep up, which on a
            % short agent-rate episode discards most of the animation. The pause sets the
            % pace instead.
            drawnow;
            pause(frame_pause);
        end
        is_animating = false;
    end

    function set_animating(tf)
        is_animating = tf;
    end
end
