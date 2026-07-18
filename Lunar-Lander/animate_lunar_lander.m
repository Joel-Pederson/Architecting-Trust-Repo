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
    xlim(ax_phase, [min(min(dy), -50), max(max(dy), 10)]);
    ylim(ax_phase, [0, max(15000, max(y)) + 1000]);
    
    % FIGURE 2: Main Dual-Cam & Dashboard Visualizer
    fig_main = figure('Name', 'Lunar Lander Advanced Visualizer', 'Position', [600, 100, 1000, 700]);
    
    % Subplot 1: Global View (Rows 1-4, Column 1)
    ax_global = subplot(6, 2, [1, 3, 5, 7]);
    hold(ax_global, 'on');
    grid(ax_global, 'on');
    title(ax_global, 'Global Descent View', 'FontSize', 14);
    xlabel(ax_global, 'Horizontal Position (m)');
    ylabel(ax_global, 'Vertical Altitude (m)');
    
    % Subplot 2: Tracking Camera (Rows 1-4, Column 2)
    ax_track = subplot(6, 2, [2, 4, 6, 8]);
    hold(ax_track, 'on');
    grid(ax_track, 'on');
    title(ax_track, 'Tracking Camera', 'FontSize', 14);
    xlabel(ax_track, 'Horizontal Position (m)');
    ylabel(ax_track, 'Vertical Altitude (m)');
    
    % Subplot 3: Dummy Axis (Row 5, Column 1)
    % This is critical: It forces MATLAB's auto-layout to be perfectly symmetric, 
    % preventing the left plot from expanding downward and misaligning with the right plot.
    ax_dummy = subplot(6, 2, 9);
    axis(ax_dummy, 'off');
    
    % Subplot 4: Safety Veto Alarm (Row 5, Column 2)
    ax_veto = subplot(6, 2, 10);
    set(ax_veto, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none', 'Color', 'none');
    
    % Subplot 5: Systems Dashboard (Row 6, Columns 1 & 2)
    ax_dash = subplot(6, 2, [11, 12]);
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
    
    % Safety Veto Alarm (In its own mini subplot below tracking cam)
    txt_veto = text(ax_veto, 0.5, 0.5, 'SAFETY OVERRIDE ACTIVE', 'Units', 'normalized', 'FontSize', 16, 'FontName', 'Helvetica Neue', 'FontWeight', 'bold', 'Color', 'w', 'BackgroundColor', 'r', 'EdgeColor', 'k', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Visible', 'off');
    
    % Set initial views
    axis(ax_track, 'equal');
    
    % Force the Global View to be perfectly square in data units, 
    % so that 'axis equal' gives it the exact same height as Tracking Cam
    max_alt = max(15000, max(y));
    y_span = max_alt + 1000;
    ylim(ax_global, [0, y_span]);
    xlim(ax_global, [-y_span/2, y_span/2]); 
    axis(ax_global, 'equal');
    
    skip_frames = 2; 
    
    % --- 3. Animation Loop ---
    for i = 1:skip_frames:length(t)
        if ~isvalid(fig_main) || ~isvalid(fig_phase)
            break; 
        end
        
        curr_x = x(i);
        curr_y = y(i);
        curr_dy = dy(i);
        curr_theta = theta(i);
        curr_thrust = thrust(i);
        curr_fuel = fuel(i);
        is_veto = veto(i);
        
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
        
        global_scale = 50; 
        
        % Global View patches
        body_patch_g.XData = (b_coords(1,:) * global_scale) + curr_x;
        body_patch_g.YData = (b_coords(2,:) * global_scale) + curr_y;
        nozzle_patch_g.XData = (n_coords(1,:) * global_scale) + curr_x;
        nozzle_patch_g.YData = (n_coords(2,:) * global_scale) + curr_y;
        
        % Tracking View patches
        body_patch_t.XData = b_coords(1,:) + curr_x;
        body_patch_t.YData = b_coords(2,:) + curr_y;
        nozzle_patch_t.XData = n_coords(1,:) + curr_x;
        nozzle_patch_t.YData = n_coords(2,:) + curr_y;
        
        % Flame Update & Safety Alarm
        if is_veto
            txt_veto.Visible = 'on';
            flame_color = [0 0.5 1]; 
        else
            txt_veto.Visible = 'off';
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
            
            flame_patch_g.XData = (f_coords(1,:) * global_scale) + curr_x;
            flame_patch_g.YData = (f_coords(2,:) * global_scale) + curr_y;
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
        
        % Update Telemetry Text
        telemetry_str = sprintf('T: %6.1f s\nAlt: %6.1f m\nX: %6.1f m\nTheta: %6.1f deg', ...
            t(i), curr_y, curr_x, rad2deg(curr_theta));
        txt_telemetry.String = telemetry_str;
        
        drawnow limitrate;
    end
end
