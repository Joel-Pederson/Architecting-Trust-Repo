function animate_lunar_lander(t, x, y, theta, thrust, fuel, params)
    % animate_lunar_lander Visualizes the 2D flight path of the Lunar Lander
    
    % --- 1. Set up the Figure ---
    fig = figure('Name', 'Lunar Lander Dual-Cam Visualizer', 'Position', [100, 100, 1200, 600]);
    
    % Global View Axes
    ax1 = subplot(1, 2, 1);
    hold(ax1, 'on');
    grid(ax1, 'on');
    title(ax1, 'Global Descent View', 'FontSize', 14);
    xlabel(ax1, 'Horizontal Position (m)');
    ylabel(ax1, 'Vertical Altitude (m)');
    
    % Tracking Camera Axes
    ax2 = subplot(1, 2, 2);
    hold(ax2, 'on');
    grid(ax2, 'on');
    title(ax2, 'Tracking Camera', 'FontSize', 14);
    xlabel(ax2, 'Horizontal Position (m)');
    ylabel(ax2, 'Vertical Altitude (m)');
    
    % --- 2. Define Lander Geometry (Base coordinates, centered at 0,0) ---
    W = 7; 
    H = 4.3;
    
    body_x = [-W/2, W/2, W/2, 0, -W/2];
    body_y = [-H/2, -H/2, H/2, H/2 + 2, H/2];
    
    nozzle_W = 2;
    nozzle_H = 1.5;
    nozzle_x = [-nozzle_W/2, nozzle_W/2, 0];
    nozzle_y = [-H/2 - nozzle_H, -H/2 - nozzle_H, -H/2];
    
    % Function to draw static environment for both axes
    function draw_environment(ax_target)
        % Ground line
        plot(ax_target, [-100000, 100000], [0, 0], 'Color', [0.5 0.5 0.5], 'LineWidth', 2);
        
        % Target Landing Zone (Origin)
        target_W = 20;
        plot(ax_target, [-target_W/2, target_W/2], [0, 0], 'g-', 'LineWidth', 6);
        
        % Flags at each end (10m tall)
        plot(ax_target, [-target_W/2, -target_W/2], [0, 10], 'k-', 'LineWidth', 1.5); % Left pole
        patch(ax_target, [-target_W/2, -target_W/2, -target_W/2-4], [10, 7, 8.5], 'r', 'EdgeColor', 'k'); % Left flag
        plot(ax_target, [target_W/2, target_W/2], [0, 10], 'k-', 'LineWidth', 1.5); % Right pole
        patch(ax_target, [target_W/2, target_W/2, target_W/2+4], [10, 7, 8.5], 'r', 'EdgeColor', 'k'); % Right flag
        
        box(ax_target, 'on');
    end
    
    draw_environment(ax1);
    draw_environment(ax2);
    
    % Initialize graphics objects for Global View (ax1)
    trail_line = plot(ax1, x(1), y(1), 'm-', 'LineWidth', 2);
    body_patch1 = patch(ax1, 'XData', body_x, 'YData', body_y, 'FaceColor', [0.8 0.7 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
    nozzle_patch1 = patch(ax1, 'XData', nozzle_x, 'YData', nozzle_y, 'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'k', 'LineWidth', 1.5);
    flame_patch1 = patch(ax1, 'XData', [], 'YData', [], 'FaceColor', [1 0.5 0], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    
    % Initialize graphics objects for Tracking Cam (ax2)
    body_patch2 = patch(ax2, 'XData', body_x, 'YData', body_y, 'FaceColor', [0.8 0.7 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
    nozzle_patch2 = patch(ax2, 'XData', nozzle_x, 'YData', nozzle_y, 'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'k', 'LineWidth', 1.5);
    flame_patch2 = patch(ax2, 'XData', [], 'YData', [], 'FaceColor', [1 0.5 0], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    
    % Text for telemetry (only on ax2)
    txt_telemetry = text(ax2, 0.02, 0.95, '', 'Units', 'normalized', 'FontSize', 12, 'FontName', 'Helvetica Neue', 'BackgroundColor', 'w', 'EdgeColor', 'k', 'VerticalAlignment', 'top');
    
    % Set initial view for ax2
    axis(ax2, 'equal');
    
    % Pre-calculate global bounds for ax1
    max_alt = max(15000, max(y));
    ylim(ax1, [0, max_alt + 1000]);
    % Fixed horizontal view for global map, wide enough to see horizontal drift
    xlim(ax1, [-2000, 2000]); 
    
    % Force equal aspect ratio so the ship does not appear stretched/distorted
    axis(ax1, 'equal');
    
    % Set playback speed
    skip_frames = 2; 
    
    % --- 3. Animation Loop ---
    for i = 1:skip_frames:length(t)
        if ~isvalid(fig)
            break; 
        end
        
        curr_x = x(i);
        curr_y = y(i);
        curr_theta = theta(i);
        curr_thrust = thrust(i);
        curr_fuel = fuel(i);
        
        % Rotation matrix
        R = [cos(curr_theta), -sin(curr_theta); 
             sin(curr_theta),  cos(curr_theta)];
             
        % Transform coordinates
        b_coords = R * [body_x; body_y];
        n_coords = R * [nozzle_x; nozzle_y];
        
        % Scale up the ship for the Global View so it is visible from 15,000m away
        global_scale = 50; 
        
        % Update Global View patches
        body_patch1.XData = (b_coords(1,:) * global_scale) + curr_x;
        body_patch1.YData = (b_coords(2,:) * global_scale) + curr_y;
        nozzle_patch1.XData = (n_coords(1,:) * global_scale) + curr_x;
        nozzle_patch1.YData = (n_coords(2,:) * global_scale) + curr_y;
        
        % Update Tracking View patches
        body_patch2.XData = b_coords(1,:) + curr_x;
        body_patch2.YData = b_coords(2,:) + curr_y;
        nozzle_patch2.XData = n_coords(1,:) + curr_x;
        nozzle_patch2.YData = n_coords(2,:) + curr_y;
        
        % Draw Flame based on thrust
        if curr_thrust > 0
            flame_L = 15 * (curr_thrust / params.max_main_thrust);
            flame_x_base = [-nozzle_W/2, nozzle_W/2, 0];
            flame_y_base = [-H/2 - nozzle_H, -H/2 - nozzle_H, -H/2 - nozzle_H - flame_L];
            flame_y_base(3) = flame_y_base(3) * (0.9 + 0.2*rand()); % flicker
            
            f_coords = R * [flame_x_base; flame_y_base];
            
            flame_patch1.XData = (f_coords(1,:) * global_scale) + curr_x;
            flame_patch1.YData = (f_coords(2,:) * global_scale) + curr_y;
            flame_patch2.XData = f_coords(1,:) + curr_x;
            flame_patch2.YData = f_coords(2,:) + curr_y;
        else
            flame_patch1.XData = [];
            flame_patch1.YData = [];
            flame_patch2.XData = [];
            flame_patch2.YData = [];
        end
        
        % Update Trail on Global View
        trail_line.XData = x(1:i);
        trail_line.YData = y(1:i);
        
        % Dynamic X-axis for Global View if lander wanders off screen
        if curr_x > ax1.XLim(2) - 500
            xlim(ax1, [curr_x - 3500, curr_x + 500]);
        elseif curr_x < ax1.XLim(1) + 500
            xlim(ax1, [curr_x - 500, curr_x + 3500]);
        end
        
        % Update Camera Window (Tracking Camera)
        cam_width = 100;
        cam_height = 100;
        
        min_y = max(-10, curr_y - cam_height/2);
        max_y = min_y + cam_height;
        if min_y <= -10
            min_y = -10;
            max_y = -10 + cam_height;
        end
        
        xlim(ax2, [curr_x - cam_width/2, curr_x + cam_width/2]);
        ylim(ax2, [min_y, max_y]);
        
        % Update Telemetry Text
        telemetry_str = sprintf('T: %6.1f s\nAlt: %6.1f m\nX: %6.1f m\nTheta: %6.1f deg\nThrust: %3.0f%%\nFuel: %6.1f kg', ...
            t(i), curr_y, curr_x, rad2deg(curr_theta), (curr_thrust/params.max_main_thrust)*100, curr_fuel);
        txt_telemetry.String = telemetry_str;
        
        drawnow limitrate;
    end
end
