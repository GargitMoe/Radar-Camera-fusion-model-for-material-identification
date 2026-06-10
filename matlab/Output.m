data_root = "";  % 改成你的路径，例如 "D:\radar_raw"
out_root  = "";  % 输出 RA 的 .mat 目录，例如 "D:\radar_ra_mat"

scenes = ["scene1", "scene2", "scene3"];
classes = ["metal", "glass", "plastic", "cardboard"];  % 或你当前实际类名

% 统一尺寸
H_target = 128;
W_target = 128;

% 参数结构体
params.Num_ADCSamples = 512;
params.Num_Tx         = 3;
params.Num_Rx         = 4;
params.Num_chirp      = 8;      % 关键：这里决定一份 bin 生成几个 .mat
params.Num_readframe  = 1;
params.fc             = 60.9e9;
params.Sampling_freq  = 5e6;
params.Chirp_slope    = 29.982e12;  % 注意你代码里用的是 29.982e12

for s = 1:numel(scenes)
    scene_name = scenes(s);
    for cidx = 1:numel(classes)
        cls_name = classes(cidx);

        in_dir  = fullfile(data_root, scene_name, cls_name);
        out_dir = fullfile(out_root,  scene_name, cls_name);
        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end

        bin_files = dir(fullfile(in_dir, "*.bin"));
        for k = 1:numel(bin_files)
            bin_path = fullfile(in_dir, bin_files(k).name);
            fprintf("Processing %s\n", bin_path);

            % ====== 现在这里返回的是 [Num_chirp, Nang, Nrange] ======
            [RA_mag_all, theta_deg, R_axis] = compute_RA_from_bin(bin_path, params);

            % 把 bin 文件名拆出来（不带后缀）
            [~, base_name, ~] = fileparts(bin_files(k).name);

            % ====== 对每个 chirp 单独生成一个 .mat ======
            for ch = 1:params.Num_chirp
                RA_mag = squeeze(RA_mag_all(ch, :, :));  % [Nang × Nrange]

                % ---- 裁剪 0~maxR 距离范围 ----
                maxR = 5;   % 你现在常用的最大距离，可按需微调
                valid_idx = find(R_axis <= maxR);
                RA_mag_crop = RA_mag(:, valid_idx);
                R_axis_crop = R_axis(valid_idx);

                % ---- 插值到 H_target × W_target ----
                theta_target = linspace(min(theta_deg), max(theta_deg), H_target);
                R_target     = linspace(min(R_axis_crop), max(R_axis_crop), W_target);

                [R_grid, T_grid] = meshgrid(R_axis_crop, theta_deg);
                [R_new,  T_new ] = meshgrid(R_target, theta_target);

                RA_mag_resize = interp2(R_grid, T_grid, RA_mag_crop, ...
                                        R_new, T_new, 'linear', 0);  % 超出范围补 0

                % ---- 归一化到 [0,1]（每张图各自归一化） ----
                max_val = max(RA_mag_resize(:));
                if max_val > 0
                    RA_norm = RA_mag_resize / max_val;
                else
                    RA_norm = RA_mag_resize;
                end

                % ---- 保存 .mat ----
                % 文件名加上 chirp 序号，如 xxx_ch01.mat
                out_name = sprintf('%s_ch%02d.mat', base_name, ch);
                out_path = fullfile(out_dir, out_name);

                ra     = RA_norm;    % H_target × W_target, 单通道图
                label  = cls_name;
                scene  = scene_name;
                chirp_index = ch;

                save(out_path, 'ra', 'label', 'scene', 'chirp_index', ...
                               'theta_target', 'R_target', '-v7');
            end
        end
    end
end
