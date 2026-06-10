%% === 基本参数（沿用你的配置，修正单位） ===
c  = 3e8;
fc = 60.9e9;                 % 中心频率
lambda = c / fc;
d_ant  = lambda / 2;         % 理论 λ/2 阵元间距

Ntx    = Num_Tx;
Nrx    = Num_Rx;
Nchirp = Num_chirp;
Ns     = Num_ADCSamples;

% 注意：Chirp_slope 要用 Hz/s，而不是 MHz/us
% 你原始 cfg 是 29.982 MHz/us → 29.982e6 * 1e6 = 29.982e12 Hz/s
S      = 29.982e12;          % Chirp_slope in Hz/s
fs     = Sampling_freq;      % 这里已经是 Hz（5e6）

%% === 1. 取一帧数据 & 重构为 [Rx, Tx, chirp, sample] ===
frame_data = retVal(:, 1 : (Ntx * Nchirp * Ns));   % 只用第一帧

data4d = zeros(Nrx, Ntx, Nchirp, Ns);
for tx = 1:Ntx
    for ch = 1:Nchirp
        idx0 = ((tx-1)*Nchirp + (ch-1))*Ns + 1;
        idx1 = idx0 + Ns - 1;
        data4d(:, tx, ch, :) = frame_data(:, idx0:idx1);
    end
end

%% === 2. 构造 12 路虚拟通道，并在 chirp 维上平均 ===
virtChan = Ntx * Nrx;        % 12
virtData = zeros(virtChan, Ns);

vcIdx = 1;
for tx = 1:Ntx
    for rx = 1:Nrx
        % 该 virtual channel 上所有 chirp 的数据 [Nchirp × Ns]
        tmp = squeeze(data4d(rx, tx, :, :));
        % 论文里使用每帧 32 chirp 做 RAheatmap，你这里 Nchirp=8，
        % 用 “平均/累加” 方式得到一帧的稳态回波（忽略多普勒）
        avg_tmp = mean(tmp, 1);   % 1 × Ns
        virtData(vcIdx, :) = avg_tmp;
        vcIdx = vcIdx + 1;
    end
end

%% === 3. 选 azimuth 相关的 8 个虚拟通道（类似论文中 C1/C2/C3） ===
% 你之前的经验：1,2,3,4,9,10,11,12 为水平向 azimuth 通道
az_idx       = [1 2 3 4 9 10 11 12];     % 8 × Ns
virtData_az  = virtData(az_idx, :);

%% === 4. Range-FFT：快时间方向做 1D FFT（对应论文 4.2 的 Range-FFT） ===
Nfft_r  = Ns;
win_r   = hann(Ns).';                        % Hann 窗
rangeFFT = fft(virtData_az .* win_r, Nfft_r, 2);   % 沿列做 FFT（行：通道）

% 只保留正频半谱
rangeFFT = rangeFFT(:, 1:Nfft_r/2);

% 距离轴（公式 (10): d = f_IF * c / (2S)）
fb     = (0:(Nfft_r/2-1)) * (fs / Nfft_r);   % IF 频率 (Hz)
R_axis = c * fb / (2 * S);                   % 距离 (m)

%% === 5. Angle-FFT：阵列方向做 FFT 得到 AoA（对应论文的 Angle-FFT） ===
% 论文是对阵列做 1D FFT，这里对 “通道方向” 做 FFT
Nc_az    = size(rangeFFT, 1);        % 这里是 8 通道
Nfft_ang = 64;                       % 角度 FFT 点数

win_ang   = hann(Nc_az);             % 通道维 Hann 窗（8×1）
data_win  = rangeFFT .* win_ang;     % 每一列乘以 8×1 窗

% 沿通道方向做 FFT，得到 [Nfft_ang × Nfft_r/2]
RA_complex = fftshift( fft(data_win, Nfft_ang, 1), 1);

% 构造角度轴，参考 AoA 公式 (11)，但这里用 ULA 近似
m = (-Nfft_ang/2 : Nfft_ang/2-1);
% 理论上：sin(theta) ≈ k*d_ant / (π)，这里用经验映射：2*m/Nfft_ang
sin_theta = (m / (Nfft_ang/2));     % 归一化到 [-1,1]
sin_theta = max(min(sin_theta, 1), -1);
theta_deg = asind(sin_theta);

%% === 6. 高通 / 背景抑制（对应论文的 high-pass filtering & 环境抑制） ===
% (1) 去掉每个 range bin 的全局平均 → 类似高通
% 先在角度维对每个距离 bin 做减均值
RA_hp = RA_complex - mean(RA_complex, 1);

% (2) 可选：加载一张“背景 RA”做相减（你已有 RA_bg_25cm 的逻辑）
do_subtract_bg = false;   % 需要再打开
if do_subtract_bg
    bg = load('RA_bg_25cm.mat', 'RA_bg');
    RA_bg = bg.RA_bg;
    if ~isequal(size(RA_bg), size(RA_hp))
        error('RA_bg size mismatch.');
    end

    % 可选只在某个距离段做背景相减
    R_min = 0.55;
    R_max = 0.75;
    roi_idx = find(R_axis >= R_min & R_axis <= R_max);

    RA_clean = RA_hp;
    RA_clean(:, roi_idx) = RA_hp(:, roi_idx) - RA_bg(:, roi_idx);
else
    RA_clean = RA_hp;
end

%% === 7. 2D-CFAR：仿照论文 4.2.4 的 2D-CFAR （简单 cell-averaging 版本） ===
% 在 |RA_clean| 的 dB 图上做 CFAR，得到 mask
RA_abs = abs(RA_clean);
RA_dB  = 20*log10(RA_abs + 1e-6);

% CFAR 窗参数（可以根据效果调大调小）
guard_r = 1;   % 距离向 guard cells
guard_a = 1;   % 角度向 guard cells
train_r = 4;   % 距离向 training cells
train_a = 4;   % 角度向 training cells
alpha   = 3;   % 阈值因子：大一点 → 目标更少，背景更干净

[Na, Nr] = size(RA_dB);
CFAR_mask = false(Na, Nr);

for ia = 1+guard_a+train_a : Na-guard_a-train_a
    for ir = 1+guard_r+train_r : Nr-guard_r-train_r

        % 提取参考窗口（不含自己 & guard cells）
        a_min = ia - guard_a - train_a;
        a_max = ia + guard_a + train_a;
        r_min = ir - guard_r - train_r;
        r_max = ir + guard_r + train_r;

        win = RA_dB(a_min:a_max, r_min:r_max);

        % 把中心 + guard 区域清零，剩下才是 reference cells
        win( train_a+1 : train_a+2*guard_a+1, ...
             train_r+1 : train_r+2*guard_r+1 ) = NaN;

        % 取均值作为噪声估计
        mu_ref = mean(win(~isnan(win)), 'all');

        % 判决
        if RA_dB(ia, ir) > mu_ref + alpha
            CFAR_mask(ia, ir) = true;
        end
    end
end

% 用 CFAR_mask 把噪声压低一点（而不是完全置零）
RA_cfar = RA_dB;
noise_idx = ~CFAR_mask;
RA_cfar(noise_idx) = RA_cfar(noise_idx) - 10;   % 把背景往下压 10 dB

%% === 8. 角度范围裁剪：只保留 [-50°, 50°]（跟论文一致） ===
ang_keep = (theta_deg >= -50) & (theta_deg <= 50);
theta_plot = theta_deg(ang_keep);
RA_plot    = RA_cfar(ang_keep, :);

%% === 9. 画 RAheatmap ===
figure;
imagesc(R_axis, theta_plot, RA_plot);
set(gca,'YDir','normal');
xlabel('Range (m)');
ylabel('Angle (°)');
title('Range-Angle Heatmap with CFAR & High-pass');
colormap jet;
colorbar;

% 如果你只关心 0~2 m，可以这样限制横轴：
xlim([0 2]);
xticks(0:0.1:2);
xtickformat('%.1f');
grid on;
