%% IWR6843 FMCW Basic Parameter
% Sensor Configuration
Start_freq = 60.000000e9; % 60GHz
Chirp_slope = 29.982e6; % 29.982MHz/us
Idle_time = 100.00e-6; % 100us
Tx_start_time = 0.00e-6; % 0us
ADC_start_time = 6.00e-6; % 6us
Num_ADCSamples = 512; % number of ADC samples per chirp
Sampling_freq = 5000e3; % kHz
Ramp_endtime = 60.00e-6; % 60us

% Chirp Configuration
Num_Tx = 3; % number of transmitter
Num_Rx = 4; % number of receivers
Chirp_config_num = Num_Tx;
Chirp_cycle_time = Idle_time + Ramp_endtime;

% Frame Configuration
Num_frame = 128;
Num_chirp = 8; % number of chrip in one frame
Frame_periodcity = 62.000000e-3; % 62ms
Duty_cycle = ((Chirp_config_num*Chirp_cycle_time)*Num_chirp/Frame_periodcity)*100; % Active-ramp (%)

% ADC sampling parameter
Sampling_start_freq = Start_freq + Chirp_slope*6;
Sampling_bandwidth = (Num_ADCSamples/Sampling_freq)*1e6*Chirp_slope;
Sampling_end_freq = Sampling_start_freq + Sampling_bandwidth;

%% Convert .bin file to .mat file
Num_readframe = 1; % 5s data, frame_length = 62ms
filename = "adc_data_1021_61_1_25_30.bin";
retVal = IWR6843_readDCA1000(filename, Num_ADCSamples, Num_Tx, Num_Rx, Num_chirp, Num_readframe);

%% === RA Map using all 12 virtual channels ===
Chirp_slope = 29.982e12; 
% 物理参数
c = 3e8;
fc = 60.9e9;                       
lambda = c / fc;                  % 波长
d = lambda / 2;                   % 阵元间距 λ/

% 重构维度
Ntx = Num_Tx;
Nrx = Num_Rx;
Nchirp = Num_chirp;
Ns = Num_ADCSamples;

% 从 retVal 提取第一帧数据 (假设 Num_readframe>=1)
frame_data = retVal(:, 1 : (Ntx * Nchirp * Ns));

% reshape 成通道立方 [Rx, Tx, chirp, sample]
data4d = zeros(Nrx, Ntx, Nchirp, Ns);
for tx = 1:Ntx
    for ch = 1:Nchirp
        idx0 = ((tx-1)*Nchirp + (ch-1))*Ns + 1;
        idx1 = idx0 + Ns -1;
        data4d(:, tx, ch, :) = frame_data(:, idx0:idx1);
    end
end

virtChan = Ntx * Nrx;
virtData = zeros(virtChan, Ns);

vcIdx = 1;
for tx = 1:Ntx
    for rx = 1:Nrx
        % 针对该 virtual channel：对所有 chirp平均
        tmp = squeeze(data4d(rx, tx, :, :));     % Nchirp × Ns
        avg_tmp = mean(tmp, 1);                   % 1 × Ns
        virtData(vcIdx, :) = avg_tmp;
        vcIdx = vcIdx + 1;
    end
end
az_idx    = [1 2 3 4 9 10 11 12];   % 前四 + 后四
virtData_az = virtData(az_idx, :);  % 8 × Ns
% Range FFT（快时间方向）
win_r = hann(Ns).';
Nfft_r = Ns;    
rangeFFT_mat = fft(virtData_az.* win_r, Nfft_r, 2);   % 按行做 FFT
% 取前一半（正频）
rangeFFT_mat = rangeFFT_mat(:, 1:(Nfft_r/2));

% 构造 range 轴（米单位）
fs = Sampling_freq;
fb = (0:(Nfft_r/2 -1)) * (fs/Nfft_r);
S = Chirp_slope;  
R_axis = c * fb / (2 * S);

% Angle FFT（阵元方向）
Nfft_ang =1024;  
virtChan = size(rangeFFT_mat,1);        % 应该是 12
win_ang = hann(virtChan);                % 12×1 列向量
angFFT_mat = fftshift( fft(rangeFFT_mat .* win_ang, Nfft_ang, 1), 1 );   % [Nfft_ang × Nfft_r/2]
% 构造角度轴 (deg)
m = (-Nfft_ang/2 : Nfft_ang/2 -1);
sin_theta = (m / Nfft_ang) * (lambda / d);
sin_theta = max(min(sin_theta, 1), -1);
theta_deg = asind(sin_theta);
RA = angFFT_mat;   % Nfft_ang × Nr

% 对于每一个 range bin，把各个角度的“平均值”减掉
RA = RA - mean(RA, 1);   % 角度维度去 DC：每一列减去自己的均值

RA_clean = RA;
RA_mag = abs(RA_clean);
% 画图
RA_dB = 20*log10( abs(RA_clean) + 1e-6 );
figure;
datacursormode on
imagesc(R_axis, theta_deg,RA_mag);
set(gca,'YDir','normal');
xlabel('Range (m)');
ylabel('Angle (°)');
title('Range‑Angle Map');
colormap jet;
colorbar;
% ===== 这里开始控制横坐标刻度 =====
maxR = 5;          % 当前RA图的最大距离
xlim([0, maxR]);             % 可选：只看 0 ~ maxR 这段

% 每 0.1 m 一个刻度（10 cm）
xticks(0 : 0.10 : round(maxR, 1));

% 刻度标签保留 1 位小数，避免太丑
xtickformat('%.1f');

% 如果想配合网格线看得更清楚：
grid on;
