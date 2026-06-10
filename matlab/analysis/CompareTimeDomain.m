%% 
% 6843ISK mmWave Board - Compare Time-domain Waveforms of 4 Materials
% Transmiter: 3 --> Receiver: 4
% -------------------------------------------------------------------------
% Developed by: Moey + ChatGPT (基于 WiNS 原始示例改写)
% -------------------------------------------------------------------------
%  Goal:
%   在同一个 Tx-Rx 通道内，对比 4 个材料 (61,62,65,67) 的时域波形
%   - 只看 Tx1 和 Tx3
%   - 每个 Rx 单独画一张图，图中叠加 4 种材料
% -------------------------------------------------------------------------

clear;
close all;
clc;
dbstop if error;

%% IWR6843 FMCW Basic Parameter (和原脚本一致)
% Sensor Configuration
Start_freq        = 60.000000e9;     % 60GHz
Chirp_slope       = 29.982e6;        % 29.982 MHz/us
Idle_time         = 100.00e-6;       % 100 us
Tx_start_time     = 0.00e-6;         % 0 us
ADC_start_time    = 6.00e-6;         % 6 us
Num_ADCSamples    = 512;             % number of ADC samples per chirp
Sampling_freq     = 10e6;            % 10 MHz
Ramp_endtime      = 60.00e-6;        % 60 us

% Chirp Configuration
Num_Tx            = 3;               % number of transmitter
Num_Rx            = 4;               % number of receivers
Num_chirp         = 8;               % number of chirp in one frame
Chirp_config_num  = Num_Tx;
Chirp_cycle_time  = Idle_time + Ramp_endtime;

% Frame Configuration
Num_frame         = 128;
Frame_periodcity  = 62.000000e-3;    % 62ms
Duty_cycle        = ((Chirp_config_num*Chirp_cycle_time)*Num_chirp/Frame_periodcity)*100;

% Time axis for one chirp
t = (0:Num_ADCSamples-1)./Sampling_freq;   % [s]

%% 材料与文件名设置
materialIDs   = [61 62 65 67];  % 四种材料的编号
numMaterials  = numel(materialIDs);

fileList = arrayfun(@(id) sprintf("adc_data_1021_%d_1_25_30.bin", id), ...
                    materialIDs, "UniformOutput", true);

legendLabels = arrayfun(@(id) sprintf("ID%d", id), materialIDs, ...
                        "UniformOutput", false);

%% 预分配：保存 Tx1 / Tx3 在各 Rx 下的一个 chirp 波形
% 维度: [numMaterials, Num_Rx, Num_ADCSamples]
wave_Tx1 = zeros(numMaterials, Num_Rx, Num_ADCSamples);
wave_Tx3 = zeros(numMaterials, Num_Rx, Num_ADCSamples);

%% 逐材料读取 .bin 并提取 Tx1 / Tx3 的第 1 chirp 波形
Num_readframe = 1;   % 只看第 1 帧

frame_sample_points = Num_Tx * Num_chirp * Num_ADCSamples;   % 一帧总采样点数 (每 Rx)
chirp_sample_points = Num_Tx * Num_ADCSamples;               % 一条 chirp 的采样点数 (所有 Tx)

for m = 1:numMaterials
    filename = fileList(m);
    fprintf('Reading file: %s\n', filename);

    % 这里假设 IWR6843_readDCA1000 已经在路径中
    retVal = IWR6843_readDCA1000(filename, ...
                                 Num_ADCSamples, Num_Tx, Num_Rx, ...
                                 Num_chirp, Num_readframe);
    % retVal 尺寸: [Num_Rx, Num_Tx * Num_chirp * Num_ADCSamples * Num_readframe]

    % 只取第 1 帧
    frame_index = 1;
    frame_data = retVal(:, frame_sample_points*(frame_index-1)+1 : ...
                            frame_sample_points*(frame_index-1)+frame_sample_points);
    
    % 只取第 1 条 chirp (对应原脚本的 chirp_index = 1)
    chirp_index = 1;
    chirp_data = frame_data(:, chirp_sample_points*(chirp_index-1)+1 : ...
                                chirp_sample_points*(chirp_index-1)+chirp_sample_points);
    % 现在 chirp_data 尺寸: [Num_Rx, Num_Tx * Num_ADCSamples]
    % 按照原脚本的约定：
    %   block1: Tx1, block2: Tx3, block3: Tx2
    %   每个 block 的长度为 Num_ADCSamples

    for rx = 1:Num_Rx
        % Tx1
        wave_Tx1(m, rx, :) = chirp_data(rx, 1 : Num_ADCSamples);
        % Tx3
        wave_Tx3(m, rx, :) = chirp_data(rx, Num_ADCSamples+1 : 2*Num_ADCSamples);
        % 如果之后你想用 Tx2，也可以类似地取第三个 block:
        % wave_Tx2(m, rx, :) = chirp_data(rx, 2*Num_ADCSamples+1 : 3*Num_ADCSamples);
    end
end

%% 画图：Tx1 各 Rx，四种材料的时域波形对比
% 这里用 |s(t)| （幅度）对比，如果你想看实部，改成 real(sig)；
% 想看虚部，改成 imag(sig)。

figure('Name','Tx1 - Time-domain comparison of 4 materials');
for rx = 1:Num_Rx
    subplot(2,2,rx);
    hold on; grid on;

    for m = 1:numMaterials
        sig = squeeze(wave_Tx1(m, rx, :)).';   % 1 x Num_ADCSamples
        plot(t, abs(sig));                     % 对比幅度 |s(t)|
        % 如果你想看实部：
        % plot(t, real(sig));
        % 或者同时画实部/虚部，可以再 plot 一条 imag(sig)，不过会比较乱
    end

    title(sprintf('Tx1 - Rx%d', rx));
    xlabel('Time (s)');
    ylabel('|s(t)|');
    xlim([t(1), t(end)]);
    if rx == 1
        legend(legendLabels, 'Location','best');
    end
end

%% 画图：Tx3 各 Rx，四种材料的时域波形对比
figure('Name','Tx3 - Time-domain comparison of 4 materials');
for rx = 1:Num_Rx
    subplot(2,2,rx);
    hold on; grid on;

    for m = 1:numMaterials
        sig = squeeze(wave_Tx3(m, rx, :)).';
        plot(t, abs(sig));   % 幅度
    end

    title(sprintf('Tx3 - Rx%d', rx));
    xlabel('Time (s)');
    ylabel('|s(t)|');
    xlim([t(1), t(end)]);
    if rx == 1
        legend(legendLabels, 'Location','best');
    end
end
