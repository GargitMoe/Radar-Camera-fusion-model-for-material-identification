% 6843ISK mmWave Board Save Binary File to Channel Data
% Transmiter: 3 --> Receiver: 4
% -------------------------------------------------------------------------
% Developed by:
% Jiangyou Zhu
%
% -------------------------------------------------------------------------
%  Welcome to the Wireless Networking and Sensing (WiNS) Group @ CUHK
% -------------------------------------------------------------------------

clear
close all
clc
dbstop if error


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
filename = "adc_data_1021_61_1_25_0.bin";
retVal = IWR6843_readDCA1000(filename, Num_ADCSamples, Num_Tx, Num_Rx, Num_chirp, Num_readframe);

%% Tx-Rx pair

numChirps = Num_Tx*Num_readframe*Num_chirp;
adcData = zeros(Num_Rx,numChirps*Num_ADCSamples);
T1R1 = zeros(Num_readframe, Num_ADCSamples);
T1R2 = zeros(Num_readframe, Num_ADCSamples);
T1R3 = zeros(Num_readframe, Num_ADCSamples);
T1R4 = zeros(Num_readframe, Num_ADCSamples);
T2R1 = zeros(Num_readframe, Num_ADCSamples);
T2R2 = zeros(Num_readframe, Num_ADCSamples);
T2R3 = zeros(Num_readframe, Num_ADCSamples);
T2R4 = zeros(Num_readframe, Num_ADCSamples);
T3R1 = zeros(Num_readframe, Num_ADCSamples);
T3R2 = zeros(Num_readframe, Num_ADCSamples);
T3R3 = zeros(Num_readframe, Num_ADCSamples);
T3R4 = zeros(Num_readframe, Num_ADCSamples);

frame_sample_points = Num_Tx*Num_chirp*Num_ADCSamples;
chirp_sample_points = Num_Tx*1*Num_ADCSamples;
for frame_index = 1:Num_readframe
    chirp_index = 1;
    frame_data = retVal(:, frame_sample_points*(frame_index-1)+1: frame_sample_points*(frame_index-1)+frame_sample_points);
    chirp_data = frame_data(:, chirp_sample_points*(chirp_index-1)+1: chirp_sample_points*(chirp_index-1)+chirp_sample_points);
    
    % This part we use 3 Transmitter and 4 Receiver
    T1R1(frame_index,:) = chirp_data(1, Num_ADCSamples*(1-1)+1: Num_ADCSamples*(1-1)+Num_ADCSamples);
    T1R2(frame_index,:) = chirp_data(2, Num_ADCSamples*(1-1)+1: Num_ADCSamples*(1-1)+Num_ADCSamples);
    T1R3(frame_index,:) = chirp_data(3, Num_ADCSamples*(1-1)+1: Num_ADCSamples*(1-1)+Num_ADCSamples);
    T1R4(frame_index,:) = chirp_data(4, Num_ADCSamples*(1-1)+1: Num_ADCSamples*(1-1)+Num_ADCSamples);

    T3R1(frame_index,:) = chirp_data(1, Num_ADCSamples*(2-1)+1: Num_ADCSamples*(2-1)+Num_ADCSamples);
    T3R2(frame_index,:) = chirp_data(2, Num_ADCSamples*(2-1)+1: Num_ADCSamples*(2-1)+Num_ADCSamples);
    T3R3(frame_index,:) = chirp_data(3, Num_ADCSamples*(2-1)+1: Num_ADCSamples*(2-1)+Num_ADCSamples);
    T3R4(frame_index,:) = chirp_data(4, Num_ADCSamples*(2-1)+1: Num_ADCSamples*(2-1)+Num_ADCSamples);

    T2R1(frame_index,:) = chirp_data(1, Num_ADCSamples*(3-1)+1: Num_ADCSamples*(3-1)+Num_ADCSamples);
    T2R2(frame_index,:) = chirp_data(2, Num_ADCSamples*(3-1)+1: Num_ADCSamples*(3-1)+Num_ADCSamples);
    T2R3(frame_index,:) = chirp_data(3, Num_ADCSamples*(3-1)+1: Num_ADCSamples*(3-1)+Num_ADCSamples);
    T2R4(frame_index,:) = chirp_data(4, Num_ADCSamples*(3-1)+1: Num_ADCSamples*(3-1)+Num_ADCSamples);

end

%% Show the results
t = [0:1:(Num_ADCSamples-1)]./Sampling_freq;

figure,
subplot(221)
plot(t, real(T1R1(frame_index,:)), 'b')
hold on
plot(t, imag(T1R1(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx1-Rx1'])
xlim([t(1), t(end)])

subplot(222)
plot(t, real(T1R2(frame_index,:)), 'b')
hold on
plot(t, imag(T1R2(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx1-Rx2'])
xlim([t(1), t(end)])

subplot(223)
plot(t, real(T1R3(frame_index,:)), 'b')
hold on
plot(t, imag(T1R3(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx1-Rx3'])
xlim([t(1), t(end)])

subplot(224)
plot(t, real(T1R4(frame_index,:)), 'b')
hold on
plot(t, imag(T1R4(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx1-Rx4'])
xlim([t(1), t(end)])

figure,
subplot(221)
plot(t, real(T3R1(frame_index,:)), 'b')
hold on
plot(t, imag(T3R1(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx3-Rx1'])
xlim([t(1), t(end)])

subplot(222)
plot(t, real(T3R2(frame_index,:)), 'b')
hold on
plot(t, imag(T3R2(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx3-Rx2'])
xlim([t(1), t(end)])

subplot(223)
plot(t, real(T3R3(frame_index,:)), 'b')
hold on
plot(t, imag(T3R3(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx3-Rx3'])
xlim([t(1), t(end)])

subplot(224)
plot(t, real(T3R4(frame_index,:)), 'b')
hold on
plot(t, imag(T3R4(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx3-Rx4'])
xlim([t(1), t(end)])

figure,
subplot(221)
plot(t, real(T2R1(frame_index,:)), 'b')
hold on
plot(t, imag(T2R1(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx2-Rx1'])
xlim([t(1), t(end)])

subplot(222)
plot(t, real(T2R2(frame_index,:)), 'b')
hold on
plot(t, imag(T2R2(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx2-Rx2'])
xlim([t(1), t(end)])

subplot(223)
plot(t, real(T2R3(frame_index,:)), 'b')
hold on
plot(t, imag(T2R3(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx2-Rx3'])
xlim([t(1), t(end)])

subplot(224)
plot(t, real(T2R4(frame_index,:)), 'b')
hold on
plot(t, imag(T2R4(frame_index,:)), 'r')
grid on
title(['Franm ID: ', num2str(frame_index), ' Tx2-Rx4'])
xlim([t(1), t(end)])
%% === Feature example: Early / Late energy ratio for Tx1 ===
w = 10;   % 窗口半宽，可以根据需要调 5~20

ER_Tx1 = zeros(1,4);  % 存 T1R1~T1R4 的能量比

for rx = 1:4
    % 取出当前 Rx 通道的一帧数据（Tx1）
    if rx == 1
        x = squeeze(T1R1(frame_index,:));
    elseif rx == 2
        x = squeeze(T1R2(frame_index,:));
    elseif rx == 3
        x = squeeze(T1R3(frame_index,:));
    else
        x = squeeze(T1R4(frame_index,:));
    end

    % 1) 幅度
    mag = abs(x);

    % 2) 找主峰位置
    [~, peak_idx] = max(mag);

    % 3) Early window（主反射）
    idx_early_start = max(1, peak_idx - w);
    idx_early_end   = min(Num_ADCSamples, peak_idx + w);
    early = mag(idx_early_start:idx_early_end);

    % 4) Late window（尾部散射）
    late_start = min(Num_ADCSamples, idx_early_end + 1);
    late = mag(late_start:end);

    % 5) 能量
    E_early = sum(early.^2);
    E_late  = sum(late.^2);

    % 6) 特征：尾部能量 / 主反射能量
    ER_Tx1(rx) = E_late / (E_early + eps);
end

disp('Early/Late energy ratio for Tx1 (R1~R4):');
disp(ER_Tx1);
