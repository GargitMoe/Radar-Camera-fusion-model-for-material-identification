clear
close all
clc
dbstop if error

%% IWR6843 FMCW Basic Parameter
Start_freq      = 60.000000e9; % 60GHz
Chirp_slope     = 29.982e6;    % 29.982MHz/us
Idle_time       = 100.00e-6;   % 100us
Tx_start_time   = 0.00e-6;     % 0us
ADC_start_time  = 6.00e-6;     % 6us
Num_ADCSamples  = 512;         % number of ADC samples per chirp
Sampling_freq   = 5000e3;      % Hz （你的原注释写 kHz，这里实际是 5e6 Hz）
Ramp_endtime    = 60.00e-6;    % 60us

Num_Tx          = 3;
Num_Rx          = 4;
Num_frame       = 128;
Num_chirp       = 8;
Chirp_config_num= Num_Tx;
Chirp_cycle_time= Idle_time + Ramp_endtime;

Frame_periodcity= 62.000000e-3; % 62ms
Duty_cycle      = ((Chirp_config_num*Chirp_cycle_time)*Num_chirp/Frame_periodcity)*100;

Num_readframe   = 1; % 先跟你原来一样，只读 1 帧

%% 批量遍历所有 adc_data*.bin 文件
fileList = dir('adc_data*.bin');

if isempty(fileList)
    error('当前文件夹下没有匹配 adc_data*.bin 的文件');
end

% 用一个 struct 数组存所有结果，最后转 table
results = struct('filename',{}, 'frame',{}, 'tx',{}, 'rx',{}, 'ER_earlyLate',{});

rowIdx = 0;

for f = 1:numel(fileList)
    filename = fileList(f).name;
    fprintf('Processing file: %s\n', filename);

    % 读取该文件的数据
    retVal = IWR6843_readDCA1000(filename, Num_ADCSamples, Num_Tx, Num_Rx, Num_chirp, Num_readframe);

    % 预分配 Tx-Rx 数据（这里只演示 Tx1，其他 Tx 你可以按需加）
    T1R1 = zeros(Num_readframe, Num_ADCSamples);
    T1R2 = zeros(Num_readframe, Num_ADCSamples);
    T1R3 = zeros(Num_readframe, Num_ADCSamples);
    T1R4 = zeros(Num_readframe, Num_ADCSamples);

    frame_sample_points = Num_Tx*Num_chirp*Num_ADCSamples;
    chirp_sample_points = Num_Tx*1*Num_ADCSamples;

    for frame_index = 1:Num_readframe
        chirp_index = 1; % 你原来就是取第一个 chirp

        frame_data = retVal(:, frame_sample_points*(frame_index-1)+1 : ...
                                 frame_sample_points*(frame_index-1)+frame_sample_points);
        chirp_data = frame_data(:, chirp_sample_points*(chirp_index-1)+1 : ...
                                     chirp_sample_points*(chirp_index-1)+chirp_sample_points);

        % Tx1, 4 Rx
        T1R1(frame_index,:) = chirp_data(1, Num_ADCSamples*(1-1)+1 : Num_ADCSamples*(1-1)+Num_ADCSamples);
        T1R2(frame_index,:) = chirp_data(2, Num_ADCSamples*(1-1)+1 : Num_ADCSamples*(1-1)+Num_ADCSamples);
        T1R3(frame_index,:) = chirp_data(3, Num_ADCSamples*(1-1)+1 : Num_ADCSamples*(1-1)+Num_ADCSamples);
        T1R4(frame_index,:) = chirp_data(4, Num_ADCSamples*(1-1)+1 : Num_ADCSamples*(1-1)+Num_ADCSamples);

        %% === 对 Tx1 的 4 个 Rx 通道计算 early/late 能量比 ===
        w = 10;                 % 主峰左右窗口半宽，按需要调节
        ER_Tx1 = zeros(1,4);    % 存 T1R1~T1R4 的结果

        for rx = 1:4
            if rx == 1
                x = squeeze(T1R1(frame_index,:));
            elseif rx == 2
                x = squeeze(T1R2(frame_index,:));
            elseif rx == 3
                x = squeeze(T1R3(frame_index,:));
            else
                x = squeeze(T1R4(frame_index,:));
            end

            mag = abs(x);
            [~, peak_idx] = max(mag);

            idx_early_start = max(1, peak_idx - w);
            idx_early_end   = min(Num_ADCSamples, peak_idx + w);
            early = mag(idx_early_start:idx_early_end);

            late_start = min(Num_ADCSamples, idx_early_end + 1);
            late = mag(late_start:end);

            E_early = sum(early.^2);
            E_late  = sum(late.^2);

            ER_Tx1(rx) = E_late / (E_early + eps);

            % 写入结果表
            rowIdx = rowIdx + 1;
            results(rowIdx).filename      = filename;
            results(rowIdx).frame         = frame_index;
            results(rowIdx).tx            = 1;      % Tx1
            results(rowIdx).rx            = rx;     % Rx1~4
            results(rowIdx).ER_earlyLate  = ER_Tx1(rx);
        end

        % 你要的话，这里也可以顺便画图对比，不影响结果保存
        % 比如简单看一下一个通道的时域波形：
        %{
        if f==1 && frame_index==1
            t = (0:Num_ADCSamples-1) ./ Sampling_freq;
            figure;
            plot(t, real(T1R1(frame_index,:))); hold on;
            plot(t, imag(T1R1(frame_index,:)));
            title(sprintf('File: %s, Tx1-Rx1, frame %d', filename, frame_index));
            grid on;
        end
        %}
    end
end

%% 所有文件处理完，存成 CSV
T = struct2table(results);
writetable(T, 'features_Tx1_ER.csv');
fprintf('Done. Features saved to features_Tx1_ER.csv\n');

T