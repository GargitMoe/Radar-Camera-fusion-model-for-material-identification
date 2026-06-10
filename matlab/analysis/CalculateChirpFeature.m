% 批量处理 IWR6843 .bin 文件，提取材料特征到 CSV
% 文件名格式示例：
%   adc_data_1021_61_1_25_0.bin
%                │  │  │  └── angle (deg: 0 / 30)
%                │  │  └──── sample index
%                │  └──────  distance (cm: 25 / 50)
%                └────────── material ID (61/62/65/67)

clear
close all
clc
dbstop if error

%% ================== IWR6843 基本参数 ==================
Start_freq      = 60.000000e9;      % 60 GHz
Chirp_slope     = 29.982e6;         % 数值 29.982e6, 注释为 MHz/us
Idle_time       = 100.00e-6;        % 100 us
Tx_start_time   = 0.00e-6;          % 0 us
ADC_start_time  = 6.00e-6;          % 6 us
Num_ADCSamples  = 512;              % ADC samples per chirp
Sampling_freq   = 5000e3;           % 5 MHz
Ramp_endtime    = 60.00e-6;         % 60 us

Num_Tx          = 3;
Num_Rx          = 4;
Num_chirp       = 8;                % chirps per frame
Num_frame       = 128;

fs  = Sampling_freq;                % Hz
Ns  = Num_ADCSamples;
c   = 3e8;                          % m/s

% FMCW 斜率：MHz/us → Hz/s
% 29.982 MHz/us = 29.982e6 * 1e6 Hz/s ≈ 2.9982e13 Hz/s
S   = Chirp_slope * 1e6;            % Hz/s

% 实际读取的帧数
Num_readframe = 1;

%% ================== 预计算 Range 轴 ==================
Nfft       = Ns;
freqAxis   = (0:Nfft-1) * (fs / Nfft);  % Hz
rangeAxis  = c * freqAxis / (2 * S);    % m

validBins  = 1:floor(Nfft/2);           % 单边谱
rangeValid = rangeAxis(validBins);      % 对应距离（正频率）

%% ================== .bin 文件列表 ==================
binFiles = dir('adc_data_*.bin');
if isempty(binFiles)
    error('当前目录下没有找到 adc_data_*.bin 文件。');
end
fprintf('Found %d bin files.\n', numel(binFiles));

%% ================== 主循环：逐个 .bin 处理 ==================
for f = 1:numel(binFiles)
    filename = fullfile(binFiles(f).folder, binFiles(f).name);
    fprintf('\nProcessing file [%d/%d]: %s\n', f, numel(binFiles), filename);
    
    % ---- 解析文件名中的信息 ----
    [~, baseName, ~] = fileparts(filename);  % e.g. 'adc_data_1021_61_1_25_0'
    parts = strsplit(baseName, '_');
    if numel(parts) < 7
        warning('文件名格式异常，跳过：%s', filename);
        continue;
    end
    
    materialID  = str2double(parts{4});   % 61 / 62 / 65 / 67
    % sampleIdx = str2double(parts{5});   % 暂时不用
    dist_cm     = str2double(parts{6});   % 25 / 50
    angle_deg   = str2double(parts{7});   % 0 / 30
    
    switch materialID
        case 61
            materialName = 'metal';
        case 62
            materialName = 'glass';
        case 65
            materialName = 'plastic';
        case 67
            materialName = 'cardboard';
        otherwise
            materialName = 'unknown';
    end
    
    fprintf('  Material ID = %d (%s), dist = %d cm, angle = %d deg\n', ...
        materialID, materialName, dist_cm, angle_deg);
    
    % ---- 距离窗口随距离切换（初始手工 ROI）----
    if dist_cm == 25
        R_min = 0.25;
        R_max = 0.45;
    elseif dist_cm == 50
        R_min = 0.55;
        R_max = 0.75;
    else
        % 其他距离时，给一个默认窗口：目标前后各 0.1 m
        R_c   = dist_cm / 100;     % 中心距离（m）
        R_min = R_c - 0.10;
        R_max = R_c + 0.10;
        warning('未预设该距离 %d cm，使用自适应窗口 [%.2f, %.2f] m。', dist_cm, R_min, R_max);
    end
    
    roiMask = (rangeValid >= R_min) & (rangeValid <= R_max);
    if ~any(roiMask)
        warning('ROI 为空，检查 R_min/R_max 是否在可测范围内：%s', filename);
        continue;
    end
    rangeROI = rangeValid(roiMask);   % 初始 ROI 对应的距离轴
    
    % ---- 距离归一化使用的 R^2 因子（这里先设为 1；如需真正乘 R^2，可在后面改）----
    R2_factor_global = 1;        % 标量，占位
    
    %% ========== 读取 .bin → retVal ==========
    retVal = IWR6843_readDCA1000(filename, Num_ADCSamples, Num_Tx, Num_Rx, Num_chirp, Num_readframe);
    % retVal: [Num_Rx, Num_Tx * Num_chirp * Num_ADCSamples * Num_readframe]
    
    %% ========== 特征初始化 ==========
    % 1) peakAmpR2    : 在“ROI×所有虚通道（角度）”二维上，自适应窗口内的峰值（乘 R^2）
    % 2) meanAmpR2    : 在同一窗口内的平均值（乘 R^2）
    % 3) mainlobeSLR  : 主瓣/最大旁瓣比（用 range 方向的 1D 包络计算）
    % 4) mainlobeWidth: 主瓣宽度（-6dB，沿 range 方向，单位 m）
    % 5) envSpikiness : 时域包络毛刺程度（对所有虚通道平均后的时域信号）
    featNames     = {'peakAmpR2','meanAmpR2','mainlobeSLR','mainlobeWidth','envSpikiness'};
    featureMatrix = [];
    
    numFrames = Num_readframe;
    frame_sample_points = Num_Tx * Num_chirp * Num_ADCSamples;
    
    %% ========== 逐帧 × 每个 chirp 提特征（每个 chirp 一条样本） ==========
    for frameIdx = 1:numFrames
        idxStart = (frameIdx-1) * frame_sample_points + 1;
        idxEnd   = frameIdx    * frame_sample_points;
        frame_data = retVal(:, idxStart:idxEnd);   % [Num_Rx, frame_sample_points]
        
        % 重构 4 维：[Rx, Tx, chirp, sample]
        data4d = zeros(Num_Rx, Num_Tx, Num_chirp, Ns);
        for tx = 1:Num_Tx
            for ch = 1:Num_chirp
                s0 = ((tx-1)*Num_chirp + (ch-1)) * Ns + 1;
                s1 = s0 + Ns - 1;
                data4d(:, tx, ch, :) = frame_data(:, s0:s1);
            end
        end
        
        % ===== 每个 chirp 各算一条特征 =====
        for chirpIdx = 1:Num_chirp
            % 当前 chirp 的 [Rx, Tx, sample]
            data3d = squeeze(data4d(:, :, chirpIdx, :));   % [Num_Rx, Num_Tx, Ns]
            
            % 组合虚通道（12 个虚通道）
            virtChan_full = Num_Tx * Num_Rx;
            virtData_full = zeros(virtChan_full, Ns);      % [12 × Ns]
            vcIdx = 1;
            for tx = 1:Num_Tx
                for rx = 1:Num_Rx
                    virtData_full(vcIdx, :) = squeeze(data3d(rx, tx, :)).';
                    vcIdx = vcIdx + 1;
                end
            end
            
            % 只取方位相关的 8 个虚通道（azimuth channels）
            az_idx   = [1 2 3 4 9 10 11 12];              % 8 个通道
            virtData = virtData_full(az_idx, :);          % [8 × Ns]
            numAzChan = size(virtData, 1);
            
            %% ----- (1) 时域包络毛刺 envSpikiness -----
            % 对 8 个虚通道先求平均，得到一个“全阵列”的时域信号
            y = mean(virtData, 1);                        % 1 × Ns 复数 I/Q
            
            env    = abs(y);
            maxEnv = max(env);
            if maxEnv < eps
                envSpikiness = NaN;
            else
                envNorm  = env / maxEnv;
                diffEnv  = diff(envNorm);
                tv       = sum(abs(diffEnv));   % 总变差
                energy   = sum(envNorm) + eps;  % 总“能量”
                envSpikiness = tv / energy;
            end
            
            %% ----- (2) 频域 Range-FFT 特征（ROI × 所有虚通道 2D） -----
            % 对每个虚通道做 Range FFT
            x_win = virtData .* hann(Ns).';                 % [8 × Ns]
            Xf    = fft(x_win, Nfft, 2);                    % 沿 sample 方向 FFT，结果 [8 × Nfft]
            
            specMagFull2D = abs(Xf(:, validBins));         % [8 × N_valid]
            
            if max(specMagFull2D(:)) < eps
                % 整个 2D 区域几乎无信号
                peakAmpR2     = NaN;
                meanAmpR2     = NaN;
                mainlobeSLR   = NaN;
                mainlobeWidth = NaN;
            else
                % --- 初始 ROI 对应的 2D 谱块 ---
                specROI_raw2D = specMagFull2D(:, roiMask);   % [8 × N_roi]
                % 对应的距离轴：rangeROI 已在外层根据 roiMask 定义
                
                % 先对虚通道取 max，得到 range 方向的 1D 包络，用于找主峰
                rangeProfile = max(specROI_raw2D, [], 1);    % [1 × N_roi]
                
                % 在初始 ROI 内找最大峰及其对应距离
                [peakVal_roi, idxPeakROI] = max(rangeProfile);
                if isempty(idxPeakROI) || peakVal_roi < eps
                    peakAmpR2     = NaN;
                    meanAmpR2     = NaN;
                    mainlobeSLR   = NaN;
                    mainlobeWidth = NaN;
                else
                    % 初始 ROI 中峰值对应的物理距离
                    peakRange = rangeROI(idxPeakROI);        % m
                    
                    % 以峰值±0.15 m 作为真正的“自适应区间”
                    R_ref_min = peakRange - 0.15;
                    R_ref_max = peakRange + 0.15;
                    
                    % 在初始 ROI 内进一步做一次二次筛选
                    refMask = (rangeROI >= R_ref_min) & (rangeROI <= R_ref_max);
                    if ~any(refMask)
                        refMask = true(size(rangeROI));      % 回退到原始 ROI
                    end
                    
                    % 自适应 ROI 对应的 2D 谱块与距离
                    specROI_raw2D_ref = specROI_raw2D(:, refMask);  % [8 × N_ref]
                    rangeROI_ref      = rangeROI(refMask);          % [1 × N_ref]


%                     % 距离归一化因子（如果未来要乘 R^2，可在这里改成 rangeROI_ref.^2）
%                     R2_factor_ref = R2_factor_global * ones(1, numel(rangeROI_ref));  % 1×N_ref
%                     
%                     % 在自适应 ROI 上做 R^2 加权
%                     specROI_R2_2D = specROI_raw2D_ref .* R2_factor_ref;   % [8 × N_ref]
                    
                    % (a) 自适应 ROI 内距离归一化后的峰值和平均值（2D 上统计）
                    R_peak2   = peakRange^1.5;                          % 主峰对应距离的 R^2
                    spec2D    = specROI_raw2D_ref;                    % 原始 2D 谱块
                    peakAmpR2 = max(spec2D(:)) * R_peak2;
                    meanAmpR2 = mean(spec2D(:)) * R_peak2;
                    
                    % (b) 主瓣/最大旁瓣比：在自适应 ROI 内，对 rangeProfile 再截一次
                    rangeProfile_ref = max(specROI_raw2D_ref, [], 1);     % [1 × N_ref]
                    sortedROI = sort(rangeProfile_ref, 'descend');
                    if numel(sortedROI) >= 2
                        mainlobeSLR = sortedROI(1) / (sortedROI(2) + eps);
                    else
                        mainlobeSLR = NaN;
                    end
                    
                    % (c) 主瓣宽度（-6 dB，沿 range 方向）
                    peakVal_raw = sortedROI(1);
                    if peakVal_raw < eps
                        mainlobeWidth = NaN;
                    else
                        thr   = 0.5 * peakVal_raw;                    % -6 dB 对应振幅一半
                        above = rangeProfile_ref >= thr;              % [1 × N_ref]
                        peakIdxInROI2 = find(rangeProfile_ref == peakVal_raw, 1, 'first');
                        
                        if isempty(peakIdxInROI2) || ~above(peakIdxInROI2)
                            mainlobeWidth = NaN;
                        else
                            left  = peakIdxInROI2;
                            while left > 1 && above(left-1)
                                left = left - 1;
                            end
                            right = peakIdxInROI2;
                            while right < numel(above) && above(right+1)
                                right = right + 1;
                            end
                            
                            R_left  = rangeROI_ref(left);
                            R_right = rangeROI_ref(right);
                            mainlobeWidth = R_right - R_left;   % m
                        end
                    end
                end
            end
            
            % 汇总此样本（frameIdx × chirpIdx）
            featVec = [peakAmpR2, meanAmpR2, mainlobeSLR, mainlobeWidth, envSpikiness];
            featureMatrix = [featureMatrix; featVec];
        end
    end
    
    %% ========== 组装表并写 CSV ==========
    featureTable = array2table(featureMatrix, 'VariableNames', featNames);
    
    nSamples = size(featureMatrix, 1);   % 对当前 .bin，一般 = Num_readframe * Num_chirp
    featureTable.materialID   = repmat(materialID,        nSamples, 1);
    featureTable.materialName = repmat({materialName},    nSamples, 1);
    featureTable.dist_cm      = repmat(dist_cm,           nSamples, 1);
    featureTable.angle_deg    = repmat(angle_deg,         nSamples, 1);
    
    outFile = fullfile(binFiles(f).folder, [baseName, '_features_chirp.csv']);
    writetable(featureTable, outFile);
    
    fprintf('  -> %s written. (%d samples × %d columns)\n', ...
        outFile, size(featureTable,1), width(featureTable));
end

fprintf('\nAll files processed.\n');
