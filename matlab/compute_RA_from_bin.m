function [RA_mag_all, theta_deg, R_axis] = compute_RA_from_bin(filename, params)
    % params: Num_ADCSamples, Num_Tx, Num_Rx, Num_chirp, Num_readframe, fc, Sampling_freq, Chirp_slope

    Num_ADCSamples = params.Num_ADCSamples;
    Num_Tx         = params.Num_Tx;
    Num_Rx         = params.Num_Rx;
    Num_chirp      = params.Num_chirp;
    Num_readframe  = params.Num_readframe;

    % ==================== 读 ADC 原始数据 ====================
    retVal = IWR6843_readDCA1000(filename, Num_ADCSamples, ...
                                 Num_Tx, Num_Rx, Num_chirp, Num_readframe);

    % ---------------- 构造 data4d: [Rx, Tx, chirp, sample] ----------------
    Ntx    = Num_Tx;
    Nrx    = Num_Rx;
    Nchirp = Num_chirp;
    Ns     = Num_ADCSamples;

    numChirps = Ntx * Nchirp * Num_readframe;
    frame_data = retVal(:, 1 : (numChirps * Ns));  % 只用第一帧

    data4d = zeros(Nrx, Ntx, Nchirp, Ns);
    for tx = 1:Ntx
        for ch = 1:Nchirp
            idx0 = ((tx-1)*Nchirp + (ch-1))*Ns + 1;
            idx1 = idx0 + Ns - 1;
            data4d(:, tx, ch, :) = frame_data(:, idx0:idx1);
        end
    end

    % ==================== 公共物理参数 ====================
    c      = 3e8;
    fc     = params.fc;      % 例如 60.9e9
    lambda = c / fc;
    d      = lambda / 2;

    fs = params.Sampling_freq;   % 5e6
    S  = params.Chirp_slope;     % 29.982e12

    % ---- Range FFT 参数 ----
    win_r   = hann(Ns).';
    Nfft_r  = Ns;
    fb      = (0:(Nfft_r/2 - 1)) * (fs / Nfft_r);
    R_axis  = c * fb / (2 * S);

    % ---- Angle FFT 参数 ----
    Nfft_ang   = 1024;

    % 虚阵列索引：你原来用的是前4 + 后4
    az_idx       = [1 2 3 4 9 10 11 12];
    Nvirt_used   = numel(az_idx);
    win_ang      = hann(Nvirt_used);

    % 角度轴
    m         = (-Nfft_ang/2 : Nfft_ang/2 - 1);
    sin_theta = (m / Nfft_ang) * (lambda / d);
    sin_theta = max(min(sin_theta, 1), -1);
    theta_deg = asind(sin_theta);

    Nr = numel(fb)/1;  % Nfft_r/2

    % 预分配：每个 chirp 一张 RA 幅度图
    RA_mag_all = zeros(Nchirp, Nfft_ang, Nfft_r/2);   % [chirp, angle, range]

    % ==================== 核心：对每个 chirp 单独算 RA ====================
    for ch = 1:Nchirp
        % 构造该 chirp 的虚阵列数据 [virtChan=12, Ns]
        virtChan = Ntx * Nrx;
        virtData = zeros(virtChan, Ns);
        vcIdx = 1;
        for tx = 1:Ntx
            for rx = 1:Nrx
                tmp_ch = squeeze(data4d(rx, tx, ch, :)).';   % [1 × Ns]
                virtData(vcIdx, :) = tmp_ch;
                vcIdx = vcIdx + 1;
            end
        end

        % 选取角度向的 8 个虚阵元
        virtData_az = virtData(az_idx, :);   % [Nvirt_used × Ns]

        % ---- Range FFT ----
        rangeFFT_mat = fft(virtData_az .* win_r, Nfft_r, 2);   % 行方向 FFT
        rangeFFT_mat = rangeFFT_mat(:, 1:(Nfft_r/2));          % 正频

        % ---- Angle FFT ----
        angFFT_mat = fftshift( fft(rangeFFT_mat .* win_ang, Nfft_ang, 1), 1 );
        % angFFT_mat: [Nfft_ang × (Nfft_r/2)]

        % 去 DC
        RA = angFFT_mat - mean(angFFT_mat, 1);

        % 幅度
        RA_mag = abs(RA);       % [Nfft_ang × Nfft_r/2]

        RA_mag_all(ch, :, :) = RA_mag;
    end
end
