% Function of Read DCA1000 Binary Data
% -------------------------------------------------------------------------
% Developed by:
% Jiangyou Zhu
%
% -------------------------------------------------------------------------
%  Welcome to the Wireless Networking and Sensing (WiNS) Group @ CUHK
% -------------------------------------------------------------------------

function [retVal] = IWR6843_readDCA1000(filename, Num_ADCSamples, Num_Tx, Num_Rx, Num_chirp, Num_readframe)
%% global variables
% change based on sensor config
numADCBits = 16; % number of ADC bits per sample
% numLanes = 2; % do not change. number of lanes is always 2
% isReal = 0; % set to 1 if real only data, 0 if complex data0
%% read file
% read .bin file
fid = fopen(filename,'r');
raw_adcData = fread(fid, 'int16');
% if 12 or 14 bits ADC per sample compensate for sign extension
if numADCBits ~= 16
    l_max = 2^(numADCBits-1)-1;
    raw_adcData(raw_adcData > l_max) = raw_adcData(raw_adcData > l_max) - 2^numADCBits;
end
fclose(fid);

read_fileSize = 2*Num_Rx*((Num_Tx*Num_chirp*Num_readframe)*Num_ADCSamples); % 2 means complex data
adcData = raw_adcData(1:read_fileSize);

%% for complex data
% filesize = 2 * numADCSamples*numChirps
numChirps = Num_Tx*Num_chirp*Num_readframe;
LVDS = zeros(1, read_fileSize/2);
%combine real and imaginary part into complex data
%read in file: 2I is followed by 2Q
counter = 1;
for k = 1: 4: read_fileSize-1
    LVDS(1,counter) = adcData(k) + sqrt(-1)*adcData(k+2);
    LVDS(1,counter+1) = adcData(k+1)+sqrt(-1)*adcData(k+3);
    counter = counter + 2;
end
% create column for each chirp
LVDS = reshape(LVDS, Num_ADCSamples*Num_Rx, numChirps);
%each row is data from one chirp
LVDS = LVDS.';



%organize data per RX
adcData = zeros(Num_Rx, numChirps*Num_ADCSamples);
for row = 1:Num_Rx
    for k = 1: numChirps
        adcData(row, (k-1)*Num_ADCSamples+1:k*Num_ADCSamples) = LVDS(k, (row-1)*Num_ADCSamples+1:row*Num_ADCSamples);
    end
end
% return receiver data
retVal = adcData;