clear all;
addpath('C:\Users\migue\OneDrive - uantwerpen\Documents\MATLAB\Examples\R2019b\wlan\MACFrameGenerationExample')
 
%loading packet
qos_data = load('qos_data.mat');
 
sampleRate = 20e6;
oversampling = 1.0;
 
masterClockRate = sampleRate*oversampling;
platform = 'B200';
serialNum = '310A880';
centerFrequency = 2.412e9;
cm = 1; %selected TX
%If more than numFramesInBurst frames are sent to the radio, it could cause underrun. 
%This could interfere with how packets are received. Check wireless card
%documentation.
numFramesInBurst = 10;
sdrTransmitter = comm.SDRuTransmitter('Platform', platform, 'SerialNum', serialNum, ...
  'CenterFrequency', centerFrequency, ...
  'MasterClockRate', masterClockRate, ...
  'InterpolationFactor', 1, ...
  'Gain', 8, ... % 40 dB could saturate the Tx stages, better to start with default
  'ChannelMapping', cm, ...
  "EnableBurstMode", true, ...
  "NumFramesInBurst", numFramesInBurst); 
sdrTransmitter
 
 
psdu_d = reshape(de2bi(hex2dec(qos_data.p1), 8)', [], 1);
%psdu_d = double(psdu_t)
 
% create pcap to visualize the wlan qos data frame
helperWLANExportToPCAP({qos_data.p1}, 'macFrames_p1.pcap');
 
 
%create HT object
HTconfig = wlanHTConfig;
HTconfig.GuardInterval="Long";
HTconfig.MCS=3;
HTconfig.PSDULength = length(qos_data.p1);   % Set the PSDU length
HTconfig.AggregatedMPDU=0;
 
% Generate baseband NonHT packets separated by idle time
% Better to give a large idle time to begin with and check if packets can
% all be received 
txWaveform_t = wlanWaveformGenerator(psdu_d,HTconfig, 'IdleTime',20e-6);
 
% Resample transmit waveform
txWaveform_t  = resample(txWaveform_t,sampleRate*oversampling,sampleRate);
 
fprintf('\nGenerating WLAN transmit waveform:\n')
% Scale the normalized signal to avoid saturation of RF stages
powerScaleFactor = 0.8;
txWaveform_t = txWaveform_t.*(1/max(abs(txWaveform_t))*powerScaleFactor);
% Cast the transmit signal to int16, this is the native format for the SDR
% hardware
%txWaveform_t_int = int16(txWaveform_t*2^15); % This is handled already by the HSP
 
for i=1:numFramesInBurst % only send one burst to begin with
    underrun = sdrTransmitter(txWaveform_t_int);
    if underrun~=0
         disp(['Underrun detected in packet # ', int2str(i)]);
    end
end

% Another thing you could try would be to concatenate 10 packets together
% and sent once, instead of in a for-loop, with NumFramesInBurst = 1;
 
sdrTransmitter.release()
plot(abs(txWaveform_t))