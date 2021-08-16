%loading packet
qos_data = load('qos_data.mat').;

%configuring sdr -> Not clear why the oversampling but follows example from
%https://nl.mathworks.com/help/wlan/examples/image-transmission-and-reception-using-wlan-toolbox-and-one-plutosdr.html
%which works well. 

sampleRate = 20e6;
oversampling = 1.5;
masterClockRate = sampleRate*oversampling;
platform = 'B200';
serialNum = '310A880';
centerFrequency = 2.472e9;
cm = 1; %selected TX
sdrTransmitter = comm.SDRuTransmitter('Platform', platform, 'SerialNum', serialNum, ...
  'CenterFrequency', centerFrequency, ...
  'MasterClockRate', masterClockRate, ...
  'InterpolationFactor', 1, ...
  'Gain', 40, ...
  'ChannelMapping', cm, ...
  'UnderrunOutputPort',  true);

psdu_d = reshape(de2bi(hex2dec(qos_data), 8)', [], 1);
%psdu_d = double(psdu_t)

% create pcap to visualize the wlan qos data frame
helperWLANExportToPCAP({qos_data}, 'macFrames_p1.pcap');


%create HT object
HTconfig = wlanHTConfig;
HTconfig.GuardInterval="Short";
HTconfig.MCS=6;
HTconfig.ChannelCoding="LDPC";
HTconfig.PSDULength = length(qos_data);   % Set the PSDU length
HTconfig.AggregatedMPDU=0;

% Initialize the scrambler with a random integer for each packet
scramblerInitialization = randi([1 127],1,1);

% Generate baseband NonHT packets separated by idle time
txWaveform_t = wlanWaveformGenerator(psdu_d,HTconfig, ...
    'NumPackets',1,'IdleTime',0, ...
    'ScramblerInitialization',scramblerInitialization, ...
    "WindowTransitionTime", 0);

% Resample transmit waveform
txWaveform_t  = resample(txWaveform_t,sampleRate*oversampling,sampleRate);

fprintf('\nGenerating WLAN transmit waveform:\n')
% Scale the normalized signal to avoid saturation of RF stages
powerScaleFactor = 0.9;
txWaveform_t = txWaveform_t.*(1/max(abs(txWaveform_t))*powerScaleFactor);
% Cast the transmit signal to int16, this is the native format for the SDR
% hardware
txWaveform_t_int = int16(txWaveform_t*2^15);

%transmitting
sdrTransmitter(txWaveform_t_int)
sdrTransmitter.release()
plot(abs(txWaveform_t))