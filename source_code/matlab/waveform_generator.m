clear all
dir = "../../dataset/mat_from_pcap/";
filename = "07082020_2G_n_mobile_L8";
mat_to_waveform = dir+filename+".mat";
load(mat_to_waveform);

use_SNR = false;

if use_SNR
    filename_prefix = "_SNR_";
    fs = 20e6; % Channel model sampling frequency equals the channel bandwidth
    SNR = randi([20 30],1,num_packets);
else
    filename_prefix = "";
end

tic
X = cell(size(labels));
X_payload = cell(size(labels));
Y = cell(size(labels));
phy_configs = cell(size(labels));
num_packets = size(labels,2);
packets_b = 0;
packets_g = 0;
packets_n = 0;
packets_video = 0;
packets_audio = 0;
packets_no_type = 0;
packets_unknown = 0;

p = ProgressBar(num_packets);
disp('Starting waveform generations')
    
parfor i=1:num_packets
     
    phy_type = labels{i}.phy;
    p.progress; % Also percent = p.progress;
    
    %802.11b
    if phy_type == 0
        phyConfig = wlanNonHTConfig;
        phyConfig.Modulation=labels{i}.modulation;
        phyConfig.ChannelBandwidth=labels{i}.bw;
        phyConfig.DataRate=labels{i}.DataRate;
        phyConfig.Preamble=labels{i}.Preamble;
        phyConfig.PSDULength=labels{i}.psduLength;
        packets_b = packets_b+1;
    
    %802.11g
    elseif phy_type == 1
        phyConfig = wlanNonHTConfig;
        phyConfig.Modulation=labels{i}.modulation;
        phyConfig.ChannelBandwidth=labels{i}.bw;
        phyConfig.MCS=double(labels{i}.mcs);
        phyConfig.PSDULength=labels{i}.psduLength;
        packets_g = packets_g+1;
    
    %802.11n
    elseif phy_type == 2
        phyConfig = wlanHTConfig;
        phyConfig.ChannelBandwidth=labels{i}.bw;
        phyConfig.MCS=mod(labels{i}.mcs,8);
        phyConfig.MCS=labels{i}.mcs;
        phyConfig.GuardInterval=labels{i}.guard_interval;
        phyConfig.ChannelCoding=labels{i}.FEC_type;
        phyConfig.PSDULength=labels{i}.psduLength;
        phyConfig.AggregatedMPDU=labels{i}.AMPDU;
        packets_n = packets_n+1;
    else
        disp('Phy type error')
    end

    psdu_bytes = hex2dec(frames{i});
    X_payload{i}=uint8(psdu_bytes');
    
    psdu_d = reshape(de2bi(psdu_bytes, 8)', [], 1);
    
    txWaveform = wlanWaveformGenerator(psdu_d,phyConfig);
    %fprintf('\nGenerating WLAN transmit waveform:\n')
    
    %txWaveform_original = txWaveform;
    %powerScaleFactor_2 = 1;
    %txWaveform_original = txWaveform_original.*(1/max(abs(txWaveform_original))*powerScaleFactor_2);
    %txWaveform_original = int16(txWaveform_original*2^15);
    % Scale and normalize the signal
    if use_SNR
        tgnChan = wlanTGnChannel('SampleRate',fs,'LargeScaleFadingEffect', 'Pathloss and shadowing','DelayProfile','Model-B');
        txWaveform = awgn(tgnChan(txWaveform),SNR(i),'measured');
    end
    powerScaleFactor = 1;
    txWaveform = txWaveform.*(1/max(abs(txWaveform))*powerScaleFactor);
    txWaveform = int16(txWaveform*2^15);
    %break;
    %save the waveform and the label for this classification task
    %x_temp = [real(txWaveform)',imag(txWaveform)'];
    X{i} = [real(txWaveform)',imag(txWaveform)'];
    %[d0,d1]=size(x_temp);
    %X{i} = reshape(x_temp,[d1/2,2]);
                        
    if labels{i}.app_type == 0
        packets_audio=packets_audio+1;
    elseif labels{i}.app_type == 1
        packets_video=packets_video+1;
    elseif labels{i}.app_type == 2
        packets_no_type=packets_no_type+1;
    else
        packets_unknown=packets_unknown+1;
    end    
    
    if use_SNR
        Y{i} = uint8([labels{i}.frame_type, labels{i}.phy, labels{i}.app_type, labels{i}.app, SNR(i)]); 
    else
        Y{i} = uint8([labels{i}.frame_type, labels{i}.phy, labels{i}.app_type, labels{i}.app]);   
    end
%}
end
p.stop; % Also percent = p.stop;
t_packets = packets_b+packets_g+packets_n;
disp(t_packets);
disp(packets_b);
disp(packets_g);
disp(packets_n);
disp(packets_audio);
disp(packets_video);
disp(packets_no_type);
disp(packets_unknown)
t=toc;
dir_out = "../../dataset/waveforms/";
waveforms_file = dir_out+"waveforms_"+filename+filename_prefix+".mat"
save(waveforms_file,'X','X_payload','Y', '-v7.3')