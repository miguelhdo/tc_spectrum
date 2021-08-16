bits_per_byte = 8;

macConfig = wlanMACFrameConfig('FrameType','QoS Data','FrameFormat','HT-Mixed', ...
                                       'MPDUAggregation',true);
phyConfig = wlanHTConfig('MCS',7);
 
% Calculate maximum PSDU length
max_ppdu_time = 5484; %micro-seconds
max_pdsu_bytes = 65535; %802.11n
max_psdu_given_mcs = wlanPSDULength(cfgPHY,'TxTime',max_ppdu_time);
msduLengths = wlanMSDULengths(max_psdu_given_mcs, macConfig, phyConfig);
max_ampdu = sum(msduLengths);

%For max_psdu_given_mcs, it creates at most 22 MSDU, A-MSDU composed by 2 MSDUs and consuming 22*14=308 bytes in headers, encapsulated in 11 A-MDPUs with total header 11*34=374 bytes
%num_packets = numel(packets);
num_packets = 1;
n_tx = zeros(1,num_packets);

for pkt = 1:num_packets
    disp(pkt)
    payload = packets{pkt};
    bytes_payload = numel(payload);
    num_tx = ceil(bytes_payload/max_ampdu);
    n_tx(pkt)  = num_tx;
    if num_tx == 1 %if we can send everythong in one transmission
        msdu_lenghts = wlanMSDULengths(bytes_payload, macConfig, phyConfig);
        if numel(msdu_lenghts) == 1 %payload fits 1 msdu
            %msdu = {}
            %msdu{1}=payload

            payload = {'00576000103afffe80'};
            %[frame,frameLength] = wlanMACFrame(payload,macConfig,phyConfig);
            %msdu{1} = "faa12334';
            [macFrame,frameLength] = wlanMACFrame(payload,cfgMAC,cfgPHY);
            %macFrame = wlanMACFrame(msdu, cfgMAC, cfgPHY);
            % Generate bits for a QoS Data frame
            %qosDataFrameBits = wlanMACFrame(msdu, cfgMAC, cfgPHY, 'OutputFormat', 'bits');
        end
            
        %num_msdu = numel(msdu_lenghts);
        %ampdu = cell(num_msdu,1);
        %index_1 = 1;
        %for msdu_index = 1:num_msdu
        %    index_2 = index_1+msdu_lenghts{msdu_index};
        %    bytes_msdu = bytes_payload(index_1, index_2);
        %    disp(size(bytes_msdu))
        %    ampdu{msdu_index}= bytes_msdu;
        %    index_1 = index_2;
        %end
    end
   
    %msduLengthPacket = wlanMSDULengths(bytes_payload, cfgMAC, cfgPHY);
    
end


helperWLANExportToPCAP({macFrame}, 'macFrames.pcap');


%max(n_tx)
%for i = 1:numel(msduLengths)
%    msdu{i} = randi([0 255], 1, msduLengths(i));
%end
%macFrame = wlanMACFrame(msdu, cfgMAC, cfgPHY);
%macFrameBits = reshape(de2bi(hex2dec(macFrame), 8)', [], 1);


phyConfig = wlanVHTConfig('MCS',7);
%phyConfig.AggregatedMPDU=1;
%phyConfig.PSDULength=60000
disp(phyConfig)
qosDataCfg = wlanMACFrameConfig('FrameType', 'QoS Data', 'FrameFormat','VHT');
qosDataCfg.MSDUAggregation=1;
%qosDataCfg.MPDUAggregation=0;

% From DS flag
qosDataCfg.FromDS = 1;
% To DS flag
qosDataCfg.ToDS = 0;
% Acknowledgment Policy
qosDataCfg.AckPolicy = 'Normal Ack';
% Receiver address
qosDataCfg.Address1 = 'FCF8B0102001';
% Transmitter address
qosDataCfg.Address2 = 'FCF8B0102002';
disp(qosDataCfg);


payload = repmat('11', 1, 20);
payload_2 = repmat('22', 1, 20);
%payload = packets{1}
%payload = 'ffffffffaaaaaaaa'

llc = [170, 170, 3, 0, 0, 0, 8, 0];
upper_layer_data = [23,45,67,89,87,55,44,33,22];
packet = [llc, packets{1}];
% Generate octets for a QoS Data frame
%phyConfig.PSDULength=0
qosDataFrame = wlanMACFrame({packet}, qosDataCfg, phyConfig);

% Generate bits for a QoS Data frame
qosDataFrameBits = wlanMACFrame(payload, qosDataCfg, phyConfig, 'OutputFormat', 'bits');
helperWLANExportToPCAP({qosDataFrame}, 'macFrames.pcap');