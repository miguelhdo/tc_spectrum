import numpy as np
from scapy.all import *

app_type_label = {
    "audio": 0,
    "video": 1,
    "no-type": 2,
    "unknown": 3
}

app_label = {
    "spotify": 0,
    "tunein": 1,
    "gpodcast": 2,
    "youtube": 3,
    "netflix": 4,
    "twitch": 5,
    "no-app": 6,
    "unknown": 7
}

labels_string = {
    "frames": ["Mgmt","Ctrl","Data"],
    "phy": ["b", "g", "n"],
    "app-type": ["audio", "video", "no-type", "unknown"],
    "app": ["spotify", "tunein", "gpodcast", "youtube", "netflix", "twitch", "no-app", "unknown"]
}


label_index = {
    "frames": 0,
    "phy": 1,
    "app-type": 2,
    "app": 3
}

num_classes = {
    "phy": 3,
    "frames": 3,
    "app-type": 4,
    "app": 8
}



def getMCSfromRate(rate):
    rate = rate / 2
    if rate == 6:
        return 0
    elif rate == 9:
        return 1
    elif rate == 12:
        return 2
    elif rate == 18:
        return 3
    elif rate == 24:
        return 4
    elif rate == 36:
        return 5
    elif rate == 48:
        return 6
    elif rate == 54:
        return 7
    else:
        # print("Rate unknown, setting MCS to -1")
        return -1


def getMCS(radiotap):
    if radiotap.MCS_index != None:
        return radiotap.MCS_index
    elif radiotap.Rate != None:
        return (getMCSfromRate(radiotap.Rate))


def getPhy(radiotap):
    # ['res1', 'res2', 'res3', 'res4', 'Turbo', 'CCK', 'OFDM', '2GHz', '5GHz', 'Passive', 'Dynamic_CCK_OFDM', 'GFSK', 'GSM', 'StaticTurbo', '10MHz', '5MHz']
    ch_flag = str(radiotap.ChannelFlags).split('+')
    # Get phy type
    if '2GHz' in ch_flag:
        # 802.11b
        if 'CCK' in ch_flag:
            phy = 0
        # 802.11g
        elif ('OFDM' in ch_flag or 'Dynamic_CCK_OFDM' in ch_flag) and radiotap.MCS_index == None:
            phy = 1
        # 802.11n
        elif radiotap.MCS_index != None:
            phy = 2
        else:
            phy = -1
    elif '5Ghz' in ch_flag:
        # 802.11n
        if radiotap.MCS_index != None:
            phy = 2
        # 802.11a
        elif 'OFDM' in ch_flag and radiotap.Rate != None:
            phy = 3
        # 802.11ac
        elif radiotap.KnownVHT != None:
            phy = 4
        else:
            phy = -1
    else:
        phy = -1

    return phy


def getPreamble(radiotap):
    preamble = 'Long'
    flags = str(radiotap.Flags).split('+')
    if 'ShortPreamble' in flags:
        preamble = 'Short'

    return preamble


def getBW_B(radiotap):
    # Get BW in 802.11b/g
    ch_flag = str(radiotap.ChannelFlags).split('+')
    if '10MHz' in ch_flag:
        bw = 'CBW10'
    elif '5MHz' in ch_flag:
        bw = 'CBW5'
    else:
        bw = 'CBW20'
    return bw


def getDataRateB(radiotap):
    # Get BW in 802.11b/g
    dr = ''
    if radiotap.Rate == 2:
        dr = '1Mbps'
    elif radiotap.Rate == 4:
        dr = '2Mbps'
    elif radiotap.Rate == 11:
        dr = '5.5Mbps'
    elif radiotap.Rate == 22:
        dr = '11Mbps'
    return dr


def getLabels(pkt, wlan_phy):
    labels = dict(phy=[], frame_type=[], app=[], app_type=[], Preamble=[], DataRate=[], ChannelFrequency=[],
                  modulation=[], bw=[], mcs=[], psduLength=[], guard_interval=[], FEC_type=[], AMPDU=[],
                  STBC_streams=[])

    # Only get information from 802.11b/g/n
    if wlan_phy in [0, 1, 2]:
        radiotap = RadioTap(pkt)
        wlan_packet = Dot11(radiotap.payload)
        tap_len = radiotap.len

        # getting noise and signal at antena in dBms
        # labels['dBm_AntNoise'] = radiotap.dBm_AntNoise
        # labels['dBm_AntSignal'] = radiotap.dBm_AntSignal

        # setting phy type
        labels['phy'] = np.uint8(wlan_phy)

        # get wlan frame as hexa duples
        frame = pkt[tap_len:].hex()
        frame = re.findall('..', str(frame))

        # Set specific parameters for 802.11b and general ones for 802.11g and 802.11n
        if wlan_phy == 0:
            # Set data rate
            labels['DataRate'] = getDataRateB(radiotap)  # Rate in radiotap has 500Kb as unit.
            # Set modulation type to DSSS/CCK as it is 802.11g
            labels['modulation'] = 'DSSS'
            # Get preamble
            labels["Preamble"] = getPreamble(radiotap)
            # Get bandwidth
            labels['bw'] = getBW_B(radiotap)

        else:
            # Set mcs
            labels['mcs'] = float(getMCS(radiotap))
            # Set modulation type to OFDM as it is 802.11g and 802.11n
            labels['modulation'] = 'OFDM'
            # Set default channel BW for 802.11g and 802.11n
            labels['bw'] = 'CBW20'
            # Set GI
            labels['guard_interval'] = 'Long'

        # get frame type
        labels['frame_type'] = np.uint8(wlan_packet.type)

        # compute psduLengh
        labels['psduLength'] = float(len(pkt) - radiotap.len)

        # Channel frequency
        labels['ChannelFrequency'] = float(radiotap.ChannelFrequency)

        if wlan_phy == 2:

            # set AMPDU
            labels['AMPDU'] = 0.0
            if radiotap.A_MPDU_flags != None:
                labels['AMPDU'] = 1.0

            # Set STBC streams
            labels['STBC_streams'] = 0.0
            if radiotap.STBC_streams != None:
                labels['STBC_streams'] = float(radiotap.STBC_streams)

            # Set Bandwidth if it is different than 20Mhz
            if radiotap.MCS_bandwidth == 1.0:
                labels['bw'] = 'CBW40'

            # Set FEC
            labels['FEC_type'] = 'BCC'
            if radiotap.FEC_type == 1:
                labels['FEC_type'] = 'LDPC'

            # Change guard interval if it is not Long
            if radiotap.guard_interval == 1:
                labels['guard_interval'] = 'Short'

    return frame, labels


def create_dataset_from_pcap(DIR, FILENAME, app, app_type, filter_mac=False, mac_address='aa:bb:cc:dd:ee:ff'):
    # DIR = '../../dataset/pcaps/pcaps_mobile_apps/24042020/'
    # DIR = '../pcaps/pcaps_mobile_apps/'
    # FILENAME = '2G_n_multi_mobile_app_multi_channel_21042020_balanced'
    FILE_TO_READ = DIR + FILENAME + '.pcap'
    readBytes = 0
    fileSize = os.stat(FILE_TO_READ).st_size

    counter_packets = 0
    wlan_frames = []
    labels_frames = []
    counter_mgmt = 0
    counter_ctr = 0
    counter_qos = 0
    counter_error = 0
    counter_data_filtered_mac = 0
    counter_data_unknown = 0
    counter_audio = 0
    counter_video = 0
    counter_no_type = 0
    counter_type_unknown = 0
    counter_spotify = 0
    counter_tunein = 0
    counter_gpodcast = 0
    counter_youtube = 0
    counter_netflix = 0
    counter_twitch = 0
    counter_no_app = 0
    counter_app_unknown = 0
    counter_total_labeled_frames = 0



    for pkt, (sec, usec, wirelen, c) in RawPcapReader(FILE_TO_READ):
        # readBytes += len(Ether(pkt))
        # print("%.2f" % (float(readBytes) / fileSize * 100))
        counter_packets += 1
        # if counter_packets!=ref_packet:
        #    continue
        try:
            radiotap = RadioTap(pkt)
            wlan_phy = getPhy(radiotap)
            wlan_packet = Dot11(radiotap.payload)
        except:
            counter_error += 1
            print("Error decoding packet, ignoring it")
            continue

        # Only get information from 802.11b/g/n
        if wlan_phy in [0, 1, 2]:
            wlan_frame, labels = getLabels(pkt, wlan_phy)

            if labels['frame_type'] == 2:
                labels['app'] = app_label["unknown"]
                labels['app_type'] = app_type_label["unknown"]
                
                if filter_mac:
                    wlan_packet = Dot11(radiotap.payload)
                    if mac_address in [wlan_packet.addr1, wlan_packet.addr2, wlan_packet.addr3, wlan_packet.addr4]:
                        labels['app'] = app_label[app]
                        labels['app_type'] = app_type_label[app_type]
                        counter_data_filtered_mac += 1
            else:
                labels['app'] = app_label["no-app"]
                labels['app_type'] = app_type_label["no-type"]

            if labels['app'] == app_label["unknown"]:
                counter_data_unknown += 1
            
            #print('type: ' ,labels['app_type'])

            if labels['app_type'] ==0:
                counter_audio +=1
            elif labels['app_type'] ==1:
                counter_video +=1
            elif labels['app_type'] ==2:
                counter_no_type +=1
            elif labels['app_type'] ==3:
                counter_type_unknown +=1
            else:
                print('Something wrong with app-type label')
            
            #print('label: ',labels['app'])
            if labels['app'] ==0:
                counter_spotify +=1
            elif labels['app'] ==1:
                counter_tunein +=1
            elif labels['app'] ==2:
                counter_gpodcast +=1
            elif labels['app'] ==3:
                counter_youtube +=1
            elif labels['app'] ==4:
                counter_netflix +=1
            elif labels['app'] ==5:
                counter_twitch +=1
            elif labels['app'] ==6:
                counter_no_app +=1
            elif labels['app'] ==7:
                counter_app_unknown +=1
            else:
                print('Something wrong with app label')
            # print(labels['app'],labels['app_type'])

            frame_type = labels['frame_type']
            if frame_type == 0:
                counter_mgmt += 1
            elif frame_type == 1:
                counter_ctr += 1
            elif frame_type == 2:
                counter_qos += 1
            else:
                print("wrong wlan type")
                continue

            if filter_mac and (labels['app'] ==7 or labels['app_type'] ==3):
                continue
        
            wlan_frames.append(wlan_frame)
            labels_frames.append(labels)
            counter_total_labeled_frames+=1
        
        elif wlan_phy < 0:
            print(counter_packets, wlan_phy)
            print("wrong phy")
            continue
        else:
            continue

    print('----------------------------------Summary--------------------------------------------------')
    print('Total Packets: ', counter_packets)
    print('Total Packets (Mgmgt+Ctr+Data):', counter_mgmt + counter_ctr + counter_qos)
    print('Packets with error: ', counter_error)
    print('Packets Mgmt: ', counter_mgmt)
    print('Packets Ctrl: ', counter_ctr)
    print('Packets Data: ', counter_qos)
    print('Packets Type Unknown: ', counter_data_unknown)
    print('Packets audio: ', counter_audio)
    print('Packets video: ',counter_video)
    print('Packets no type: ',counter_no_type)
    print('Packets unknown type: ',counter_type_unknown)
    print('Packets spotify: ', counter_spotify)
    print('Packets tune-in: ',counter_tunein)
    print('Packets gpodcast: ',counter_gpodcast)
    print('Packets youtube: ',counter_youtube)
    print('Packets netflix: ',counter_netflix)
    print('Packets twitch: ',counter_twitch)
    print('Packets no app: ',counter_no_app)
    print('Packets app unknown: ',counter_app_unknown)
    print('Label data frames from unknown as unknown app-type and unknown app?: ', str(not filter_mac))
    if filter_mac:
        print('Total data frames with known label and filtered by mac: ', counter_data_filtered_mac)
    print('Total data frames with labels (including unknown app-type and unknown app): ', counter_total_labeled_frames)
    print('------------------------------------------------------------------------------------')
    return wlan_frames, labels_frames
