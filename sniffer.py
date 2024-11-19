import pyshark
capture = pyshark.LiveCapture(interface='WLAN',output_file='sniff_1.pcap')
capture.sniff(timeout=10)
for packet in capture.sniff_continuously(packet_count=100):
    print(packet)
capture.close()
# for packet in capture.sniff_continuously():
#     print(packet.highest_layer)
#     if hasattr(packet,'ip'):
#         print(packet.ip)

