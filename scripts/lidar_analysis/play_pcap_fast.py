import socket
import time
import dpkt

pcap_file = "pcaps/lidar_raw_5.pcap"
dst_ip = "127.0.0.1"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

start_real = time.perf_counter()
start_ts = 0

# Step 1: Preload all UDP payloads, ports, and timestamps
print("Playing back packets!")

with open(pcap_file, "rb") as f:
    pcap = dpkt.pcap.Reader(f)
    for ts, buf in pcap:
        if start_ts==0:
            start_ts = ts
        # Parse Ethernet frame
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, dpkt.ip.IP):
            ip = eth.data
            if isinstance(ip.data, dpkt.udp.UDP):
                udp = ip.data
                sock.sendto(bytes(udp.data), (dst_ip, udp.dport))
                target = start_real + (ts - start_ts)
                while True:
                    now = time.perf_counter()
                    if now >= target:
                        break

print("End of pcap file reached!")
