from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import OVSController
from mininet.log import setLogLevel, info
from mininet.cli import CLI
import time
import os

# Save the TCP info reader to be run in h1
TCP_INFO_READER = """#!/usr/bin/env python3
import socket
import ctypes

class TCPInfo(ctypes.Structure):
    _fields_ = [
        ("tcpi_state", ctypes.c_uint8),
        ("tcpi_ca_state", ctypes.c_uint8),
        ("tcpi_retransmits", ctypes.c_uint8),
        ("tcpi_probes", ctypes.c_uint8),
        ("tcpi_backoff", ctypes.c_uint8),
        ("tcpi_options", ctypes.c_uint8),
        ("tcpi_snd_wscale", ctypes.c_uint8, 4),
        ("tcpi_rcv_wscale", ctypes.c_uint8, 4),
        ("tcpi_rto", ctypes.c_uint32),
        ("tcpi_ato", ctypes.c_uint32),
        ("tcpi_snd_mss", ctypes.c_uint32),
        ("tcpi_rcv_mss", ctypes.c_uint32),
        ("tcpi_unacked", ctypes.c_uint32),
        ("tcpi_sacked", ctypes.c_uint32),
        ("tcpi_lost", ctypes.c_uint32),
        ("tcpi_retrans", ctypes.c_uint32),
        ("tcpi_fackets", ctypes.c_uint32),
        ("tcpi_last_data_sent", ctypes.c_uint32),
        ("tcpi_last_ack_sent", ctypes.c_uint32),
        ("tcpi_last_data_recv", ctypes.c_uint32),
        ("tcpi_last_ack_recv", ctypes.c_uint32),
        ("tcpi_pmtu", ctypes.c_uint32),
        ("tcpi_rcv_ssthresh", ctypes.c_uint32),
        ("tcpi_rtt", ctypes.c_uint32),
        ("tcpi_rttvar", ctypes.c_uint32),
        ("tcpi_snd_ssthresh", ctypes.c_uint32),
        ("tcpi_snd_cwnd", ctypes.c_uint32),
        ("tcpi_advmss", ctypes.c_uint32),
        ("tcpi_reordering", ctypes.c_uint32),
        ("tcpi_rcv_rtt", ctypes.c_uint32),
        ("tcpi_rcv_space", ctypes.c_uint32),
        ("tcpi_total_retrans", ctypes.c_uint32),
    ]

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("10.0.0.2", 12345))

buf = ctypes.create_string_buffer(104)
sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, buf)
info = TCPInfo.from_buffer_copy(buf)

print(f"wscale:{info.tcpi_snd_wscale};"
      f"rto:{info.tcpi_rto};"
      f"rtt:{info.tcpi_rtt};"
      f"mss:{info.tcpi_snd_mss};"
      f"pmtu:{info.tcpi_pmtu};"
      f"rcvmss:{info.tcpi_rcv_mss};"
      f"advmss:{info.tcpi_advmss};"
      f"cwnd:{info.tcpi_snd_cwnd};"
      f"ssthresh:{info.tcpi_snd_ssthresh};"
      f"bytes_sent:NA;"
      f"bytes_retrans:{info.tcpi_total_retrans};"
      f"bytes_acked:{info.tcpi_sacked};"
      f"segs_out:NA;"
      f"segs_in:NA;"
      f"data_segs_out:NA;"
      f"lastrcv:{info.tcpi_last_data_recv};"
      f"delivered:NA;"
      f"rcv_space:{info.tcpi_rcv_space};"
      f"rcv_ssthresh:{info.tcpi_rcv_ssthresh};")
sock.close()
"""


class SingleLinkTopo(Topo):
    def build(self):
        h1 = self.addHost("h1")
        h2 = self.addHost("h2")
        self.addLink(h1, h2)


def run():
    topo = SingleLinkTopo()
    net = Mininet(topo=topo, controller=OVSController)
    net.start()

    h1 = net.get("h1")
    h2 = net.get("h2")

    # Start a simple TCP server on h2
    h2.cmd("nohup nc -l -p 12345 >/dev/null 2>&1 &")
    time.sleep(1)

    # Write TCP info script to h1
    with open("tcp_info_reader.py", "w") as f:
        f.write(TCP_INFO_READER)
    h1.cmd("cp tcp_info_reader.py /tmp/tcp_info_reader.py")
    h1.cmd("chmod +x /tmp/tcp_info_reader.py")

    print("\n=== TCP Info from h1 ===")
    result = h1.cmd("python3 /tmp/tcp_info_reader.py")
    print(result.strip())

    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    run()
