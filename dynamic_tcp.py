from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel
import os
import time
import threading

# Define a simple topology
class SingleLinkTopo(Topo):
    def build(self):
        sender = self.addHost('h1')
        receiver = self.addHost('h2')
        switch = self.addSwitch('s1')
        
        # Add links with some delay and loss (simulating real conditions)
        self.addLink(sender, switch, cls=TCLink, bw=10, delay='10ms', loss=1)
        self.addLink(receiver, switch, cls=TCLink, bw=10, delay='10ms', loss=1)

# Function to change TCP congestion control algorithm dynamically
def switch_tcp_algo(host, algorithms, interval=15):
    """ Dynamically switches the TCP congestion control algorithm. """
    index = 0
    while True:
        algo = algorithms[index]
        print(f"\n[+] Switching TCP Congestion Control to: {algo}")
        host.cmd(f"sysctl -w net.ipv4.tcp_congestion_control={algo}")
        index = (index + 1) % len(algorithms)  # Cycle through the algorithms
        time.sleep(interval)

# Run the Mininet topology
def run():
    setLogLevel('info')
    net = Mininet(topo=SingleLinkTopo(), link=TCLink)
    net.start()

    h1, h2 = net.get('h1'), net.get('h2')

    # Enable IP forwarding
    os.system('sysctl -w net.ipv4.ip_forward=1')

    # Start iperf server on h2 (receiver)
    print("\n[+] Starting iperf server on h2...")
    h2.cmd("iperf -s -i 1 > iperf_server.log &")

    # List of congestion control algorithms to cycle through
    tcp_algorithms = ["reno", "cubic", "bbr"]

    # Start a thread to switch TCP congestion control every 15 seconds
    switch_thread = threading.Thread(target=switch_tcp_algo, args=(h1, tcp_algorithms))
    switch_thread.daemon = True
    switch_thread.start()

    # Start iperf client on h1 (sender)
    print("\n[+] Starting iperf client on h1...")
    h1.cmd("iperf -c " + h2.IP() + " -t 60 > iperf_client.log &")

    # Open CLI for debugging
    CLI(net)

    # Stop the network
    net.stop()

if __name__ == '__main__':
    run()