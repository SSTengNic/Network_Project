#!/usr/bin/env python3

"""
Real-time TCP Congestion Control Algorithm Prediction
This script collects network data from Mininet in real-time for 15 seconds,
then uses the predict_loss module to recommend the best TCP algorithm.
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.clean import cleanup

import numpy as np
import time
import os
import sys
import re
import signal
from src.predict_loss import predict_best_algorithm, load_normalization_params

# Define a simple topology
class SimpleTopo(Topo):
    """Simple dumbbell topology with 2 hosts."""
    
    def build(self):
        # Create switches
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        
        # Add hosts
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        
        # Add host links with high bandwidth and low delay
        self.addLink(h1, s1, bw=100, delay='1ms')
        self.addLink(h2, s2, bw=100, delay='1ms')
        
        # Create bottleneck link
        self.addLink(
            s1, 
            s2, 
            bw=50, 
            delay='10ms', 
            loss=0,
            max_queue_size=100
        )

def collect_live_data(net, duration=15, algorithm="reno"):
    """
    Collect network data in real-time from Mininet for a specified duration.
    
    Args:
        net: Mininet network instance
        duration: Duration to collect data (in seconds)
        algorithm: Current TCP algorithm ('reno' or 'cubic')
        
    Returns:
        numpy.ndarray: Array of collected data points (15x15)
    """
    print(f"\n[+] Starting data collection for {duration} seconds using {algorithm} algorithm...")
    
    # Set the TCP congestion control algorithm
    h1, h2 = net.get('h1', 'h2')
    h1.cmd(f"sysctl -w net.ipv4.tcp_congestion_control={algorithm}")
    
    # Start iperf server on h2
    h2.cmd('iperf -s > /dev/null &')
    server_pid = h2.cmd("echo $!")
    time.sleep(1)  # Wait for server to start
    
    # Start iperf client on h1
    h1.cmd(f'iperf -c {h2.IP()} -t {duration+5} > /dev/null &')
    client_pid = h1.cmd("echo $!")
    
    # Initialize data collection
    data_points = []
    
    # Create a function to parse ss output and extract relevant metrics
    def parse_ss_output(output):
        metrics = {}
        # Parse relevant fields using regex
        wscale_match = re.search(r'wscale:(\d+)', output)
        rto_match = re.search(r'rto:(\d+)', output)
        rtt_match = re.search(r'rtt:([0-9.]+)', output)
        mss_match = re.search(r'mss:(\d+)', output)
        cwnd_match = re.search(r'cwnd:(\d+)', output)
        ssthresh_match = re.search(r'ssthresh:(\d+)', output)
        bytes_sent_match = re.search(r'bytes_sent:(\d+)', output)
        bytes_retrans_match = re.search(r'bytes_retrans:(\d+)', output)
        bytes_acked_match = re.search(r'bytes_acked:(\d+)', output)
        segs_out_match = re.search(r'segs_out:(\d+)', output)
        segs_in_match = re.search(r'segs_in:(\d+)', output)
        data_segs_out_match = re.search(r'data_segs_out:(\d+)', output)
        
        # Extract values with defaults if not found
        metrics['wscale'] = float(wscale_match.group(1)) if wscale_match else 0
        metrics['rto'] = float(rto_match.group(1)) if rto_match else 0
        metrics['rtt'] = float(rtt_match.group(1)) if rtt_match else 0
        metrics['mss'] = float(mss_match.group(1)) if mss_match else 0
        metrics['cwnd'] = float(cwnd_match.group(1)) if cwnd_match else 0
        metrics['ssthresh'] = float(ssthresh_match.group(1)) if ssthresh_match else 0
        metrics['bytes_sent'] = float(bytes_sent_match.group(1)) if bytes_sent_match else 0
        metrics['bytes_retrans'] = float(bytes_retrans_match.group(1)) if bytes_retrans_match else 0
        metrics['bytes_acked'] = float(bytes_acked_match.group(1)) if bytes_acked_match else 0
        metrics['segs_out'] = float(segs_out_match.group(1)) if segs_out_match else 0
        metrics['segs_in'] = float(segs_in_match.group(1)) if segs_in_match else 0
        metrics['data_segs_out'] = float(data_segs_out_match.group(1)) if data_segs_out_match else 0
        
        # Calculate approximate loss_ratio
        metrics['loss_ratio'] = metrics['bytes_retrans'] / max(1, metrics['bytes_sent'])
        
        # Add timestamp and TCP type
        metrics['timestamp'] = time.time()
        metrics['tcp_type'] = 1 if algorithm.lower() == 'reno' else 0
        
        return metrics
    
    # Collect data for specified duration
    start_time = time.time()
    while time.time() - start_time < duration:
        # Get current TCP connection data using ss command
        ss_output = h1.cmd("ss -tinem -o state established")
        
        # Only process if we have data
        if "ESTAB" in ss_output:
            metrics = parse_ss_output(ss_output)
            
            # Organize metrics into a list in the same order as the training data
            data_point = [
                metrics['wscale'],
                metrics['rto'],
                metrics['rtt'],
                metrics['mss'],
                metrics['cwnd'],
                metrics['ssthresh'],
                metrics['bytes_sent'],
                metrics['bytes_retrans'],
                metrics['bytes_acked'],
                metrics['segs_out'],
                metrics['segs_in'],
                metrics['data_segs_out'],
                metrics['loss_ratio'],
                metrics['tcp_type'],
                metrics['timestamp']
            ]
            
            data_points.append(data_point)
            print(f"Collected data point {len(data_points)}: loss_ratio={metrics['loss_ratio']:.8f}, cwnd={metrics['cwnd']}")
        
        # Sleep for 1 second between data points
        time.sleep(1)
    
    # Clean up
    h2.cmd(f"kill -9 {server_pid}")
    h1.cmd(f"kill -9 {client_pid}")
    
    # Ensure we have exactly 15 data points
    if len(data_points) > 15:
        data_points = data_points[-15:]  # Take the most recent 15 points
    elif len(data_points) < 15:
        # If we have fewer than 15 points, duplicate the last one to fill
        last_point = data_points[-1] if data_points else [0] * 15
        while len(data_points) < 15:
            data_points.append(last_point)
    
    return np.array(data_points)

def clean_up_mininet():
    """Clean up any Mininet residual processes"""
    cleanup()

def main():
    """Run the experiment and predict the best TCP algorithm"""
    # Clean up any previous Mininet run
    clean_up_mininet()
    
    # Set Mininet log level
    setLogLevel('info')
    
    # Get current algorithm
    try:
        current_algorithm = os.popen("sysctl -n net.ipv4.tcp_congestion_control").read().strip()
        print(f"Current TCP congestion control algorithm: {current_algorithm}")
    except:
        current_algorithm = "cubic"  # Default to cubic if we can't determine
        print(f"Could not determine current algorithm, defaulting to: {current_algorithm}")
    
    # Create and set up the network
    topo = SimpleTopo()
    net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink)
    net.start()
    
    # Dump node connections for debugging
    print("Dumping host connections")
    dumpNodeConnections(net.hosts)
    
    try:
        # Ensure model normalization parameters are loaded
        load_normalization_params()
        
        # Collect data for 15 seconds
        data = collect_live_data(net, duration=15, algorithm=current_algorithm)
        
        print("\nPredicting best TCP algorithm...")
        # Call prediction function
        recommended_algorithm = predict_best_algorithm(current_algorithm, data)
        
        print(f"\n[RESULT] Current algorithm: {current_algorithm}")
        print(f"[RESULT] Recommended algorithm: {recommended_algorithm}")
        
        # If a switch is recommended, provide command to change it
        if recommended_algorithm != current_algorithm:
            print("\nTo switch to the recommended algorithm, run:")
            print(f"sudo sysctl -w net.ipv4.tcp_congestion_control={recommended_algorithm}")
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        # Cleanup
        print("\nCleaning up Mininet...")
        net.stop()
        clean_up_mininet()

if __name__ == "__main__":
    # Register signal handler for SIGINT (ctrl+c)
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    main() 