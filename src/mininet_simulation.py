#!/usr/bin/env python

"""
TCP Congestion Control Data Collection Script using Mininet
This script creates network simulations with different conditions and collects 
performance data for TCP congestion control algorithms (Reno, Cubic).
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.clean import cleanup

import os
import time
import subprocess
from datetime import datetime
import argparse

class DumbbellTopo(Topo):
    """Dumbbell topology with n left hosts and n right hosts.
    
    Left hosts are connected to left switch, right hosts to right switch.
    Switches are connected with a single bottleneck link.
    """
    
    def build(self, n=2, bw_bottleneck=10, delay_bottleneck='10ms', loss=0, 
              bw_access=100, delay_access='1ms', queue_size=100):
        # Create switches
        left_switch = self.addSwitch('s1')
        right_switch = self.addSwitch('s2')
        
        # Create bottleneck link
        self.addLink(
            left_switch, 
            right_switch, 
            bw=bw_bottleneck, 
            delay=delay_bottleneck, 
            loss=loss,
            max_queue_size=queue_size
        )
        
        # Add hosts and links on left side
        for i in range(n):
            host = self.addHost(f'h{i+1}')
            self.addLink(
                host, 
                left_switch, 
                bw=bw_access, 
                delay=delay_access
            )
        
        # Add hosts and links on right side
        for i in range(n):
            host = self.addHost(f'h{i+n+1}')
            self.addLink(
                host, 
                right_switch, 
                bw=bw_access, 
                delay=delay_access
            )


def collect_tcp_data(host, congestion_algo, log_file, duration=60):
    """Collect TCP data using ss and log to a file."""
    # Set TCP congestion algorithm
    host.cmd(f'sysctl -w net.ipv4.tcp_congestion_control={congestion_algo}')
    
    # Start data collection in background
    cmd = f'while true; do ss -i --tcp -o state established -p dst {host.IP()}:'
    cmd += f'5001 >> {log_file}; sleep 0.1; done & echo $! > /tmp/ss_pid'
    host.cmd(cmd)
    
    # Return the process ID for later termination
    pid = host.cmd('cat /tmp/ss_pid').strip()
    return pid


def stop_collection(host, pid):
    """Stop the data collection process."""
    host.cmd(f'kill {pid}')


def process_tcp_data(log_file, output_csv):
    """Process the raw TCP data into a CSV format similar to existing datasets."""
    # Use our processing script to convert the raw ss output into CSV format
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'process_tcp_data.py')
    
    # Extract congestion algorithm from output filename
    algo = 'reno' if 'reno' in output_csv.lower() else 'cubic'
    
    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(output_csv)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Run the processing script
    cmd = f'python {script_path} {log_file} {output_csv} {algo}'
    print(f"Processing data with command: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Successfully processed data to {output_csv}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing data: {e}")


def run_experiment(net, congestion_algo, bw_bottleneck, delay_bottleneck, 
                  loss_rate, duration=60):
    """Run a single experiment with specified parameters."""
    
    print(f"\nRunning experiment with {congestion_algo}, " 
          f"BW: {bw_bottleneck}Mbps, Delay: {delay_bottleneck}, Loss: {loss_rate}%\n")
    
    # Get hosts
    h1, h3 = net.get('h1', 'h3')
    
    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Log file for raw data
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    raw_log = f"{data_dir}/raw_{congestion_algo}_{timestamp}.log"
    output_csv = f"{data_dir}/{congestion_algo}_{timestamp}.csv"
    
    # Start iperf server on h3
    h3.cmd('iperf -s -p 5001 > /dev/null &')
    time.sleep(1)  # Wait for server to start
    
    # Start data collection
    pid = collect_tcp_data(h1, congestion_algo, raw_log, duration)
    
    # Start iperf client on h1
    h1.cmd(f'iperf -c {h3.IP()} -p 5001 -t {duration} > /dev/null &')
    
    # Wait for the experiment to complete
    print(f"Experiment running for {duration} seconds...")
    time.sleep(duration + 5)  # Add a buffer of 5 seconds
    
    # Stop data collection
    stop_collection(h1, pid)
    
    # Stop iperf server
    h3.cmd('pkill -f "iperf -s"')
    
    # Process the raw data into CSV format
    process_tcp_data(raw_log, output_csv)
    
    print(f"Experiment completed. Data saved to {output_csv}")
    return output_csv


def run_all_experiments(args):
    """Run multiple experiments with different parameters and collect data."""
    
    congestion_algos = ['reno', 'cubic']
    bandwidths = [5, 10, 20]  # Mbps
    delays = ['10ms', '50ms', '100ms']
    loss_rates = [0, 1, 2, 5]  # %
    
    # Create output directory if it doesn't exist
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run experiments
    for algo in congestion_algos:
        for bw in bandwidths:
            for delay in delays:
                for loss in loss_rates:
                    # Create topology
                    topo = DumbbellTopo(n=2, bw_bottleneck=bw, 
                                        delay_bottleneck=delay, loss=loss)
                    
                    # Create network
                    net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink)
                    
                    # Start network
                    net.start()
                    
                    try:
                        # Run experiment
                        run_experiment(net, algo, bw, delay, loss, args.duration)
                    except Exception as e:
                        print(f"Error in experiment: {e}")
                    finally:
                        # Stop network
                        net.stop()
                        cleanup()
                        time.sleep(2)  # Wait before next experiment
    
    print("All experiments completed!")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TCP Congestion Control Data Collection')
    parser.add_argument('--duration', type=int, default=60, 
                        help='Duration of each experiment in seconds')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode with CLI')
    args = parser.parse_args()
    
    # Set log level
    setLogLevel('info')
    
    if args.interactive:
        # Create a simple dumbbell topology
        topo = DumbbellTopo(n=2, bw_bottleneck=10, delay_bottleneck='10ms', loss=0)
        
        # Create and start network
        net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink)
        net.start()
        
        # Print node connections
        print("\n*** Dumping network connections:")
        dumpNodeConnections(net.hosts)
        
        # Test network connectivity
        print("\n*** Testing network connectivity:")
        net.pingAll()
        
        # Start CLI
        print("\n*** Starting CLI:")
        CLI(net)
        
        # Stop network
        net.stop()
    else:
        # Run all experiments
        run_all_experiments(args) 