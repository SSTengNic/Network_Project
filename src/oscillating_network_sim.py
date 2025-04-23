#!/usr/bin/env python3

"""
Mininet script to continuously monitor TCP connection metrics
while oscillating between network conditions that favor different 
TCP congestion control algorithms.
"""

import time
import re
import argparse
import sys
import signal
from collections import deque
import os

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.clean import cleanup

# --- Model Loading ---
import torch
import numpy as np
import pandas as pd

# Network conditions that favor TCP Cubic (high BDP network)
CUBIC_FAVORABLE = {
    "BW": 100,     # High bandwidth (Mbps)
    "DELAY": "80ms", # High delay
    "LOSS": 1,     # Low to moderate loss
    "QSIZE": 800   # Large buffer to handle high BDP
}

# Network conditions that favor TCP Reno (moderate to low BDP network)
RENO_FAVORABLE = {
    "BW": 20,      # Lower bandwidth (Mbps)
    "DELAY": "5ms", # Lower delay
    "LOSS": 2,     # Slightly higher loss (Reno handles moderate loss better)
    "QSIZE": 30    # Smaller buffer
}

# Import constants and functions from sim.py
from sim import MODEL_FEATURE_COLUMNS, LSTM_pt, ModelConfig, load_normalization_params
from sim import RENO_CONFIG, CUBIC_CONFIG, RENO_CUBIC_CONFIG, load_model, normalize_data
from sim import predict_loss, predict_best_algorithm

# --- Data Storage ---
# A deque automatically maintains a fixed size, discarding oldest entries
data_history = None

# --- Topology Definition ---
class DumbbellTopo(Topo):
    """Simple dumbbell topology with 2 hosts and configurable bottleneck."""
    def build(self, bw=CUBIC_FAVORABLE.get('BW'), loss=CUBIC_FAVORABLE.get('LOSS'), 
              delay=CUBIC_FAVORABLE.get('DELAY'), queue=CUBIC_FAVORABLE.get('QSIZE')):
        # create switches
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')

        # add hosts
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')

        # add host links (access links)
        self.addLink(h1, s1, bw=1000, delay='1ms')
        self.addLink(h2, s2, bw=1000, delay='1ms')

        # add bottleneck link
        self.addLink(
            s1,
            s2,
            bw=bw,
            delay=delay,
            loss=loss,
            max_queue_size=queue
        )

# --- Data Parsing Logic ---
def parse_ss_output(ss_output, h1_ip, h2_ip, current_tcp_algo, port):
    """
    Parses the output of 'ss -tinem -o' to extract relevant TCP metrics
    for connections between h1 and h2.

    Args:
        ss_output (str): The raw string output from the ss command.
        h1_ip (str): IP address of host h1.
        h2_ip (str): IP address of host h2.
        current_tcp_algo (str): The currently active TCP algorithm.
        port (int): The port to filter for.

    Returns:
        list: A list of dictionaries, each containing metrics for a found connection.
              Returns an empty list if no relevant connection or metrics are found.
    """
    connections_data = []
    lines = ss_output.strip().split('\n')
    now = time.time() # timestamp for this data collection cycle

    # regular expressions for parsing ss output
    wscale_match_re = re.compile(r'\swscale:([0-9]+,[0-9]+)\s')
    rto_match_re = re.compile(r'\srto:([0-9]+)\s')
    rtt_match_re = re.compile(r'\srtt:([0-9]+.[0-9]+)/')
    mss_match_re = re.compile(r'\smss:([0-9]+)\s')
    pmtu_match_re = re.compile(r'\spmtu:([0-9]+)\s')
    rcvmss_match_re = re.compile(r'\srcvmss:([0-9]+)\s')
    advmss_match_re = re.compile(r'\sadvmss:([0-9]+)\s')
    cwnd_match_re = re.compile(r'\scwnd:([0-9]+)\s')
    ssthresh_match_re = re.compile(r'\sssthresh:([0-9]+)\s')
    bytes_sent_match_re = re.compile(r'\sbytes_sent:([0-9]+)\s')
    bytes_retrans_match_re = re.compile(r'\sbytes_retrans:([0-9]+)\s')
    bytes_acked_match_re = re.compile(r'\sbytes_acked:([0-9]+)\s')
    segs_out_match_re = re.compile(r'\ssegs_out:([0-9]+)\s')
    segs_in_match_re = re.compile(r'\ssegs_in:([0-9]+)\s')
    data_segs_out_match_re = re.compile(r'\sdata_segs_out:([0-9]+)\s')
    lastrcv_match_re = re.compile(r'\slastrcv:([0-9]+)\s')
    delivered_match_re = re.compile(r'\sdelivered:([0-9]+)\s')
    rcv_space_match_re = re.compile(r'\srcv_space:([0-9]+)\s')
    rcv_ssthresh_match_re = re.compile(r'\srcv_ssthresh:([0-9]+)\s')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # check if this line contains the IPs we care about
        is_relevant_connection = (h1_ip in line and f"{h2_ip}:{port}" in line)
        if is_relevant_connection:
            print(f"Relevant connection found: {line}")

        # look for the next line which should contain the metrics
        if is_relevant_connection and i < len(lines) - 1 and 'timer:(' in line:
            i = i + 1 # move to the next line for metrics
            metrics_line = lines[i].strip()

            metrics = {'timestamp': now, 'tcp_type': current_tcp_algo}

            # extract fields
            wscale_match = wscale_match_re.search(metrics_line)
            metrics['wscale'] = float(wscale_match.group(1).replace(',', '.')) if wscale_match else 0

            rto_match = rto_match_re.search(metrics_line)
            metrics['rto'] = int(rto_match.group(1)) if rto_match else 0

            rtt_match = rtt_match_re.search(metrics_line)
            metrics['rtt'] = float(rtt_match.group(1)) if rtt_match else 0.0

            mss_match = mss_match_re.search(metrics_line)
            metrics['mss'] = int(mss_match.group(1)) if mss_match else 0

            pmtu_match = pmtu_match_re.search(metrics_line)
            metrics['pmtu'] = int(pmtu_match.group(1)) if pmtu_match else 0

            rcvmss_match = rcvmss_match_re.search(metrics_line)
            metrics['rcvmss'] = int(rcvmss_match.group(1)) if rcvmss_match else 0

            advmss_match = advmss_match_re.search(metrics_line)
            metrics['advmss'] = int(advmss_match.group(1)) if advmss_match else 0

            cwnd_match = cwnd_match_re.search(metrics_line)
            metrics['cwnd'] = int(cwnd_match.group(1)) if cwnd_match else 0

            ssthresh_match = ssthresh_match_re.search(metrics_line)
            metrics['ssthresh'] = int(ssthresh_match.group(1)) if ssthresh_match else 0

            bytes_sent_match = bytes_sent_match_re.search(metrics_line)
            metrics['bytes_sent'] = int(bytes_sent_match.group(1)) if bytes_sent_match else 0

            bytes_retrans_match = bytes_retrans_match_re.search(metrics_line)
            metrics['bytes_retrans'] = int(bytes_retrans_match.group(1)) if bytes_retrans_match else 0

            bytes_acked_match = bytes_acked_match_re.search(metrics_line)
            metrics['bytes_acked'] = int(bytes_acked_match.group(1)) if bytes_acked_match else 0

            segs_out_match = segs_out_match_re.search(metrics_line)
            metrics['segs_out'] = int(segs_out_match.group(1)) if segs_out_match else 0

            segs_in_match = segs_in_match_re.search(metrics_line)
            metrics['segs_in'] = int(segs_in_match.group(1)) if segs_in_match else 0

            data_segs_out = data_segs_out_match_re.search(metrics_line)
            metrics['data_segs_out'] = int(data_segs_out.group(1)) if data_segs_out else 0

            lastrcv_match = lastrcv_match_re.search(metrics_line)
            metrics['lastrcv'] = int(lastrcv_match.group(1)) if lastrcv_match else 0

            delivered_match = delivered_match_re.search(metrics_line)
            metrics['delivered'] = int(delivered_match.group(1)) if delivered_match else 0

            recv_space_match = rcv_space_match_re.search(metrics_line)
            metrics['rcv_space'] = int(recv_space_match.group(1)) if recv_space_match else 0

            rcv_ssthresh_match = rcv_ssthresh_match_re.search(metrics_line)
            metrics['rcv_ssthresh'] = int(rcv_ssthresh_match.group(1)) if rcv_ssthresh_match else 0

            connections_data.append(metrics)

        i += 1

    return connections_data

def update_network_conditions(net, condition_type="cubic"):
    """
    Update the network conditions to favor either Cubic or Reno.
    
    Args:
        net: The mininet network instance
        condition_type: Which conditions to apply - "cubic" or "reno"
    """
    if condition_type.lower() == "cubic":
        settings = CUBIC_FAVORABLE
    else:
        settings = RENO_FAVORABLE
    
    print(f"\n[+] Updating network to {condition_type.upper()}-favorable conditions:")
    print(f"    Bandwidth: {settings['BW']} Mbps")
    print(f"    Delay: {settings['DELAY']}")
    print(f"    Loss: {settings['LOSS']}%")
    print(f"    Queue Size: {settings['QSIZE']} packets")
    
    # Get the link between s1 and s2 (the bottleneck link)
    s1, s2 = net.get('s1', 's2')
    link = None
    for l in net.links:
        if (l.intf1.node == s1 and l.intf2.node == s2) or (l.intf1.node == s2 and l.intf2.node == s1):
            link = l
            break
    
    if link:
        print("updating link parameters")
        # Update link parameters using tc
        for intf in [link.intf1, link.intf2]:
            # Clear existing TC rules
            intf.node.cmd(f'tc qdisc del dev {intf.name} root')
            # Apply new TC rules
            intf.node.cmd(f'tc qdisc add dev {intf.name} root handle 1: htb default 10')
            intf.node.cmd(f'tc class add dev {intf.name} parent 1: classid 1:10 htb rate {settings["BW"]}Mbit')
            intf.node.cmd(f'tc qdisc add dev {intf.name} parent 1:10 handle 10: netem delay {settings["DELAY"]} loss {settings["LOSS"]}% limit {settings["QSIZE"]}')
    else:
        print("Error: Could not find bottleneck link between s1 and s2")

# --- Main Execution ---
def main():
    global data_history

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Mininet TCP Data Collector with Oscillating Network Conditions')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Data collection interval in seconds (default: 1.0)')
    parser.add_argument('--history', type=int, default=15,
                        help='Number of data points to keep in memory (default: 15)')
    parser.add_argument('--oscillation', type=int, default=30,
                        help='Time in seconds between network oscillations (default: 30)')
    parser.add_argument('--fixed_algo', type=str, default=None,
                        help='Fix TCP algorithm instead of switching (options: reno, cubic, none)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='File to log performance metrics (default: none)')
    args = parser.parse_args()

    if args.interval <= 0:
        print("Error: Interval must be positive.")
        sys.exit(1)
    if args.history <= 0:
        print("Error: History size must be positive.")
        sys.exit(1)
    if args.oscillation <= 0:
        print("Error: Oscillation period must be positive.")
        sys.exit(1)

    # Initialize the deque
    data_history = deque(maxlen=args.history)
    
    # Check if logs should be saved
    log_file = None
    if args.log_file:
        try:
            log_file = open(args.log_file, 'w')
            log_file.write("timestamp,network_type,tcp_algorithm,rtt,cwnd,bytes_acked,segs_out,loss_ratio\n")
        except IOError as e:
            print(f"Error opening log file: {e}")
            log_file = None

    # Define ports to use for the connections
    port1 = 5001
    port2 = 5002
    ports = [port1, port2]

    # --- Mininet Setup ---
    setLogLevel('info')
    cleanup() # Clean previous runs

    topo = DumbbellTopo()
    net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink, autoSetMacs=True)
    net_running = False
    h1, h2 = None, None # Initialize hosts
    active_iperf_pid = None
    active_port = None

    def stop_network(signum=None, frame=None):
        print("\nStopping network and cleaning up...")
        if h1: h1.cmd('pkill -f iperf') # Stop iperf client
        if h2: h2.cmd('pkill -f iperf') # Stop iperf server
        if net_running:
            try:
                net.stop()
            except Exception as e:
                print(f"Error stopping Mininet: {e}")
        if log_file:
            log_file.close()
        cleanup()
        print("Exiting.")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, stop_network) # Ctrl+C
    signal.signal(signal.SIGTERM, stop_network) # kill

    try:
        print("Starting Mininet network with oscillating conditions...")
        net.start()
        net_running = True
        print("Network started.")

        h1, h2 = net.get('h1', 'h2')
        h1_ip = h1.IP()
        h2_ip = h2.IP()
        print(f"h1 IP: {h1_ip}, h2 IP: {h2_ip}")

        # Start with CUBIC-favorable network
        current_network = "cubic"
        update_network_conditions(net, current_network)

        # --- Set Initial TCP Algorithm ---
        # Start with CUBIC as default
        current_algo = "cubic"
        if args.fixed_algo and args.fixed_algo.lower() in ["reno", "cubic"]:
            current_algo = args.fixed_algo.lower()
        
        print(f"\n[+] Setting TCP Congestion Control on h1 to: {current_algo}")
        h1.cmd(f"sysctl -w net.ipv4.tcp_congestion_control={current_algo}")
        verified_algo = h1.cmd("sysctl net.ipv4.tcp_congestion_control").strip().split('=')[-1].strip()
        print(f"Current TCP Algorithm on h1: {verified_algo}")
        if verified_algo != current_algo:
            print(f"Warning: Failed to set TCP algorithm to {current_algo}, using {verified_algo}")
            current_algo = verified_algo

        # --- Start iperf servers on h2 ---
        print(f"Starting iperf servers on h2 (ports {port1}, {port2})...")
        h2.cmd(f'iperf -s -p {port1} > /dev/null &')
        h2.cmd(f'iperf -s -p {port2} > /dev/null &')
        time.sleep(1) # Wait for servers to start

        # --- Start initial iperf client on h1 ---
        active_port = port1
        print(f"Starting iperf client on h1 -> {h2_ip}:{active_port} (algo: {current_algo})...")
        h1.cmd(f'iperf -c {h2_ip} -p {active_port} -t 9999999 -i {args.interval} > /dev/null &')
        time.sleep(1) # Wait for client connection to establish
        
        # Get the PID of the iperf client
        pid_output = h1.cmd('echo $!')
        try:
            active_iperf_pid = int(pid_output.strip())
            print(f"Initial iperf client PID: {active_iperf_pid} on port {active_port}")
        except ValueError:
            print(f"Error: Could not get PID for initial iperf client. Output: '{pid_output}'")
            stop_network()

        # Variables for oscillation timing
        last_oscillation_time = time.time()
        oscillation_count = 0
        
        # --- Main Data Collection Loop ---
        print(f"\nStarting data collection with oscillating network conditions...")
        print(f"Network will switch every {args.oscillation} seconds.")
        print(f"Data collected every {args.interval} seconds, keeping last {args.history} points.")
        print("Press Ctrl+C to stop.")
        
        while True:
            # Check if it's time to oscillate the network
            current_time = time.time()
            if current_time - last_oscillation_time >= args.oscillation:
                # Toggle network conditions
                oscillation_count += 1
                if current_network == "cubic":
                    current_network = "reno"
                else:
                    current_network = "cubic"
                    
                print(f"\n[+] Oscillation #{oscillation_count}: Switching to {current_network.upper()}-favorable network")
                update_network_conditions(net, current_network)
                last_oscillation_time = current_time

            # Get current TCP stats from h1 for the connection to h2
            ss_cmd_output = h1.cmd(f'ss -tinem -o state established dst {h2_ip}')
            
            # Parse the output
            parsed_data_list = parse_ss_output(ss_cmd_output, h1_ip, h2_ip, current_algo, active_port)
            
            if parsed_data_list:
                # Add the first found connection's data to our history deque
                connection_data = parsed_data_list[0]
                data_history.append(connection_data)
                
                # Log data if requested
                if log_file:
                    # Calculate loss ratio (bytes_retrans / bytes_sent)
                    if connection_data.get('bytes_sent', 0) > 0:
                        loss_ratio = connection_data.get('bytes_retrans', 0) / connection_data.get('bytes_sent', 1)
                    else:
                        loss_ratio = 0
                        
                    log_file.write(f"{connection_data['timestamp']},{current_network},{current_algo},"
                                   f"{connection_data.get('rtt', 0)},{connection_data.get('cwnd', 0)},"
                                   f"{connection_data.get('bytes_acked', 0)},{connection_data.get('segs_out', 0)},"
                                   f"{loss_ratio}\n")
                    log_file.flush()  # Ensure data is written even if program crashes

                if len(data_history) > 0:
                    print(f"History has {len(data_history)} entries. Network: {current_network.upper()}, TCP: {current_algo.upper()}")
                    print(f"RTT: {connection_data.get('rtt', 0):.2f}ms, CWND: {connection_data.get('cwnd', 0)}, Bytes acked: {connection_data.get('bytes_acked', 0)}")

                # Check if we should run prediction and potentially switch algorithms
                if len(data_history) >= args.history and not args.fixed_algo:
                    recommended_algo = predict_best_algorithm(data_history)
                    print(f"Recommended algorithm: {recommended_algo}")
                    
                    if recommended_algo != current_algo and recommended_algo in ['reno', 'cubic']:
                        print(f"\nRecommendation changed: {current_algo} -> {recommended_algo}. Initiating switch.")

                        # A. Change system default TCP algorithm
                        print(f"[+] Setting system TCP to: {recommended_algo}")
                        h1.cmd(f"sysctl -w net.ipv4.tcp_congestion_control={recommended_algo}")
                        verified_algo = h1.cmd("sysctl net.ipv4.tcp_congestion_control").strip().split('=')[-1].strip()
                        
                        if verified_algo == recommended_algo:
                            print(f"System TCP successfully set to {verified_algo}")
                            current_algo = verified_algo
                        else:
                            print(f"ERROR: Failed to set system TCP to {recommended_algo}, remains {verified_algo}. Aborting switch.")
                            time.sleep(args.interval)
                            continue

                        # Store info of the connection to be stopped
                        old_iperf_pid = active_iperf_pid
                        old_port = active_port

                        # Determine the new port to use
                        new_port = port2 if active_port == port1 else port1

                        # B. Start new iperf client on the other port
                        print(f"Starting NEW iperf client on h1 -> {h2_ip}:{new_port} (algo: {current_algo})...")
                        h1.cmd(f'iperf -c {h2_ip} -p {new_port} -t 999999 -i {args.interval} > /dev/null &')
                        time.sleep(1)
                        new_pid_output = h1.cmd('echo $!')
                        try:
                            new_iperf_pid = int(new_pid_output.strip())
                            print(f"New iperf client PID: {new_iperf_pid} on port {new_port}")

                            # C. Update active state
                            active_iperf_pid = new_iperf_pid
                            active_port = new_port

                            # D. Stop the OLD iperf client
                            print(f"Stopping OLD iperf client (PID: {old_iperf_pid}, Port: {old_port})...")
                            if os.path.exists(f"/proc/{old_iperf_pid}"):
                                h1.cmd(f'kill {old_iperf_pid}')
                                print(f"Kill signal sent to PID {old_iperf_pid}.")
                            else:
                                print(f"Old PID {old_iperf_pid} not found, likely already stopped.")

                        except ValueError:
                            print(f"ERROR: Could not get PID for NEW iperf client. Output: '{new_pid_output}'. Switch failed.")
                            # Attempt to kill the potentially orphaned new iperf if PID wasn't captured
                            h1.cmd(f'pkill -f "iperf -c {h2_ip} -p {new_port}"')
                            print("Continuing with the old connection active.")

            else:
                # Handle case where no connection/data was found in this cycle
                print(".", end="", flush=True) # Indicate activity even if no connection parsed

            # Wait for the next interval
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # This cleanup runs on normal exit, Ctrl+C, or error
        stop_network()

if __name__ == '__main__':
    main() 