#!/usr/bin/env python3

"""
Mininet script to continuously monitor TCP connection metrics
for the last N seconds and store them in memory.
"""

import time
import re
import argparse
import sys
from collections import deque
import signal

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
import os

MODEL_FEATURE_COLUMNS = ['wscale', 'rto', 'rtt', 'mss', 'rcvmss', 'advmss', 'cwnd', 'ssthresh',
       'bytes_acked', 'segs_out', 'segs_in', 'data_segs_out', 'lastrcv',
       'rcv_ssthresh', 'tcp_type', 'timestamp'] # tcp_type is also used for the switch model, but will be passed into predict_best_algorithm separately

class LSTM_pt(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_pt, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # LSTM cell
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)

        # Linear layer for final prediction
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs, cell_state=None, hidden_state=None):
        # Forward pass through the LSTM cell
        if hidden_state is None or cell_state is None:
            device = inputs.device
            hidden_state = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(device)
            cell_state = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(device)
        hidden = (cell_state, hidden_state)
        output, new_memory = self.lstm(inputs, hidden)
        cell_state, hidden_state = new_memory
        output = self.linear(output)  # Linear layer on all time steps
        return output, cell_state, hidden_state

class ModelConfig:
    """Configuration for a model including normalization parameters"""
    def __init__(self, model_path, input_size, hidden_size, num_layers, output_size,
                 target_min, target_max, feature_mins=None, feature_maxs=None):
        self.model_path = model_path
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.target_min = target_min
        self.target_max = target_max
        self.feature_mins = feature_mins
        self.feature_maxs = feature_maxs

# Model configurations with correct parameters based on the training code
RENO_CONFIG = ModelConfig(
    model_path="model_files/Reno_weights.pth",
    input_size=15,
    hidden_size=20,
    num_layers=1,
    output_size=1,
    target_min=0.0,  # These will be updated with actual values
    target_max=1.0   # These will be updated with actual values
)

CUBIC_CONFIG = ModelConfig(
    model_path="model_files/my_model_weights.pth",
    input_size=15,
    hidden_size=20,
    num_layers=1,
    output_size=1,
    target_min=0.0,  # These will be updated with actual values
    target_max=1.0   # These will be updated with actual values
)

RENO_CUBIC_CONFIG = ModelConfig(
    model_path="model_files/RenoCubic_weights.pth",
    input_size=16,
    hidden_size=20,
    num_layers=1,
    output_size=1,
    target_min=0.0,  # These will be updated with actual values
    target_max=1.0   # These will be updated with actual values
)

def load_normalization_params():
    """
    Load the actual normalization parameters from the training data.
    This should be run once before making predictions.
    """
    # Try to load cached normalization parameters if they exist
    cache_file = "normalization_params.npz"
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        RENO_CONFIG.target_min = float(data['reno_target_min'])
        RENO_CONFIG.target_max = float(data['reno_target_max'])
        CUBIC_CONFIG.target_min = float(data['cubic_target_min'])
        CUBIC_CONFIG.target_max = float(data['cubic_target_max'])
        RENO_CUBIC_CONFIG.target_min = float(data['reno_cubic_target_min'])
        RENO_CUBIC_CONFIG.target_max = float(data['reno_cubic_target_max'])
        print("Loaded normalization parameters from cache")
        return

    # Otherwise, extract from the original training data if available
    try:
        # Load Reno data
        reno_path = "../src/data/reno_tcp.csv"
        if os.path.exists(reno_path):
            reno_df = pd.read_csv(reno_path, delimiter=";")
            RENO_CONFIG.target_min = float(reno_df['loss_ratio'].min())
            RENO_CONFIG.target_max = float(reno_df['loss_ratio'].max())

        # Load Cubic data
        cubic_path = "../src/data/cubic_tcp.csv"
        if os.path.exists(cubic_path):
            cubic_df = pd.read_csv(cubic_path, delimiter=";")
            CUBIC_CONFIG.target_min = float(cubic_df['loss_ratio'].min())
            CUBIC_CONFIG.target_max = float(cubic_df['loss_ratio'].max())

        # Load Switch data
        switch_path = "../src/data/switch_tcp.csv"
        if os.path.exists(switch_path):
            switch_df = pd.read_csv(switch_path, delimiter=";")
            RENO_CUBIC_CONFIG.target_min = float(switch_df['loss_ratio'].min())
            RENO_CUBIC_CONFIG.target_max = float(switch_df['loss_ratio'].max())

        # Cache the parameters for future use
        np.savez(
            cache_file,
            reno_target_min=RENO_CONFIG.target_min,
            reno_target_max=RENO_CONFIG.target_max,
            cubic_target_min=CUBIC_CONFIG.target_min,
            cubic_target_max=CUBIC_CONFIG.target_max,
            reno_cubic_target_min=RENO_CUBIC_CONFIG.target_min,
            reno_cubic_target_max=RENO_CUBIC_CONFIG.target_max
        )

        print("Extracted and cached normalization parameters from training data")

    except Exception as e:
        print(f"Could not load original training data: {e}")
        print("Using default normalization parameters")
        # Set fallback values if we can't read the original data
        # These should be positive values representing realistic loss ratios
        RENO_CONFIG.target_min = 0.0
        RENO_CONFIG.target_max = 0.1  # Assuming 10% max loss ratio
        CUBIC_CONFIG.target_min = 0.0
        CUBIC_CONFIG.target_max = 0.1
        RENO_CUBIC_CONFIG.target_min = 0.0
        RENO_CUBIC_CONFIG.target_max = 0.1

def load_model(config):
    """Load a model based on its configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model with correct parameters
    model = LSTM_pt(
        config.input_size,
        config.hidden_size,
        config.num_layers,
        config.output_size
    ).to(device)

    # Load pre-trained weights
    model.load_state_dict(torch.load(config.model_path, map_location=device))

    # Set model to evaluation mode
    model.eval()

    return model

def normalize_data(data, feature_mins, feature_maxs):
    """Normalize input data based on min and max values."""
    normalized = (data - feature_mins) / (feature_maxs - feature_mins)
    # Replace NaN with 0 (if division by zero occurred)
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized

def predict_loss(model, input_data, config):
    """
    Predict the loss ratio using the specified model.

    Args:
        model: The trained LSTM model
        input_data: Normalized input tensor of shape [1, seq_len, features]
        config: The model's configuration

    Returns:
        float: The predicted loss ratio (non-negative)
    """
    device = input_data.device

    with torch.no_grad():
        # Make prediction
        output, _, _ = model(input_data)

        # Denormalize the output
        output_denorm = output * (config.target_max - config.target_min) + config.target_min

        # Average the predictions across all time steps
        avg_prediction = output_denorm.mean().item()

        # Ensure non-negative loss ratio
        avg_prediction = max(0.0, avg_prediction)

    return avg_prediction

def predict_best_algorithm(last_15_data_points, switching_threshold=0.01):
    """
    Predict which TCP congestion control algorithm will have a lower loss ratio.

    Args:
        current_algorithm (str): Current TCP congestion control algorithm ('reno' or 'cubic')
        last_15_data_points (list/array of dictionaries): Last 15 data points with features for prediction
        switching_threshold (float): Minimum improvement required to switch algorithms

    Returns:
        str: Recommended algorithm ('reno' or 'cubic')
    """
    # Ensure normalization parameters are loaded
    load_normalization_params()

    # Get current algorithm
    current_algorithm = last_15_data_points[-1]['tcp_type']
    # Map the algorithm to numeric value
    algo_map = {'reno': 1, 'cubic': 0}
    if algo_map.get(current_algorithm.lower()) is None:
        raise ValueError("Algorithm must be either 'reno' or 'cubic'")

    # Convert input data to the right format
    # Last 15 data points is expected to be a deque or list of dictionaries
    input_data_list = []
    for data_point in last_15_data_points:
        row = [data_point.get(col) if col != 'tcp_type' else algo_map.get(data_point[col].lower()) for col in MODEL_FEATURE_COLUMNS]
        input_data_list.append(row)
    input_data = np.array(input_data_list)

    # Prepare input
    feature_mins = np.min(input_data, axis=0)
    feature_maxs = np.max(input_data, axis=0)
    normalized_input = normalize_data(input_data, feature_mins, feature_maxs)

    # Drop the tcp_type column for the non-switch models
    tcp_type_col_index = MODEL_FEATURE_COLUMNS.index('tcp_type')
    normalized_input_notype = np.concatenate((normalized_input[:, :tcp_type_col_index], normalized_input[:, tcp_type_col_index+1:]), axis=1)

    # Convert to tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalized_input_tensor = torch.tensor(normalized_input, dtype=torch.float32).unsqueeze(0).to(device)
    normalized_input_notype_tensor = torch.tensor(normalized_input_notype, dtype=torch.float32).unsqueeze(0).to(device)

    # Load the models, make predictions, and return recommendation
    if current_algorithm == 'reno':
        reno_model = load_model(RENO_CONFIG)
        switch_model = load_model(RENO_CUBIC_CONFIG)
        loss_reno = predict_loss(reno_model, normalized_input_notype_tensor, RENO_CONFIG)
        loss_switch = predict_loss(switch_model, normalized_input_tensor, RENO_CUBIC_CONFIG)
        print(f"Predicted loss with Reno: {loss_reno:.6f}")
        print(f"Predicted loss with switch to Cubic: {loss_switch:.6f}")
        if loss_switch < (loss_reno - switching_threshold):
            print(f"Improvement by switching: {loss_reno - loss_switch:.6f}")
            return 'cubic'
        else:
            print(f"Improvement not sufficient: {loss_reno - loss_switch:.6f} < {switching_threshold}")
            return 'reno'

    elif current_algorithm == 'cubic':
        cubic_model = load_model(CUBIC_CONFIG)
        switch_model = load_model(RENO_CUBIC_CONFIG)
        loss_cubic = predict_loss(cubic_model, normalized_input_notype_tensor, CUBIC_CONFIG)
        loss_switch = predict_loss(switch_model, normalized_input_tensor, RENO_CUBIC_CONFIG)
        print(f"Predicted loss with Cubic: {loss_cubic:.6f}")
        print(f"Predicted loss with switch to Reno: {loss_switch:.6f}")
        if loss_switch < (loss_cubic - switching_threshold):
            print(f"Improvement by switching: {loss_cubic - loss_switch:.6f}")
            return 'reno'
        else:
            print(f"Improvement not sufficient: {loss_cubic - loss_switch:.6f} < {switching_threshold}")
            return 'cubic'

# Example usage:
# if __name__ == "__main__":
#     # Example data (should be replaced with actual data)
#     example_data = np.random.rand(15, 15)  # 15 time steps, 15 features

#     print("=== Testing with default threshold (0.01) ===")
#     # Example prediction with Reno
#     print("\nTesting with Reno as current algorithm:")
#     result_reno = predict_best_algorithm('reno', example_data)
#     print(f"Recommended algorithm: {result_reno}")

#     # Example prediction with Cubic
#     print("\nTesting with Cubic as current algorithm:")
#     result_cubic = predict_best_algorithm('cubic', example_data)
#     print(f"Recommended algorithm: {result_cubic}")

#     print("\n=== Testing with higher threshold (0.05) ===")
#     # Example prediction with Reno and higher threshold
#     print("\nTesting with Reno as current algorithm:")
#     result_reno = predict_best_algorithm('reno', example_data, switching_threshold=0.05)
#     print(f"Recommended algorithm: {result_reno}")

#     # Example prediction with Cubic and higher threshold
#     print("\nTesting with Cubic as current algorithm:")
#     result_cubic = predict_best_algorithm('cubic', example_data, switching_threshold=0.05)
#     print(f"Recommended algorithm: {result_cubic}")

# --- Data Storage ---
# A deque automatically maintains a fixed size, discarding oldest entries
# This will be populated by the main loop.
data_history = None

# --- Topology Definition ---
class DumbbellTopo(Topo):
    """Simple dumbbell topology with 2 hosts and configurable bottleneck."""
    def build(self, bw=10, loss=1, delay='20ms', queue=50):
        # Create switches
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')

        # Add hosts
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')

        # Add host links (access links)
        self.addLink(h1, s1, bw=1000, delay='1ms')
        self.addLink(h2, s2, bw=1000, delay='1ms')

        # Add bottleneck link
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

    Returns:
        list: A list of dictionaries, each containing metrics for a found connection.
              Returns an empty list if no relevant connection or metrics are found.
    """
    # Debug: Print the raw ss output and parameters
    # print("======================================\nParse ss output called with:\n{\nss_output:\n", ss_output, "\nh1_ip:\n", h1_ip, "\nh2_ip:\n", h2_ip, "\ncurrent_tcp_algo:\n", current_tcp_algo, "\n}\n======================================")

    connections_data = []
    lines = ss_output.strip().split('\n')
    now = time.time() # Timestamp for this data collection cycle

    # Fields required:
    # wscale;rto;rtt;mss;pmtu;rcvmss;advmss;cwnd;ssthresh;bytes_sent;bytes_retrans;bytes_acked;segs_out;segs_in;data_segs_out;lastrcv;delivered;rcv_space;rcv_ssthresh;
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

        # Check if this line contains the IPs we care about
        is_relevant_connection = (h1_ip in line and f"{h2_ip}:{port}" in line)
        if is_relevant_connection:
            print(f"Relevant connection found: {line}") # Debug: Show relevant connection

        # print(f"Line {i}: {line} - Relevant: {is_relevant_connection}") # Debug: Show line and relevance
        # print(f"Timer string in line: {'timer:(' in lines[i]}") # Debug: Check for timer presence

        # Look for the next line which should contain the metrics
        if is_relevant_connection and i < len(lines) - 1 and 'timer:(' in line: # Metrics line often has 'timer'

            # print("Running metrics extraction...") # Debug: Indicate metrics extraction

            i = i + 1 # Move to the next line for metrics
            metrics_line = lines[i].strip()

            # print(f"Metrics Line {i}: {metrics_line}") # Debug: Show metrics line

            metrics = {'timestamp': now, 'tcp_type': current_tcp_algo}

            # --- Extract Fields ---
            wscale_match = wscale_match_re.search(metrics_line)
            metrics['wscale'] = float(wscale_match.group(1).replace(',', '.')) if wscale_match else 0

            rto_match = rto_match_re.search(metrics_line)
            # print(f"RTO match: {rto_match}") # Debug: Show RTO match
            metrics['rto'] = int(rto_match.group(1)) if rto_match else 0

            rtt_match = rtt_match_re.search(metrics_line)
            # print(f"RTT match: {rtt_match}") # Debug: Show RTT match
            metrics['rtt'] = float(rtt_match.group(1)) if rtt_match else 0.0

            mss_match = mss_match_re.search(metrics_line)
            # print(f"MSS match: {mss_match}") # Debug: Show MSS match
            metrics['mss'] = int(mss_match.group(1)) if mss_match else 0

            pmtu_match = pmtu_match_re.search(metrics_line)
            # print(f"PMTU match: {pmtu_match}") # Debug: Show PMTU match
            metrics['pmtu'] = int(pmtu_match.group(1)) if pmtu_match else 0

            rcvmss_match = rcvmss_match_re.search(metrics_line)
            # print(f"RCVMSS match: {rcvmss_match}") # Debug: Show RCVMSS match
            metrics['rcvmss'] = int(rcvmss_match.group(1)) if rcvmss_match else 0

            advmss_match = advmss_match_re.search(metrics_line)
            # print(f"ADVMSS match: {advmss_match}") # Debug: Show ADVMSS match
            metrics['advmss'] = int(advmss_match.group(1)) if advmss_match else 0

            cwnd_match = cwnd_match_re.search(metrics_line)
            # print(f"CWND match: {cwnd_match}") # Debug: Show CWND match
            metrics['cwnd'] = int(cwnd_match.group(1)) if cwnd_match else 0

            ssthresh_match = ssthresh_match_re.search(metrics_line)
            # print(f"SSTHRESH match: {ssthresh_match}") # Debug: Show SSTHRESH match
            metrics['ssthresh'] = int(ssthresh_match.group(1)) if ssthresh_match else 0

            bytes_sent_match = bytes_sent_match_re.search(metrics_line)
            # print(f"Bytes sent match: {bytes_sent_match}") # Debug: Show bytes sent match
            metrics['bytes_sent'] = int(bytes_sent_match.group(1)) if bytes_sent_match else 0

            bytes_retrans_match = bytes_retrans_match_re.search(metrics_line)
            # print(f"Bytes retrans match: {bytes_retrans_match}") # Debug: Show bytes retrans match
            metrics['bytes_retrans'] = int(bytes_retrans_match.group(1)) if bytes_retrans_match else 0

            bytes_acked_match = bytes_acked_match_re.search(metrics_line)
            # print(f"Bytes acked match: {bytes_acked_match}") # Debug: Show bytes acked match
            metrics['bytes_acked'] = int(bytes_acked_match.group(1)) if bytes_acked_match else 0

            segs_out_match = segs_out_match_re.search(metrics_line)
            # print(f"Segs out match: {segs_out_match}") # Debug: Show segs out match
            metrics['segs_out'] = int(segs_out_match.group(1)) if segs_out_match else 0

            segs_in_match = segs_in_match_re.search(metrics_line)
            # print(f"Segs in match: {segs_in_match}") # Debug: Show segs in match
            metrics['segs_in'] = int(segs_in_match.group(1)) if segs_in_match else 0

            data_segs_out = data_segs_out_match_re.search(metrics_line)
            # print(f"Data segs out match: {data_segs_out}") # Debug: Show data segs out match
            metrics['data_segs_out'] = int(data_segs_out.group(1)) if data_segs_out else 0

            lastrcv_match = lastrcv_match_re.search(metrics_line)
            # print(f"Last received match: {lastrcv_match}") # Debug: Show last received match
            metrics['lastrcv'] = int(lastrcv_match.group(1)) if lastrcv_match else 0

            delivered_match = delivered_match_re.search(metrics_line)
            # print(f"Delivered match: {delivered_match}") # Debug: Show delivered match
            metrics['delivered'] = int(delivered_match.group(1)) if delivered_match else 0

            recv_space_match = rcv_space_match_re.search(metrics_line)
            # print(f"Receive space match: {recv_space_match}") # Debug: Show receive space match
            metrics['rcv_space'] = int(recv_space_match.group(1)) if recv_space_match else 0

            rcv_ssthresh_match = rcv_ssthresh_match_re.search(metrics_line)
            # print(f"Receive ssthresh match: {rcv_ssthresh_match}") # Debug: Show receive ssthresh match
            metrics['rcv_ssthresh'] = int(rcv_ssthresh_match.group(1)) if rcv_ssthresh_match else 0

            connections_data.append(metrics)
            # For simplicity, let's just take the first relevant connection found
            # break # Uncomment if you only want the first match per cycle

        i += 1

    return connections_data

def sim_algo(tcp_algo, filename, n_datapoints, args):
    """
    Simulates traffic and with the TCP Reno or Cubic algorithms and measures network data
    for an amount of time and saves the data to a file.

    Args:
        tcp_algo (str): The name of the tcp congestion control algorithm to use.
        filename (str): The name of the file to write the data to.
        n_datapoints (int): Number of network data points to collect.
        args (args object): Options specified from the cli.

    Returns:
        None
    """
    topo = DumbbellTopo(bw=args.bw, loss=args.loss, delay=args.delay, queue=args.qsize)
    net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink, autoSetMacs=True)
    h1, h2 = None, None # Initialize hosts
    net_running = False
    data_history = deque(maxlen=n_datapoints)
    try:
        print("Starting Mininet network for Reno test...")
        net.start()
        net_running = True
        print("Network started.")
        h1, h2 = net.get('h1', 'h2')
        h1_ip = h1.IP()
        h2_ip = h2.IP()
        print(f"h1 IP: {h1_ip}, h2 IP: {h2_ip}")

        # --- Set TCP Algorithm ---
        print(f"\n[+] Setting TCP Congestion Control on h1 to: {tcp_algo}")
        h1.cmd(f"sysctl -w net.ipv4.tcp_congestion_control={tcp_algo}")
        current_algo = h1.cmd("sysctl net.ipv4.tcp_congestion_control").strip().split('=')[-1].strip()
        print(f"Current TCP Algorithm on h1: {current_algo}")
        if current_algo != tcp_algo:
            print(f"Warning: Failed to set TCP algorithm to {tcp_algo}, using {current_algo}")
            raise ValueError
        print("Starting iperf server on h2...")
        h2.cmd('iperf -s -p 5001 &')
        time.sleep(1) # Wait for server to start

        print("Starting iperf client on h1 -> {h2_ip}:5001 (algo: {current_algo})...")
        # Use -t 9999999 for long run, -i args.interval for reports (though we ignore output)
        h1.cmd(f'iperf -c {h2_ip} -p 5001 -t 9999999 -i {args.interval} > /dev/null &')
        time.sleep(1) # Wait for client connection to establish
        while True:
            ss_cmd_output = h1.cmd(f'ss -tinem -o state established dst {h2_ip}')
            parsed_data_list = parse_ss_output(ss_cmd_output, h1_ip, h2_ip, current_algo, 5001)
            if parsed_data_list:
                data_history.append(parsed_data_list[0])
                if len(data_history) > 0:
                    print(f"History has {len(data_history)} entries. Oldest timestamp: {data_history[0]['timestamp']:.2f}")
                if len(data_history) == n_datapoints:
                    try:
                        with open(filename, 'w') as file:
                            for row in data_history:
                                file.write(str(row) + '\n')
                        print(f"Data from the deque written to '{filename}' successfully.")
                    except IOError as e:
                        print(f"Error writing to file: {e}")
                    data_history.clear()
                    break
            time.sleep(args.interval)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if h1: h1.cmd('pkill -f iperf') # Stop iperf client
        if h2: h2.cmd('pkill -f iperf') # Stop iperf server
        if net_running:
            try:
                net.stop()
            except Exception as e:
                print(f"Error stopping Mininet: {e}")
        cleanup()
        print("Exiting.")
        time.sleep(10)

# --- Main Execution ---
def main():
    global data_history # Allow modification of the global deque

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Mininet TCP Data Collector (In-Memory)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Data collection interval in seconds (default: 1.0)')
    parser.add_argument('--history', type=int, default=15,
                        help='Number of data points (seconds) to keep in memory (default: 15)')
    parser.add_argument('--bw', type=int, default=10,
                        help='Bottleneck bandwidth in Mbps (default: 10)')
    parser.add_argument('--loss', type=float, default=1,
                        help='Bottleneck loss percentage (default: 1)')
    parser.add_argument('--delay', type=str, default='20ms',
                        help='Bottleneck delay (default: 20ms)')
    parser.add_argument('--qsize', type=int, default=50,
                        help='Bottleneck max queue size (packets) (default: 50)')
    parser.add_argument('--tcp_algo', type=str, default='cubic',
                        help='TCP congestion control algorithm to set on h1 (default: cubic)')
    parser.add_argument('--save_history', type=bool, default=False,
                        help='TCP congestion control algorithm to set on h1 (default: cubic)')
    args = parser.parse_args()

    if args.interval <= 0:
        print("Error: Interval must be positive.")
        sys.exit(1)
    if args.history <= 0:
        print("Error: History size must be positive.")
        sys.exit(1)

    if args.save_history:
        n_datapoints = 100
        sim_algo("reno", "./src/data/reno_log.txt", n_datapoints, args)
        sim_algo("cubic", "./src/data/cubic_log.txt", n_datapoints, args)
        data_history_full = deque(maxlen=n_datapoints)

    # Initialize the deque
    data_history = deque(maxlen=args.history)
    print(f"Config: Interval={args.interval}s, History={args.history} samples, BW={args.bw}Mbps, Loss={args.loss}%, TCP={args.tcp_algo}")

    # Define ports to use
    port1 = 5001
    port2 = 5002
    ports = [port1, port2]

    # --- Mininet Setup ---
    setLogLevel('info')
    cleanup() # Clean previous runs

    topo = DumbbellTopo(bw=args.bw, loss=args.loss, delay=args.delay, queue=args.qsize)
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
        cleanup()
        print("Exiting.")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, stop_network) # Ctrl+C
    signal.signal(signal.SIGTERM, stop_network) # kill

    try:
        print("Starting Mininet network...")
        net.start()
        net_running = True
        print("Network started.")

        # Print connections for verification
        # print("*** Dumping host connections")
        # dumpNodeConnections(net.hosts)
        # print("*** Testing network connectivity")
        # net.pingAll()

        h1, h2 = net.get('h1', 'h2')
        h1_ip = h1.IP()
        h2_ip = h2.IP()
        print(f"h1 IP: {h1_ip}, h2 IP: {h2_ip}")

        # --- Set TCP Algorithm ---
        print(f"\n[+] Setting TCP Congestion Control on h1 to: {args.tcp_algo}")
        h1.cmd(f"sysctl -w net.ipv4.tcp_congestion_control={args.tcp_algo}")
        current_algo = h1.cmd("sysctl net.ipv4.tcp_congestion_control").strip().split('=')[-1].strip()
        print(f"Current TCP Algorithm on h1: {current_algo}")
        if current_algo != args.tcp_algo:
            print(f"Warning: Failed to set TCP algorithm to {args.tcp_algo}, using {current_algo}")


        # --- Start Persistent Traffic ---
        print(f"Starting iperf servers on h2 (ports {port1}, {port2})...")
        h2.cmd(f'iperf -s -p {port1} > /dev/null &')
        h2.cmd(f'iperf -s -p {port2} > /dev/null &')
        # print("Starting iperf server on h2...")
        # h2.cmd('iperf -s -p 5001 &')
        time.sleep(1) # Wait for server to start

        active_port = port1
        print("Starting iperf client on h1 -> {h2_ip}:{active_port} (algo: {current_algo})...")
        # Use -t 9999999 for long run, -i args.interval for reports (though we ignore output)
        h1.cmd(f'iperf -c {h2_ip} -p {active_port} -t 9999999 -i {args.interval} > /dev/null &')
        time.sleep(1) # Wait for client connection to establish
        # Get the PID of the last background process
        pid_output = h1.cmd('echo $!')
        print("pid_output: ", pid_output)
        try:
            active_iperf_pid = int(pid_output.strip())
            print(f"Initial iperf client PID: {active_iperf_pid} on port {active_port}")
        except ValueError:
            print(f"Error: Could not get PID for initial iperf client. Output: '{pid_output}'")
            stop_network()

        # net.interact() # Debug: Keep the Mininet CLI open for manual control

        # --- Main Data Collection Loop ---
        print(f"\nStarting data collection (every {args.interval}s, keeping last {args.history} points)... Press Ctrl+C to stop.")
        while True:
            # Get current TCP stats from h1 for the connection to h2
            # The '-o' gives memory info, '-n' numeric hosts/ports, '-e' extended details, '-m' memory, -i internal info
            ss_cmd_output = h1.cmd(f'ss -tinem -o state established dst {h2_ip}')

            # print(ss_cmd_output) # Debug: Show raw ss output

            # Parse the output
            parsed_data_list = parse_ss_output(ss_cmd_output, h1_ip, h2_ip, current_algo, active_port)

            # print(parsed_data_list) # Debug: Show parsed data

            if parsed_data_list:
                # Add the first found connection's data to our history deque
                # You could modify this to store data for *all* found connections if needed
                data_history.append(parsed_data_list[0])

                if len(data_history) > 0:
                    print(f"History has {len(data_history)} entries. Oldest timestamp: {data_history[0]['timestamp']:.2f}")

                if args.save_history:
                    data_history_full.append(parsed_data_list[0])
                    if len(data_history_full) == n_datapoints:
                        filename = './src/data/prototype_log.txt'
                        try:
                            with open(filename, 'w') as file:
                                for row in data_history_full:
                                    file.write(str(row) + '\n')
                            print(f"Data from the deque written to '{filename}' successfully.")
                        except IOError as e:
                            print(f"Error writing to file: {e}")
                        break

                if len(data_history) >= args.history:
                    recommended_algo = predict_best_algorithm(data_history)
                    print(f"Recommended algorithm: {recommended_algo}")
                    if recommended_algo != current_algo and recommended_algo in ['reno', 'cubic']:
                        print(f"\nRecommendation changed: {current_algo} -> {recommended_algo}. Initiating switch.")

                        # A. Change system default TCP algorithm
                        print(f"[+] Setting system TCP to: {recommended_algo}")
                        h1.cmd(f"sysctl -w net.ipv4.tcp_congestion_control={recommended_algo}")
                        current_algo = h1.cmd("sysctl net.ipv4.tcp_congestion_control").strip().split('=')[-1].strip()
                        if current_algo == recommended_algo:
                            print(f"System TCP successfully set to {current_algo}")
                        else:
                            print(f"ERROR: Failed to set system TCP to {recommended_algo}, remains {verify_algo}. Aborting switch.")
                            # Skip the rest of the switch logic for this cycle
                            time.sleep(args.interval) # Still wait
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

                            # D. Stop the OLD iperf client (after a brief moment for new one to connect)
                            print(f"Stopping OLD iperf client (PID: {old_iperf_pid}, Port: {old_port})...")
                            # Check if process exists before killing
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

                    # else: print(f"Recommendation ({recommended_algo}) matches current ({current_algo}). No switch needed.")


            else:
                # Handle case where no connection/data was found in this cycle
                 print(".", end="", flush=True) # Indicate activity even if no connection parsed
                 # Optionally append a placeholder if you need constant history length
                 # data_history.append({'timestamp': time.time(), 'cwnd': -1, ... other fields: 0 or None})


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