#!/usr/bin/env python

"""
Process TCP Data Collected from Mininet Simulations

This script processes raw TCP data collected using the ss command in Mininet
simulations and converts it to a format compatible with our existing datasets
(reno1.log.csv and cubic1.log.csv).
"""

import re
import csv
import os
import sys
import pandas as pd
from datetime import datetime


def parse_ss_output(raw_file):
    """
    Parse the raw ss command output and extract relevant TCP metrics.
    
    Args:
        raw_file: Path to the raw ss output file
        
    Returns:
        A list of dictionaries containing the extracted TCP metrics
    """
    data = []
    current_entry = {}
    
    with open(raw_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # New connection entry starts with a socket summary
            if line.startswith('tcp'):
                # Save previous entry if it exists
                if current_entry and 'wscale' in current_entry:
                    data.append(current_entry)
                
                # Start a new entry
                current_entry = {}
                
                # Extract basic socket information
                parts = line.split()
                for part in parts:
                    if ':' in part and not part.startswith('timer'):
                        current_entry['local_addr'] = part
            
            # Extract detailed metrics from the info line
            elif line.startswith('	'):  # Starts with a tab
                # Parse RTT
                rtt_match = re.search(r'rtt:([0-9.]+)/', line)
                if rtt_match:
                    current_entry['rtt'] = float(rtt_match.group(1))
                
                # Parse RTO
                rto_match = re.search(r'rto:([0-9]+)', line)
                if rto_match:
                    current_entry['rto'] = int(rto_match.group(1))
                
                # Parse MSS
                mss_match = re.search(r'mss:([0-9]+)', line)
                if mss_match:
                    current_entry['mss'] = int(mss_match.group(1))
                
                # Parse window scaling
                wscale_match = re.search(r'wscale:([0-9]+)', line)
                if wscale_match:
                    current_entry['wscale'] = int(wscale_match.group(1))
                
                # Parse congestion window
                cwnd_match = re.search(r'cwnd:([0-9]+)', line)
                if cwnd_match:
                    current_entry['cwnd'] = int(cwnd_match.group(1))
                
                # Parse ssthresh
                ssthresh_match = re.search(r'ssthresh:([0-9]+)', line)
                if ssthresh_match:
                    current_entry['ssthresh'] = int(ssthresh_match.group(1))
                
                # Parse bytes sent
                bytes_sent_match = re.search(r'bytes_sent:([0-9]+)', line)
                if bytes_sent_match:
                    current_entry['bytes_sent'] = int(bytes_sent_match.group(1))
                
                # Parse bytes retransmitted
                bytes_retrans_match = re.search(r'bytes_retrans:([0-9]+)', line)
                if bytes_retrans_match:
                    current_entry['bytes_retrans'] = int(bytes_retrans_match.group(1))
                
                # Parse bytes acknowledged
                bytes_acked_match = re.search(r'bytes_acked:([0-9]+)', line)
                if bytes_acked_match:
                    current_entry['bytes_acked'] = int(bytes_acked_match.group(1))
                
                # Parse segments out
                segs_out_match = re.search(r'segs_out:([0-9]+)', line)
                if segs_out_match:
                    current_entry['segs_out'] = int(segs_out_match.group(1))
                
                # Parse segments in
                segs_in_match = re.search(r'segs_in:([0-9]+)', line)
                if segs_in_match:
                    current_entry['segs_in'] = int(segs_in_match.group(1))
                
                # Parse data segments out
                data_segs_out_match = re.search(r'data_segs_out:([0-9]+)', line)
                if data_segs_out_match:
                    current_entry['data_segs_out'] = int(data_segs_out_match.group(1))
                
                # Parse last receive
                lastrcv_match = re.search(r'lastsnd:([0-9]+)\s+lastrcv:([0-9]+)', line)
                if lastrcv_match:
                    current_entry['lastrcv'] = int(lastrcv_match.group(2))
                
                # Parse delivered
                delivered_match = re.search(r'delivered:([0-9]+)', line)
                if delivered_match:
                    current_entry['delivered'] = int(delivered_match.group(1))
                
                # Parse receive window
                rcv_space_match = re.search(r'rcv_space:([0-9]+)', line)
                if rcv_space_match:
                    current_entry['rcv_space'] = int(rcv_space_match.group(1))
                
                # Parse receive ssthresh
                rcv_ssthresh_match = re.search(r'rcv_ssthresh:([0-9]+)', line)
                if rcv_ssthresh_match:
                    current_entry['rcv_ssthresh'] = int(rcv_ssthresh_match.group(1))
                
                # Parse PMTU
                pmtu_match = re.search(r'pmtu:([0-9]+)', line)
                if pmtu_match:
                    current_entry['pmtu'] = int(pmtu_match.group(1))
                
                # Parse receive MSS
                rcvmss_match = re.search(r'rcvmss:([0-9]+)', line)
                if rcvmss_match:
                    current_entry['rcvmss'] = int(rcvmss_match.group(1))
                
                # Parse advertised MSS
                advmss_match = re.search(r'advmss:([0-9]+)', line)
                if advmss_match:
                    current_entry['advmss'] = int(advmss_match.group(1))
    
    # Add the last entry if it exists
    if current_entry and 'wscale' in current_entry:
        data.append(current_entry)
    
    return data


def format_compatible_with_existing(parsed_data, congestion_algo):
    """
    Format the parsed data to be compatible with existing datasets.
    
    Args:
        parsed_data: List of dictionaries containing parsed TCP metrics
        congestion_algo: The congestion control algorithm used (reno or cubic)
        
    Returns:
        A pandas DataFrame formatted like our existing datasets
    """
    # Define the columns we need in the same order as our existing datasets
    columns = [
        'wscale', 'rto', 'rtt', 'mss', 'pmtu', 'rcvmss', 'advmss', 'cwnd', 'ssthresh',
        'bytes_sent', 'bytes_retrans', 'bytes_acked', 'segs_out', 'segs_in', 'data_segs_out',
        'lastrcv', 'delivered', 'rcv_space', 'rcv_ssthresh'
    ]
    
    # Initialize an empty DataFrame with these columns
    df = pd.DataFrame(columns=columns)
    
    # Fill in the DataFrame with the parsed data
    for entry in parsed_data:
        row = {}
        for col in columns:
            if col in entry:
                row[col] = entry[col]
            else:
                # Use default values for missing metrics
                if col == 'wscale':
                    row[col] = 6.6  # Common default
                elif col == 'ssthresh':
                    row[col] = 24  # Common default
                elif col == 'rcv_space':
                    row[col] = 14480  # Common default
                elif col == 'rcv_ssthresh':
                    row[col] = 64088  # Common default
                else:
                    row[col] = 0
        
        # Append the row to the DataFrame
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    # Calculate loss ratio
    df['loss_ratio'] = df['bytes_retrans'] / df['bytes_sent'].replace(0, 1)
    
    # Add TCP type column
    df['tcp_type'] = congestion_algo
    
    return df


def main():
    """
    Main function to process raw TCP data files.
    
    Usage: python process_tcp_data.py <raw_file> <output_csv> <congestion_algo>
    """
    if len(sys.argv) != 4:
        print("Usage: python process_tcp_data.py <raw_file> <output_csv> <congestion_algo>")
        sys.exit(1)
    
    raw_file = sys.argv[1]
    output_csv = sys.argv[2]
    congestion_algo = sys.argv[3]
    
    if not os.path.exists(raw_file):
        print(f"Error: Raw file {raw_file} does not exist.")
        sys.exit(1)
    
    if congestion_algo not in ['reno', 'cubic']:
        print(f"Error: Congestion algorithm {congestion_algo} not supported. Use 'reno' or 'cubic'.")
        sys.exit(1)
    
    # Parse the raw ss output
    parsed_data = parse_ss_output(raw_file)
    
    if not parsed_data:
        print("Error: No valid TCP data found in the raw file.")
        sys.exit(1)
    
    # Format the data to be compatible with existing datasets
    df = format_compatible_with_existing(parsed_data, congestion_algo)
    
    # Save to CSV with semicolon delimiter
    df.to_csv(output_csv, sep=';', index=False)
    
    print(f"Successfully processed {len(df)} TCP data points and saved to {output_csv}")


if __name__ == "__main__":
    main() 