#!/bin/bash

# Script to run three TCP congestion control experiments with protocol switching:
# 1. TCP Reno only (60 seconds)
# 2. TCP Cubic only (60 seconds)
# 3. TCP Reno switching to Cubic (30 seconds each, 60 seconds total)

# Default values
DURATION=60
SWITCH_INTERVAL=30
BOTTLENECK_BW=10
LOSS_RATE=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --duration=*)
      DURATION="${1#*=}"
      shift
      ;;
    --interval=*)
      SWITCH_INTERVAL="${1#*=}"
      shift
      ;;
    --bw=*)
      BOTTLENECK_BW="${1#*=}"
      shift
      ;;
    --loss=*)
      LOSS_RATE="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--duration=60] [--interval=30] [--bw=10] [--loss=1]"
      exit 1
      ;;
  esac
done

# Create data directory if it doesn't exist
mkdir -p data

# Function to run an experiment
run_experiment() {
  local mode=$1
  local description=$2
  
  echo "=========================================================="
  echo "Starting experiment: $description"
  echo "Duration: $DURATION seconds"
  echo "Network: ${BOTTLENECK_BW}Mbps bottleneck with ${LOSS_RATE}% loss"
  if [ "$mode" == "switch" ]; then
    echo "Switch interval: $SWITCH_INTERVAL seconds"
  fi
  echo "=========================================================="
  
  # Run the experiment with Python script in src directory
  sudo python3 src/single_multi_protocol_experiment.py \
    --mode="$mode" \
    --duration="$DURATION" \
    --interval="$SWITCH_INTERVAL" \
    --bw="$BOTTLENECK_BW" \
    --loss="$LOSS_RATE"
  
  # Check if the experiment was successful
  if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!"
    
    # Find the latest cwnd CSV file
    CSV_FILE=$(ls -t data/cwnd_"$mode"_*.csv 2>/dev/null | head -n 1)
    
    if [ -n "$CSV_FILE" ]; then
      echo "CWND data saved to: $CSV_FILE"
      
      # Generate a plot for the data
      echo "Generating cwnd plot..."
      python3 -c "
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df = pd.read_csv('$CSV_FILE')

# Create plot
plt.figure(figsize=(12, 6))

# If switch mode, plot both algorithms separately
if '$mode' == 'switch':
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        plt.plot(algo_df['relative_time'], algo_df['cwnd'], 
                 label=f'{algo.upper()} (avg={algo_df[\"cwnd\"].mean():.1f})')
    
    # Add vertical line at switch point
    plt.axvline(x=$SWITCH_INTERVAL, color='black', linestyle='--', 
                label='Algorithm Switch Point')
else:
    plt.plot(df['relative_time'], df['cwnd'], label='${mode^^}')
    # Add trendline
    try:
        z = np.polyfit(df['relative_time'], df['cwnd'], 1)
        p = np.poly1d(z)
        plt.plot(df['relative_time'], p(df['relative_time']), 'r--', 
                 label=f'Trend (avg={df[\"cwnd\"].mean():.1f})')
    except:
        print('Could not create trendline')

plt.title('TCP ${mode^^} Congestion Window')
plt.xlabel('Time (s)')
plt.ylabel('Congestion Window (cwnd)')
plt.grid(True)
plt.legend()
plt.savefig('data/${mode}_cwnd_plot.png')
plt.close()
print('Plot saved to data/${mode}_cwnd_plot.png')
"
      
      # Display basic cwnd statistics
      echo "CWND Statistics:"
      python3 -c "
import pandas as pd
df = pd.read_csv('$CSV_FILE')
print(f'Max cwnd: {df[\"cwnd\"].max()}')
print(f'Min cwnd: {df[\"cwnd\"].min()}')
print(f'Avg cwnd: {df[\"cwnd\"].mean():.2f}')
print(f'Data points: {len(df)}')

if '${mode}' == 'switch':
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        print(f'\n{algo.upper()} Stats (n={len(algo_df)}):')
        print(f'  Max cwnd: {algo_df[\"cwnd\"].max()}')
        print(f'  Min cwnd: {algo_df[\"cwnd\"].min()}')
        print(f'  Avg cwnd: {algo_df[\"cwnd\"].mean():.2f}')
"
    else
      echo "No CWND data file found for this experiment."
    fi
  else
    echo "Experiment failed."
    return 1
  fi
}

# Run all three experiments
echo "Running TCP congestion control experiments"
echo "Focus: Properly showing protocol-specific congestion window behavior"
echo "Network parameters: ${BOTTLENECK_BW}Mbps bottleneck with ${LOSS_RATE}% loss"

# 1. Run TCP Reno only
run_experiment "reno" "TCP Reno Only"

# 2. Run TCP Cubic only
run_experiment "cubic" "TCP Cubic Only"

# 3. Run TCP Reno switching to Cubic
run_experiment "switch" "TCP Reno switching to TCP Cubic after $SWITCH_INTERVAL seconds"

# Combine all plots for comparison if available
if [ -f data/reno_cwnd_plot.png ] && [ -f data/cubic_cwnd_plot.png ] && [ -f data/switch_cwnd_plot.png ]; then
  echo "Creating comparison of all three experiments..."
  python3 -c "
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

plt.figure(figsize=(12, 8))

# Function to read and plot cwnd data
def plot_data(mode, linestyle, color, label):
    files = sorted(glob.glob(f'data/cwnd_{mode}_*.csv'))
    if not files:
        return False
    
    # Get the most recent file
    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    
    # For switch mode, handle differently to show both algorithms
    if mode == 'switch':
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            if algo == 'reno':
                plt.plot(algo_df['relative_time'], algo_df['cwnd'], 
                         linestyle='-', color='green', label=f'Switch: RENO phase')
            else:
                plt.plot(algo_df['relative_time'], algo_df['cwnd'], 
                         linestyle='-', color='purple', label=f'Switch: CUBIC phase')
    else:
        # Plot single algorithm data
        plt.plot(df['relative_time'], df['cwnd'], linestyle=linestyle, color=color, label=label)
    return True

# Plot each experiment
reno_ok = plot_data('reno', '-', 'blue', 'Pure TCP Reno')
cubic_ok = plot_data('cubic', '-', 'red', 'Pure TCP Cubic')
switch_ok = plot_data('switch', '-', None, None)  # Special handling in function

if reno_ok or cubic_ok or switch_ok:
    # Add vertical line for the switch point if switch experiment ran
    if switch_ok:
        plt.axvline(x=${SWITCH_INTERVAL}, color='black', linestyle='--', label='Switch Point')
    
    plt.title('TCP Congestion Window Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Congestion Window (cwnd)')
    plt.grid(True)
    plt.legend()
    plt.savefig('data/cwnd_comparison.png')
    plt.close()
    print('Saved comparison plot to data/cwnd_comparison.png')
"
fi

echo "All experiments complete."
echo "================================================================="
echo "Results summary:"
for mode in reno cubic switch; do
  if [ -f "data/${mode}_cwnd_plot.png" ]; then
    echo "TCP ${mode^^} plot: data/${mode}_cwnd_plot.png"
  fi
done

if [ -f "data/cwnd_comparison.png" ]; then
  echo "Comparison plot: data/cwnd_comparison.png"
fi
echo "================================================================="
echo "To run experiments with different parameters:"
echo "./src/run_tcp_experiments.sh --duration=90 --interval=45 --bw=5 --loss=2"
echo "=================================================================" 