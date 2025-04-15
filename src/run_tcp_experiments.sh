#!/bin/bash

# Script to run three TCP congestion control experiments:
# 1. TCP Reno only (60 seconds)
# 2. TCP Cubic only (60 seconds)
# 3. TCP Reno switching to Cubic (30 seconds each, 60 seconds total)

# Default values
DURATION=60
SWITCH_INTERVAL=30

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
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--duration=60] [--interval=30]"
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
  if [ "$mode" == "switch" ]; then
    echo "Switch interval: $SWITCH_INTERVAL seconds"
  fi
  echo "=========================================================="
  
  # Run the experiment
  sudo python3 single_multi_protocol_experiment.py --mode="$mode" --duration="$DURATION" --interval="$SWITCH_INTERVAL"
  
  # Check if the experiment was successful
  if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!"
    echo "Results can be found in the data directory."
    ls -la data/"$mode"_tcp_*.csv
    
    # Display a sample of the results
    echo "Sample of the results:"
    head -n 5 $(ls -t data/"$mode"_tcp_*.csv | head -n 1)
    
    if [ "$mode" == "switch" ]; then
      # Count the number of records for each TCP algorithm
      echo "Data points by TCP algorithm:"
      CSV_FILE=$(ls -t data/"$mode"_tcp_*.csv | head -n 1)
      echo "Reno data points: $(grep -c "reno" $CSV_FILE)"
      echo "Cubic data points: $(grep -c "cubic" $CSV_FILE)"
    fi
  else
    echo "Experiment failed."
    return 1
  fi
}

# Run all three experiments
echo "Running TCP experiments with total duration of $DURATION seconds"

# 1. Run TCP Reno only
run_experiment "reno" "TCP Reno Only"

# 2. Run TCP Cubic only
run_experiment "cubic" "TCP Cubic Only"

# 3. Run TCP Reno switching to Cubic
run_experiment "switch" "TCP Reno switching to TCP Cubic after $SWITCH_INTERVAL seconds"

echo "All experiments complete."
echo "================================================================="
echo "Results summary:"
echo "TCP Reno data: $(ls -la data/reno_tcp_*.csv | tail -n 1)"
echo "TCP Cubic data: $(ls -la data/cubic_tcp_*.csv | tail -n 1)"
echo "TCP Switch data: $(ls -la data/switch_tcp_*.csv | tail -n 1)"
echo "=================================================================" 