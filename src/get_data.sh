#!/bin/bash

# Script to run only TCP Reno switching to Cubic experiment

# Default values
DURATION=500
SWITCH_INTERVAL=30
BOTTLENECK_BW=20
LOSS_RATE=0

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

echo "=========================================================="
echo "Starting TCP Reno→Cubic switching experiment"
echo "Duration: $DURATION seconds"
echo "Switch at: $SWITCH_INTERVAL seconds"
echo "Network: ${BOTTLENECK_BW}Mbps bottleneck with ${LOSS_RATE}% loss"
echo "=========================================================="


sudo python3 ./single_multi_protocol_experiment.py \
  --mode="cubic" \
  --duration="$DURATION" \
  --bw="$BOTTLENECK_BW" \
  --loss="$LOSS_RATE"

# # Run just the switch experiment
# sudo python3 ./single_multi_protocol_experiment.py \
#   --mode="reno" \
#   --duration="$DURATION" \
#   --interval="$SWITCH_INTERVAL" \
#   --bw="$BOTTLENECK_BW" \
#   --loss="$LOSS_RATE"

# Check if the experiment was successful
if [ $? -eq 0 ]; then
  echo "Experiment completed successfully!"
else
  echo "Experiment failed."
  exit 1
fi

echo "Experiment complete."
