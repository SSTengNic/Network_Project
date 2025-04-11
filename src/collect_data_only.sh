#!/bin/bash

# TCP Congestion Control Data Collection Only
# This script runs only the data collection step for debugging purposes.

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Create necessary directories
mkdir -p data

# Clean up any previous Mininet instances
echo "Cleaning up previous Mininet instances..."
sudo mn -c

echo "===== Collecting TCP Cubic data ====="
# Run with sudo since Mininet requires root privileges
sudo python3 simplified_data_collection.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to collect data"
    exit 1
fi

echo "Data collection completed successfully!"
echo "Output files are in the data/ directory"

# Final cleanup
echo "Cleaning up Mininet..."
sudo mn -c 