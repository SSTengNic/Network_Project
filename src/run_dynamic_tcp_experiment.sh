#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Run the multi-protocol data collection script
echo "Starting multi-protocol TCP data collection experiment..."
sudo python3 src/multi_protocol_data_collection.py

# Check if the experiment was successful
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!"
    echo "Results can be found in the data directory."
    ls -la data/multi_tcp_*.csv
else
    echo "Experiment failed."
    exit 1
fi

# Optional: Display a sample of the results
echo "Sample of the results:"
head -n 5 $(ls -t data/multi_tcp_*.csv | head -n 1)

echo "Experiment complete." 