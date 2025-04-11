#!/bin/bash

# Default values
DURATION=3000
INTERVAL=30

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --duration=*)
      DURATION="${1#*=}"
      shift
      ;;
    --interval=*)
      INTERVAL="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--duration=300] [--interval=30]"
      exit 1
      ;;
  esac
done

# Create data directory if it doesn't exist
mkdir -p data

# Run the multi-protocol data collection script
echo "Starting Reno-Cubic alternating TCP experiment (${DURATION} seconds with ${INTERVAL}-second intervals)..."

# Modify the main() function in the Python script to use our parameters
sed -i "s/raw_log = collect_tcp_data_with_multiple_switches(net, total_duration=300, switch_interval=30)/raw_log = collect_tcp_data_with_multiple_switches(net, total_duration=${DURATION}, switch_interval=${INTERVAL})/" multi_protocol_data_collection.py

# Run the experiment
sudo python3 multi_protocol_data_collection.py

# Check if the experiment was successful
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!"
    echo "Results can be found in the data directory."
    ls -la data/multi_tcp_*.csv
    
    # Optional: Display a sample of the results
    echo "Sample of the results:"
    head -n 5 $(ls -t data/multi_tcp_*.csv | head -n 1)
    
    # Count the number of records for each TCP algorithm
    echo "Data points by TCP algorithm:"
    CSV_FILE=$(ls -t data/multi_tcp_*.csv | head -n 1)
    echo "Reno data points: $(grep -c "reno" $CSV_FILE)"
    echo "Cubic data points: $(grep -c "cubic" $CSV_FILE)"
    
    echo "Total experiment duration: ${DURATION} seconds with switches every ${INTERVAL} seconds"
    echo "Total switches: $(expr ${DURATION} / ${INTERVAL} - 1)"
    
    # Print confirmation of alternating behavior
    echo ""
    echo "The experiment alternated between TCP Reno and TCP Cubic every ${INTERVAL} seconds."
    echo "Check the CSV file for the 'tcp_type' column to see which algorithm was active for each data point."
else
    echo "Experiment failed."
    exit 1
fi

# Restore the original values in the Python script
sed -i "s/raw_log = collect_tcp_data_with_multiple_switches(net, total_duration=${DURATION}, switch_interval=${INTERVAL})/raw_log = collect_tcp_data_with_multiple_switches(net, total_duration=300, switch_interval=30)/" multi_protocol_data_collection.py

echo "Experiment complete." 