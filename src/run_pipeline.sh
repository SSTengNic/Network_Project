#!/bin/bash

# TCP Congestion Control Optimization Pipeline
# This script runs the complete pipeline: collect data, train model, and make predictions.

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Create necessary directories
mkdir -p data models plots

# Clean up any previous Mininet instances
echo "Cleaning up previous Mininet instances..."
sudo mn -c

echo "===== Step 1: Collecting TCP Cubic data ====="
# Run with sudo since Mininet requires root privileges
sudo python3 simplified_data_collection.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to collect data"
    exit 1
fi
echo "Data collection completed"

echo ""
echo "===== Step 2: Training model on collected data ====="
python3 simplified_train_model.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to train model"
    exit 1
fi
echo "Model training completed"

echo ""
echo "===== Step 3: Making predictions using trained model ====="
python3 simplified_predictor.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to make predictions"
    exit 1
fi
echo "Prediction completed"

# Final cleanup
echo "Cleaning up Mininet..."
sudo mn -c

echo ""
echo "===== Pipeline execution completed successfully ====="
echo "Results:"
echo "- Raw data and processed CSV: data/"
echo "- Trained model: models/cubic_model.pt"
echo "- Visualizations: plots/" 