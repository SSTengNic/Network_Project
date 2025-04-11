# TCP Congestion Control Optimization using Deep Learning

This project aims to leverage deep learning to optimize TCP congestion control by dynamically selecting the most suitable congestion control algorithm based on network conditions.

## Overview

The goal is to train neural network models to predict network performance metrics (specifically loss ratios) for TCP congestion control algorithms based on network conditions, and then use these predictions to enhance network performance.

## Simplified Implementation

This repository contains a simplified implementation focusing on collecting data for TCP Cubic and training a model to predict loss ratios.

### Key Components

- `simplified_data_collection.py`: Collects TCP performance data using Mininet with a simple dumbbell topology
- `simplified_train_model.py`: Trains a neural network model to predict loss ratios
- `simplified_predictor.py`: Uses the trained model to predict TCP performance based on current network conditions
- `run_pipeline.sh`: Runs the complete pipeline (data collection, training, and prediction)

## Prerequisites

- Python 3.7+
- Mininet (for network simulation)
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Usage

### Complete Pipeline

To run the complete pipeline:

```bash
./run_pipeline.sh
```

This will:
1. Collect TCP performance data using Mininet
2. Train a neural network model on the collected data
3. Use the trained model to predict TCP performance

### Individual Steps

#### 1. Data Collection

To collect TCP performance data:

```bash
sudo python3 simplified_data_collection.py
```

This will create a Mininet simulation with a dumbbell topology, run traffic between hosts using iperf, and collect TCP metrics. The data will be saved to the `data` directory.

#### 2. Model Training

To train a neural network model:

```bash
python3 simplified_train_model.py
```

This will train a model on the collected data and save it to `models/cubic_model.pt`. Training plots will be saved to the `plots` directory.

#### 3. Performance Prediction

To use the trained model for prediction:

```bash
python3 simplified_predictor.py [model_path]
```

If `model_path` is not specified, it will use the default path `models/cubic_model.pt`.

## Implementation Details

### Data Collection

The `simplified_data_collection.py` script:
- Creates a dumbbell topology with 2 switches and 2 hosts
- Sets up a bottleneck link with 10 Mbps bandwidth, 20ms delay, and 1% loss rate
- Uses iperf to generate traffic between hosts
- Collects TCP metrics using the `ss` command
- Processes the collected data into a CSV file compatible with model training

### Model Training

The `simplified_train_model.py` script:
- Uses a simple neural network with fully connected layers
- Takes TCP metrics (RTT, RTO, CWND, etc.) as input features
- Predicts the loss ratio as the output
- Uses mean squared error (MSE) as the loss function
- Generates plots to visualize training progress and model performance

### Performance Prediction

The `simplified_predictor.py` script:
- Loads the trained model
- Collects current TCP metrics from the system
- Uses the model to predict the expected loss ratio
- Provides an interpretation of the prediction

## Future Work

- Implement data collection for other TCP congestion control algorithms (Reno, BBR, etc.)
- Train models to compare performance across different algorithms
- Develop a dynamic algorithm selector that switches between algorithms based on predictions
- Evaluate the system in more complex network scenarios

## Notes

- The data collection requires root privileges to run Mininet and modify TCP settings
- The current implementation focuses on loss ratio prediction for TCP Cubic
- The processing of raw TCP data may need adjustments based on the format of the `ss` command output on your system 