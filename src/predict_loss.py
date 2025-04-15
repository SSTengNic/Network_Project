#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import pandas as pd
import os

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

def predict_best_algorithm(current_algorithm, last_15_data_points, switching_threshold=0.01):
    """
    Predict which TCP congestion control algorithm will have a lower loss ratio.
    
    Args:
        current_algorithm (str): Current TCP congestion control algorithm ('reno' or 'cubic')
        last_15_data_points (list/array): Last 15 data points with features for prediction
        switching_threshold (float): Minimum improvement required to switch algorithms
        
    Returns:
        str: Recommended algorithm ('reno' or 'cubic')
    """
    # Ensure normalization parameters are loaded
    load_normalization_params()
    
    # Map the algorithm to numeric value
    algo_map = {'reno': 1, 'cubic': 0}
    current_algo_numeric = algo_map.get(current_algorithm.lower())
    
    if current_algo_numeric is None:
        raise ValueError("Algorithm must be either 'reno' or 'cubic'")
    
    # Load the models
    reno_model = load_model(RENO_CONFIG)
    cubic_model = load_model(CUBIC_CONFIG)
    
    # Convert input data to the right format
    # Last 15 data points is expected to be a numpy array or list of shape [15, features]
    input_data = np.array(last_15_data_points)
    
    # Prepare input for Reno and Cubic models (both use 15 features)
    feature_mins = np.min(input_data, axis=0)
    feature_maxs = np.max(input_data, axis=0)
    normalized_input = normalize_data(input_data, feature_mins, feature_maxs)
    
    # Convert to tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalized_input_tensor = torch.tensor(normalized_input, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Make predictions
    loss_reno = predict_loss(reno_model, normalized_input_tensor, RENO_CONFIG)
    loss_cubic = predict_loss(cubic_model, normalized_input_tensor, CUBIC_CONFIG)
    
    print(f"Predicted loss with Reno: {loss_reno:.6f}")
    print(f"Predicted loss with Cubic: {loss_cubic:.6f}")
    
    # If current algorithm is Reno
    if current_algo_numeric == 1:  # Reno
        # Only switch if Cubic provides significant improvement
        if loss_cubic < (loss_reno - switching_threshold):
            print(f"Improvement by switching: {loss_reno - loss_cubic:.6f}")
            return 'cubic'
        else:
            print(f"Improvement not sufficient: {loss_reno - loss_cubic:.6f} < {switching_threshold}")
            return 'reno'
        
    else:  # Cubic
        # Only switch if Reno provides significant improvement
        if loss_reno < (loss_cubic - switching_threshold):
            print(f"Improvement by switching: {loss_cubic - loss_reno:.6f}")
            return 'reno'
        else:
            print(f"Improvement not sufficient: {loss_cubic - loss_reno:.6f} < {switching_threshold}")
            return 'cubic'

# Example usage:
if __name__ == "__main__":
    # Example data (should be replaced with actual data)
    example_data = np.random.rand(15, 15)  # 15 time steps, 15 features
    
    print("=== Testing with default threshold (0.01) ===")
    # Example prediction with Reno
    print("\nTesting with Reno as current algorithm:")
    result_reno = predict_best_algorithm('reno', example_data)
    print(f"Recommended algorithm: {result_reno}")
    
    # Example prediction with Cubic
    print("\nTesting with Cubic as current algorithm:")
    result_cubic = predict_best_algorithm('cubic', example_data)
    print(f"Recommended algorithm: {result_cubic}")
    
    print("\n=== Testing with higher threshold (0.05) ===")
    # Example prediction with Reno and higher threshold
    print("\nTesting with Reno as current algorithm:")
    result_reno = predict_best_algorithm('reno', example_data, switching_threshold=0.05)
    print(f"Recommended algorithm: {result_reno}")
    
    # Example prediction with Cubic and higher threshold
    print("\nTesting with Cubic as current algorithm:")
    result_cubic = predict_best_algorithm('cubic', example_data, switching_threshold=0.05)
    print(f"Recommended algorithm: {result_cubic}") 