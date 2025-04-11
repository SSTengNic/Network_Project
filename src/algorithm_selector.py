#!/usr/bin/env python

"""
TCP Congestion Control Algorithm Selector

This script uses trained models to dynamically select the optimal TCP
congestion control algorithm based on current network conditions.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import subprocess
import time
from sklearn.preprocessing import StandardScaler


class LSTM_Model(nn.Module):
    """LSTM model for TCP performance prediction."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.fc(out)
        return out


class AlgorithmSelector:
    """
    Selects the optimal TCP congestion control algorithm based on
    current network conditions using trained models.
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the algorithm selector.
        
        Args:
            models_dir: Directory containing the trained models
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.input_cols = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the models
        self.load_models()
    
    def load_models(self):
        """Load the trained models for each algorithm."""
        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory '{self.models_dir}' not found.")
        
        # Get list of model files
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.pth')]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in '{self.models_dir}'.")
        
        for model_file in model_files:
            # Extract algorithm name from file name
            algo = model_file.split('_')[0]
            
            # Load the model checkpoint
            checkpoint = torch.load(
                os.path.join(self.models_dir, model_file),
                map_location=self.device
            )
            
            # Extract model parameters
            input_cols = checkpoint['input_cols']
            hidden_size = checkpoint['hidden_size']
            num_layers = checkpoint['num_layers']
            scaler = checkpoint['scaler']
            
            # Save the input columns (should be the same for all models)
            if self.input_cols is None:
                self.input_cols = input_cols
            
            # Create and load the model
            model = LSTM_Model(
                input_size=len(input_cols),
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Save the model and scaler
            self.models[algo] = model
            self.scalers[algo] = scaler
            
            print(f"Loaded model for {algo} algorithm.")
    
    def collect_network_metrics(self):
        """
        Collect current network metrics using ss command.
        
        Returns:
            A DataFrame containing the collected metrics
        """
        # This is a simplified version that would need to be expanded
        # in a real implementation to collect actual network metrics
        
        # Use ss command to get TCP metrics
        cmd = "ss -i --tcp -o state established"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Parse the output and extract metrics
        # For simplicity, we're returning a random set of metrics here
        # In a real implementation, this would parse the actual ss output
        
        # Example data
        data = {
            'rto': np.random.randint(300, 500, 15),
            'rtt': np.random.uniform(100, 200, 15),
            'cwnd': np.random.randint(20, 40, 15),
            'segs_in': np.random.randint(500, 1000, 15),
            'data_segs_out': np.random.randint(500, 1000, 15),
            'lastrcv': np.random.randint(1000, 2000, 15),
            'delivered': np.random.randint(500, 1000, 15)
        }
        
        return pd.DataFrame(data)
    
    def preprocess_metrics(self, metrics_df, algo):
        """
        Preprocess network metrics for model input.
        
        Args:
            metrics_df: DataFrame containing network metrics
            algo: Algorithm name for selecting the appropriate scaler
            
        Returns:
            Preprocessed metrics as a tensor ready for model input
        """
        # Extract the required columns
        metrics = metrics_df[self.input_cols].copy()
        
        # Scale the metrics using the scaler for this algorithm
        scaled_metrics = self.scalers[algo].transform(metrics)
        
        # Convert to tensor
        metrics_tensor = torch.FloatTensor(scaled_metrics).unsqueeze(0).to(self.device)
        
        return metrics_tensor
    
    def predict_loss_ratio(self, metrics_df):
        """
        Predict loss ratio for each algorithm.
        
        Args:
            metrics_df: DataFrame containing network metrics
            
        Returns:
            A dictionary mapping algorithm names to predicted loss ratios
        """
        predictions = {}
        
        for algo, model in self.models.items():
            # Preprocess the metrics
            metrics_tensor = self.preprocess_metrics(metrics_df, algo)
            
            # Make prediction
            with torch.no_grad():
                pred = model(metrics_tensor)
                predictions[algo] = pred.item()
        
        return predictions
    
    def select_algorithm(self):
        """
        Select the optimal algorithm based on predicted loss ratios.
        
        Returns:
            The name of the selected algorithm
        """
        # Collect current network metrics
        metrics_df = self.collect_network_metrics()
        
        # Predict loss ratio for each algorithm
        predictions = self.predict_loss_ratio(metrics_df)
        
        # Select the algorithm with the lowest predicted loss ratio
        selected_algo = min(predictions, key=predictions.get)
        
        print(f"Predicted loss ratios: {predictions}")
        print(f"Selected algorithm: {selected_algo}")
        
        return selected_algo
    
    def apply_algorithm(self, algorithm):
        """
        Apply the selected algorithm to the system.
        
        Args:
            algorithm: The name of the algorithm to apply
        """
        # Set the TCP congestion control algorithm
        cmd = f"sysctl -w net.ipv4.tcp_congestion_control={algorithm}"
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Successfully set TCP congestion control algorithm to {algorithm}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to set TCP congestion control algorithm: {e}")


def main():
    """Main function to run the algorithm selector."""
    parser = argparse.ArgumentParser(description='TCP Congestion Control Algorithm Selector')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing the trained models')
    parser.add_argument('--interval', type=int, default=10,
                        help='Time interval (in seconds) between algorithm selections')
    parser.add_argument('--run-once', action='store_true',
                        help='Run the selector once and exit')
    args = parser.parse_args()
    
    try:
        # Initialize the algorithm selector
        selector = AlgorithmSelector(models_dir=args.models_dir)
        
        if args.run_once:
            # Run the selector once
            algorithm = selector.select_algorithm()
            selector.apply_algorithm(algorithm)
        else:
            # Run the selector continuously
            print(f"Running algorithm selector every {args.interval} seconds...")
            print("Press Ctrl+C to stop.")
            
            while True:
                algorithm = selector.select_algorithm()
                selector.apply_algorithm(algorithm)
                time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\nAlgorithm selector stopped.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()