#!/usr/bin/env python3

"""
Simplified TCP Performance Predictor

This script uses a trained neural network model to predict TCP performance
(loss ratio) based on current network conditions.
"""

import os
import sys
import subprocess
import re
import numpy as np
import pandas as pd
import torch
from simplified_train_model import SimpleNN


class TCPPerformancePredictor:
    """
    Predicts TCP performance using a trained neural network model.
    """
    
    def __init__(self, model_path):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load the trained model from a file.
        
        Args:
            model_path: Path to the trained model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model parameters
        self.input_cols = checkpoint['input_cols']
        self.scaler = checkpoint['scaler']
        
        # Create model
        self.model = SimpleNN(len(self.input_cols)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Input features: {', '.join(self.input_cols)}")
    
    def collect_network_metrics(self):
        """
        Collect current network metrics using ss command.
        
        Returns:
            A dictionary containing the collected metrics
        """
        # Use ss command to get TCP metrics for all established connections
        cmd = "ss -i --tcp -o state established"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            output = result.stdout
            
            if not output.strip():
                print("No established TCP connections found.")
                return None
            
            # Parse the output and extract metrics
            metrics = {}
            
            # Extract RTT
            rtt_match = re.search(r'rtt:([0-9.]+)/', output)
            if rtt_match:
                metrics['rtt'] = float(rtt_match.group(1))
            else:
                metrics['rtt'] = 0
            
            # Extract RTO
            rto_match = re.search(r'rto:([0-9]+)', output)
            if rto_match:
                metrics['rto'] = int(rto_match.group(1))
            else:
                metrics['rto'] = 0
            
            # Extract CWND
            cwnd_match = re.search(r'cwnd:([0-9]+)', output)
            if cwnd_match:
                metrics['cwnd'] = int(cwnd_match.group(1))
            else:
                metrics['cwnd'] = 0
            
            # Extract segments
            segs_in_match = re.search(r'segs_in:([0-9]+)', output)
            if segs_in_match:
                metrics['segs_in'] = int(segs_in_match.group(1))
            else:
                metrics['segs_in'] = 0
            
            data_segs_out_match = re.search(r'data_segs_out:([0-9]+)', output)
            if data_segs_out_match:
                metrics['data_segs_out'] = int(data_segs_out_match.group(1))
            else:
                metrics['data_segs_out'] = 0
            
            delivered_match = re.search(r'delivered:([0-9]+)', output)
            if delivered_match:
                metrics['delivered'] = int(delivered_match.group(1))
            else:
                metrics['delivered'] = 0
            
            lastrcv_match = re.search(r'lastrcv:([0-9]+)', output)
            if lastrcv_match:
                metrics['lastrcv'] = int(lastrcv_match.group(1))
            else:
                metrics['lastrcv'] = 0
            
            return metrics
        
        except subprocess.CalledProcessError as e:
            print(f"Error running ss command: {e}")
            return None
    
    def predict_loss_ratio(self, metrics=None):
        """
        Predict loss ratio based on network metrics.
        
        Args:
            metrics: Dictionary of network metrics (if None, metrics will be collected)
            
        Returns:
            Predicted loss ratio
        """
        if metrics is None:
            metrics = self.collect_network_metrics()
        
        if metrics is None:
            print("No metrics available for prediction.")
            return None
        
        # Check if we have all required metrics
        missing_cols = [col for col in self.input_cols if col not in metrics]
        if missing_cols:
            print(f"Missing required metrics: {missing_cols}")
            # Fill missing metrics with zeros
            for col in missing_cols:
                metrics[col] = 0
        
        # Prepare input data
        input_data = np.array([[metrics[col] for col in self.input_cols]])
        
        # Scale input data
        scaled_input = self.scaler.transform(input_data)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(scaled_input).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy()[0, 0]
        
        return prediction


def main():
    """Main function to run the TCP performance predictor."""
    # Check command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Use default model path
        model_path = 'models/cubic_model.pt'
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train a model first or specify a valid model path.")
        return
    
    try:
        # Create predictor
        predictor = TCPPerformancePredictor(model_path)
        
        # Predict loss ratio
        loss_ratio = predictor.predict_loss_ratio()
        
        if loss_ratio is not None:
            print(f"\nPredicted loss ratio: {loss_ratio:.6f}")
            
            # Interpret the result
            if loss_ratio < 0.01:
                print("Network performance is excellent (low loss ratio).")
            elif loss_ratio < 0.05:
                print("Network performance is good (moderate loss ratio).")
            else:
                print("Network performance is poor (high loss ratio).")
        
    except Exception as e:
        print(f"Error using predictor: {e}")


if __name__ == "__main__":
    main() 