#!/usr/bin/env python3

"""
Simplified Training Script for TCP Congestion Control Prediction

This script trains a simple neural network model to predict loss ratios
based on network metrics collected from TCP connections.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SimpleNN(nn.Module):
    """Simple neural network for TCP performance prediction."""
    
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)


def prepare_data(data_path, input_cols=None):
    """
    Prepare data for training.
    
    Args:
        data_path: Path to the CSV file containing TCP metrics
        input_cols: List of column names to use as input features
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    if input_cols is None:
        # Use default input columns
        input_cols = ['rto', 'rtt', 'cwnd', 'segs_in', 'data_segs_out', 'lastrcv', 'delivered']
    
    # Read the data
    print(f"Reading data from {data_path}...")
    df = pd.read_csv(data_path, sep=';')
    print(f"Data shape: {df.shape}")
    
    # Check if we have the required columns
    missing_cols = [col for col in input_cols + ['loss_ratio'] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the data: {missing_cols}")
    
    # Filter to only include rows with valid data
    df = df.dropna(subset=input_cols + ['loss_ratio'])
    
    # Check if we have enough data
    if len(df) < 10:
        raise ValueError(f"Not enough data points after filtering: {len(df)}")
    
    print(f"Using {len(df)} data points for training")
    
    # Split features and target
    X = df[input_cols].values
    y = df['loss_ratio'].values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, input_cols


def train_model(X_train, y_train, X_test, y_test, input_size, num_epochs=100, batch_size=32):
    """
    Train the neural network model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        input_size: Number of input features
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model, training losses, test losses
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    train_losses = []
    test_losses = []
    
    print(f"Training model for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)
            outputs = model(X_test_tensor)
            test_loss = criterion(outputs, y_test_tensor).item()
            test_losses.append(test_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    return model, train_losses, test_losses


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        
    Returns:
        MSE, predictions, actual values
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    # Calculate MSE
    mse = np.mean((predictions - y_test) ** 2)
    
    return mse, predictions, y_test


def plot_results(train_losses, test_losses, predictions, actual, output_dir):
    """
    Plot training curves and predictions.
    
    Args:
        train_losses: List of training losses
        test_losses: List of test losses
        predictions: Model predictions
        actual: Actual values
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.scatter(actual, predictions, alpha=0.5)
    plt.plot([0, max(actual)], [0, max(actual)], 'r--')  # Diagonal line for perfect predictions
    plt.xlabel('Actual Loss Ratio')
    plt.ylabel('Predicted Loss Ratio')
    plt.title('Actual vs Predicted Loss Ratio')
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    
    # Plot sorted predictions vs actual
    plt.figure(figsize=(12, 6))
    indices = np.argsort(actual)
    plt.plot(actual[indices], label='Actual', marker='o', markersize=3)
    plt.plot(predictions[indices], label='Predicted', marker='x', markersize=3)
    plt.xlabel('Sample (sorted by actual value)')
    plt.ylabel('Loss Ratio')
    plt.title('Sorted Actual vs Predicted Loss Ratio')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'sorted_predictions.png'))
    
    plt.close('all')


def main():
    """Main function to train and evaluate the model."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Directories
    data_dir = 'data'
    output_dir = 'models'
    plots_dir = 'plots'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Find the most recent cubic data file
    cubic_files = [f for f in os.listdir(data_dir) if f.startswith('cubic_') and f.endswith('.csv')]
    
    if not cubic_files:
        print(f"No cubic data files found in {data_dir}. Please run data collection first.")
        return
    
    # Sort files by modification time (most recent first)
    cubic_files.sort(key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    data_file = os.path.join(data_dir, cubic_files[0])
    
    print(f"Using most recent data file: {data_file}")
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler, input_cols = prepare_data(data_file)
        
        # Train model
        model, train_losses, test_losses = train_model(
            X_train, y_train, X_test, y_test, 
            input_size=len(input_cols), 
            num_epochs=100,
            batch_size=32
        )
        
        # Evaluate model
        mse, predictions, actual = evaluate_model(model, X_test, y_test)
        print(f"\nTest MSE: {mse:.6f}")
        
        # Plot results
        plot_results(train_losses, test_losses, predictions, actual, plots_dir)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_cols': input_cols,
            'scaler': scaler
        }, os.path.join(output_dir, 'cubic_model.pt'))
        
        print(f"Model trained and saved successfully to {os.path.join(output_dir, 'cubic_model.pt')}")
        print(f"Plots saved to {plots_dir}")
    
    except Exception as e:
        print(f"Error training model: {e}")


if __name__ == "__main__":
    main() 