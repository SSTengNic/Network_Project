#!/usr/bin/env python

"""
Train Machine Learning Models for TCP Congestion Control Prediction

This script trains neural network models to predict TCP performance metrics
for different congestion control algorithms based on network conditions.
The trained models can then be used to dynamically select the optimal
algorithm for a given network state.
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


def prepare_sequence_data(data, sequence_length=15, target_col='loss_ratio', input_cols=None):
    """
    Prepare sequence data for LSTM model.
    
    Args:
        data: Pandas DataFrame containing the TCP metrics
        sequence_length: Length of input sequence for LSTM
        target_col: Column name for the target variable
        input_cols: List of column names to use as input features
        
    Returns:
        X: Input sequences
        y: Target values
    """
    if input_cols is None:
        # Use default input columns
        input_cols = ['rto', 'rtt', 'cwnd', 'segs_in', 'data_segs_out', 'lastrcv', 'delivered']
    
    # Filter to only include the columns we need
    data = data[input_cols + [target_col]].copy()
    
    # Scale the input features
    scaler = StandardScaler()
    data[input_cols] = scaler.fit_transform(data[input_cols])
    
    # Create sequences
    X = []
    y = []
    
    for i in range(len(data) - sequence_length):
        X.append(data[input_cols].iloc[i:i+sequence_length].values)
        y.append(data[target_col].iloc[i+sequence_length])
    
    return np.array(X), np.array(y), scaler


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=100, device='cpu'):
    """
    Train the LSTM model.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: LSTM model
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to use for training ('cpu' or 'cuda')
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return model, train_losses, val_losses


def plot_losses(train_losses, val_losses, save_path=None):
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot (if None, the plot is displayed)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained LSTM model
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        
    Returns:
        mse: Mean squared error
        mae: Mean absolute error
        predictions: Model predictions
        actual: Actual values
    """
    model.eval()
    predictions = []
    actual = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            actual.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    actual = np.array(actual)
    
    mse = np.mean((predictions - actual) ** 2)
    mae = np.mean(np.abs(predictions - actual))
    
    return mse, mae, predictions, actual


def plot_predictions(predictions, actual, save_path=None):
    """
    Plot model predictions against actual values.
    
    Args:
        predictions: Model predictions
        actual: Actual values
        save_path: Path to save the plot (if None, the plot is displayed)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Loss Ratio')
    plt.title('Actual vs Predicted Loss Ratio')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    """
    Main function to train and evaluate models for TCP performance prediction.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory for models and plots
    output_dir = 'models'
    plots_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Load and combine data from different algorithms
    data_dir = 'data'
    
    # Find all CSV files in the data directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No data files found. Please generate data first.")
        return
    
    all_data = []
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path, sep=';')
        
        # Extract algorithm from file name
        algo = 'reno' if 'reno' in file.lower() else 'cubic'
        df['tcp_type'] = algo
        
        all_data.append(df)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Use these columns as input features
    input_cols = ['rto', 'rtt', 'cwnd', 'segs_in', 'data_segs_out', 'lastrcv', 'delivered']
    
    # Train separate models for each algorithm
    algorithms = combined_data['tcp_type'].unique()
    
    for algo in algorithms:
        print(f"\nTraining model for {algo} algorithm...")
        
        # Filter data for this algorithm
        algo_data = combined_data[combined_data['tcp_type'] == algo].copy()
        
        # Prepare sequence data
        X, y, scaler = prepare_sequence_data(algo_data, sequence_length=15, target_col='loss_ratio', input_cols=input_cols)
        
        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Initialize model, loss function, and optimizer
        input_size = len(input_cols)
        hidden_size = 128
        num_layers = 2
        output_size = 1
        
        model = LSTM_Model(input_size, hidden_size, num_layers, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        model, train_losses, val_losses = train_model(
            train_loader, val_loader, model, criterion, optimizer, num_epochs=100, device=device
        )
        
        # Plot losses
        plot_losses(train_losses, val_losses, save_path=os.path.join(plots_dir, f'{algo}_losses.png'))
        
        # Evaluate the model
        mse, mae, predictions, actual = evaluate_model(model, test_loader, device=device)
        print(f"Test MSE: {mse:.6f}")
        print(f"Test MAE: {mae:.6f}")
        
        # Plot predictions
        plot_predictions(predictions, actual, save_path=os.path.join(plots_dir, f'{algo}_predictions.png'))
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_cols': input_cols,
            'scaler': scaler,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        }, os.path.join(output_dir, f'{algo}_model.pth'))
        
        print(f"Model for {algo} saved.")
    
    print("\nAll models trained and saved successfully!")


if __name__ == "__main__":
    main() 