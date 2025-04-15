import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model Definition (Must match the training script) ---
class LSTM_pt(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_pt, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # LSTM cell with increased complexity
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.2)
        
        # Add an attention mechanism to focus on important features (like tcp_type)
        self.attention = torch.nn.Linear(hidden_size, 1)
        
        # Add deeper network for better feature interactions
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        
        # Final prediction layer
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs, cell_state=None, hidden_state=None):
        # Forward pass through the LSTM cell
        if hidden_state is None or cell_state is None:
            # For prediction, batch size is 1
            hidden_state = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)
            cell_state = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)
        
        hidden = (cell_state, hidden_state)
        output, new_memory = self.lstm(inputs, hidden)
        cell_state, hidden_state = new_memory
        
        # Apply attention to focus on important timesteps
        attention_weights = torch.softmax(self.attention(output), dim=1)
        
        # Add non-linear transformations for better feature interaction
        enhanced = self.fc1(output)
        enhanced = self.relu(enhanced)
        enhanced = self.dropout(enhanced)
        
        # Final prediction
        output = self.linear(enhanced)
        return output, cell_state, hidden_state


# --- Load Normalization Parameters and Model ---
try:
    # Load normalization parameters (assuming they are saved in the same directory)
    # Read them as Series, using the first column as index and second as values
    data_min = pd.read_csv("data_min.csv", header=None, index_col=0).squeeze("columns")
    data_max = pd.read_csv("data_max.csv", header=None, index_col=0).squeeze("columns")
    print("Loaded normalization parameters.")

    # Get the index/position of the 'tcp_type' feature
    feature_names = data_min.index.tolist()
    try:
        tcp_type_index = feature_names.index('tcp_type')
        print(f"'tcp_type' found at index: {tcp_type_index}")
    except ValueError:
        print("Error: 'tcp_type' column not found in the loaded normalization parameters.")
        exit() # Or handle error appropriately

    # Define model parameters (must match training)
    input_size = 12 # Should match the number of features in data_min/data_max
    if len(feature_names) != input_size:
         print(f"Warning: Expected input_size {input_size} but found {len(feature_names)} features in normalization files.")
    hidden_size = 64 # Increased from 20 to 64
    num_layers = 2  # Increased from 1 to 2
    output_size = 1
    model_save_path = "lstm_model_state.pth"

    # Instantiate the model
    model = LSTM_pt(input_size, hidden_size, num_layers, output_size)

    # Load the trained state dictionary
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device) # Move model to the chosen device
    model.eval() # Set the model to evaluation mode
    print(f"Loaded model state from {model_save_path}")

    # Analyze model weights to understand importance of tcp_type
    # Extract LSTM weight matrices
    lstm_weight_ih = model.lstm.weight_ih_l0.detach().cpu().numpy()  # Input-hidden weights
    
    # The tcp_type feature influence can be seen in the rows of the weight matrix 
    # corresponding to the tcp_type_index column
    tcp_type_weights = lstm_weight_ih[:, tcp_type_index]
    
    print("\n--- Analysis of Model Weights ---")
    print(f"tcp_type feature weights (first 10): {tcp_type_weights[:10]}")
    
    # Calculate average magnitude of weights for each feature to compare importance
    feature_importance = []
    for i in range(input_size):
        weights = lstm_weight_ih[:, i]
        importance = np.mean(np.abs(weights))
        feature_importance.append((feature_names[i], importance))
    
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    print("\n--- Feature Importance Based on Weight Magnitude ---")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance:.6f}")

except FileNotFoundError:
    print("Error: Could not find 'data_min.csv', 'data_max.csv', or 'lstm_model_state.pth'.")
    print("Please run the training script first to generate these files.")
    exit() # Or handle error appropriately


# --- Prediction Function ---
def predict_best_tcp(last_15_steps_data):
    """
    Predicts the best TCP algorithm (Reno or Cubic) based on the last 15 time steps of data.

    Args:
        last_15_steps_data (np.ndarray or pd.DataFrame): A NumPy array or Pandas DataFrame
            with shape (15, 12) containing the unnormalized network statistics for the
            last 15 time steps. The columns must be in the same order as used during training
            (i.e., matching the order in data_min.csv/data_max.csv).

    Returns:
        str: 'reno' or 'cubic', indicating the predicted best algorithm.
             Returns None if input shape is incorrect.
    """
    if last_15_steps_data.shape != (15, input_size):
        print(f"Error: Input data must have shape (15, {input_size}), but got {last_15_steps_data.shape}")
        return None

    # --- Preprocessing ---
    # Convert to DataFrame if it's a NumPy array to ensure alignment with scaler index
    if isinstance(last_15_steps_data, np.ndarray):
        input_df = pd.DataFrame(last_15_steps_data, columns=feature_names)
    else:
        input_df = last_15_steps_data

    # Print original tcp_type values for verification
    print(f"Original tcp_type values: {input_df['tcp_type'].values}")
    
    # Normalize the data
    normalized_df = (input_df - data_min) / (data_max - data_min)
    # Handle potential division by zero if max == min for a feature (results in NaN)
    normalized_df = normalized_df.fillna(0) 
    
    # Print normalized tcp_type values for verification
    print(f"Normalized tcp_type values: {normalized_df['tcp_type'].values}")
    
    # Convert to tensor and add batch dimension: [1, 15, 12]
    input_tensor = torch.tensor(normalized_df.values, dtype=torch.float32).unsqueeze(0).to(device)

    # Print tcp_type_index for verification
    print(f"tcp_type_index: {tcp_type_index}")
    print(f"Feature names: {feature_names}")

    # Compare predictions using detailed tracking
    results = {}
    all_outputs = []  # To collect outputs for comparison
    
    # --- Predict for Reno (tcp_type = 1) ---
    with torch.no_grad():
        input_reno = input_tensor.clone()
        
        # Set tcp_type feature to 1 (normalized value for Reno)
        input_reno[:, :, tcp_type_index] = 1.0
        
        # Print the modified tensor for Reno
        print(f"Reno tcp_type values in tensor: {input_reno[0, :, tcp_type_index]}")
        
        output_reno, cell_state_reno, hidden_state_reno = model(input_reno)
        
        # Print raw output values for Reno
        print(f"Raw output for Reno: {output_reno.squeeze()}")
        all_outputs.append(("Reno", output_reno.cpu().squeeze().numpy()))
        
        # Calculate average predicted loss over the forecast horizon
        avg_loss_reno = output_reno.mean().item()
        results['reno'] = avg_loss_reno

    # --- Predict for Cubic (tcp_type = 0) ---
    with torch.no_grad():
        input_cubic = input_tensor.clone()
        
        # Set tcp_type feature to 0 (normalized value for Cubic)
        input_cubic[:, :, tcp_type_index] = 0.0
        
        # Print the modified tensor for Cubic
        print(f"Cubic tcp_type values in tensor: {input_cubic[0, :, tcp_type_index]}")

        output_cubic, cell_state_cubic, hidden_state_cubic = model(input_cubic)
        
        # Print raw output values for Cubic
        print(f"Raw output for Cubic: {output_cubic.squeeze()}")
        all_outputs.append(("Cubic", output_cubic.cpu().squeeze().numpy()))
        
        # Calculate average predicted loss over the forecast horizon
        avg_loss_cubic = output_cubic.mean().item()
        results['cubic'] = avg_loss_cubic

    # Compare hidden states to see if the model is responding differently
    print("\n--- LSTM State Comparison ---")
    # Handle None states that might be returned from simplified demo model
    if hidden_state_reno is not None and hidden_state_cubic is not None:
        hidden_diff = torch.abs(hidden_state_reno - hidden_state_cubic).mean().item()
        print(f"Average difference in hidden states: {hidden_diff:.8f}")
    else:
        print("Hidden states unavailable for comparison")
        
    if cell_state_reno is not None and cell_state_cubic is not None:
        cell_diff = torch.abs(cell_state_reno - cell_state_cubic).mean().item()
        print(f"Average difference in cell states: {cell_diff:.8f}")
    else:
        print("Cell states unavailable for comparison")

    # Plot the outputs for visual comparison
    plt.figure(figsize=(10, 6))
    for name, output in all_outputs:
        plt.plot(output, label=name)
    plt.title('Predicted Loss for Different TCP Algorithms')
    plt.xlabel('Time Step')
    plt.ylabel('Predicted Loss (Normalized)')
    plt.legend()
    plt.savefig('tcp_prediction_comparison.png')
    print("\nSaved comparison plot to 'tcp_prediction_comparison.png'")

    print(f"Predicted Avg Loss (Normalized) -> Reno: {results['reno']:.6f}, Cubic: {results['cubic']:.6f}")
    print(f"Absolute difference in predictions: {abs(results['reno'] - results['cubic']):.8f}")

    # --- Decision ---
    if results['reno'] < results['cubic']:
        return 'reno'
    else:
        return 'cubic'


# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- Running Example Prediction ---")
    # Create some dummy input data (15 steps, 12 features)
    # IMPORTANT: Replace this with your actual recent data!
    # The columns must match the order in data_min/data_max files.
    
    # Check if feature_names is loaded before creating dummy data
    if 'feature_names' in locals():
      # Create more diverse test data to see if model can distinguish between TCP types
      dummy_data = np.random.rand(15, input_size) 
      
      # Set the tcp_type column to alternating values to test sensitivity
      dummy_data[:, tcp_type_index] = np.random.randint(0, 2, size=15)
      
      # Create a DataFrame to ensure column order for the dummy data
      dummy_df = pd.DataFrame(dummy_data, columns=feature_names)
      
      print(f"Input data shape: {dummy_df.shape}")
      print(f"Input data columns: {dummy_df.columns.tolist()}")
      
      predicted_algo = predict_best_tcp(dummy_df)

      if predicted_algo:
          print(f"\nPredicted best algorithm for the next period: {predicted_algo.upper()}")
    else:
       print("Could not run example because feature names were not loaded (likely due to missing files).") 