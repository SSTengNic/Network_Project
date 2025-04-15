#!/usr/bin/env python
# coding: utf-8

# This project will take in a dataset and then predict the loss of the other 

# ### Imports and CUDA

# In[22]:


# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Torch
import torch
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd


# In[23]:


# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Loss Ratio is predicted as 
# 
# Loss Ratio=bytes_sent/bytes_retrans​

# ### Merging all the reno files

# In[24]:


# folder_path = "reno"

# # Get a list of all CSV files in the folder
# csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# # Read and merge all CSV files
# df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]

# # Concatenate all DataFrames
# merged_df = pd.concat(df_list, ignore_index=True)

# # Save the merged DataFrame to a new CSV file
# merged_df.to_csv(os.path.join(folder_path, "merged_reno.csv"), index=False)

# print("Merging complete. File saved as 'merged_output.csv'.")
# print("merged_reno.csv size: ", merged_df.shape)

# #And then move the file into the root directory


# ### Preparing the data
# 
# I need to normalize the data first, and the continue from there.

# In[25]:


# Load dataset manually using NumPy
file_path = "./src/data/multi_tcp_20250411064307.csv"  # Update with actual file path
# Load the dataset using numpy (semicolon separated)
df = pd.read_csv(file_path, delimiter=";")  # skip_header=1 if there's a header row

# remove last column, some kind of spacing i
# Drop the last column using pandas

# Convert "reno" -> 1 and "cubic" -> 0 in a new numeric column
df['tcp_type'] = df['tcp_type'].map({'reno': 1, 'cubic': 0})

# df['loss_ratio'] = (df['bytes_retrans'] / df['bytes_sent'])

# Identify constant columns
constant_columns = [col for col in df.columns if len(df[col].unique()) == 1]

constant_columns.append("bytes_retrans")
constant_columns.append("bytes_sent")

print("colunms without any variation: ", constant_columns)
# Drop constant columns from the dataset
df = df.drop(columns=constant_columns)

# Normalize using Min-Max scaling: (X - min) / (max - min)
data_min = df.min(axis=0)
data_max = df.max(axis=0)
df_normalized = (df - data_min) / (data_max - data_min)

# Save normalization parameters
data_min.to_csv("data_min.csv", header=False)
data_max.to_csv("data_max.csv", header=False)
print("Saved data_min.csv and data_max.csv")

# Now df_normalized is your dataset without the constant columns and normalized
print(df_normalized.iloc[0])

print(df_normalized.iloc[50])


df_normalized.to_csv("normalized.csv", index=False)  # Set index=False to exclude row numbers
# loss_ratio_tensor = torch.tensor(df['loss_ratio'].values, dtype=torch.float32)
# Convert to PyTorch tensor
#Changed df_normalized back to df first.
print(df_normalized.shape)


# In[26]:


# df.to_csv("output.csv", index=False)  # Set index=False to exclude row numbers


# In[27]:


#As stated in the paper
seq_length = 15
forecast_steps = 15  # Predict the next 15 seconds


def create_sequences(input, labels, seq_length, forecast_steps):
    xs, ys = [], []
    for i in range(len(input) - seq_length - forecast_steps + 1):
        xs.append(input[i : i + seq_length])  # Input sequence
        ys.append(labels[i + seq_length : i + seq_length + forecast_steps])  # Next `forecast_steps` values
    return np.array(xs), np.array(ys)


X, y = create_sequences(df_normalized.values, df_normalized['loss_ratio'].values, seq_length, forecast_steps)
data_tensor = torch.tensor(df.values, dtype=torch.float32)
new_X = torch.tensor(X,dtype=torch.float32)
new_y = torch.tensor(y[:,:, None], dtype=torch.float32)
print("Data tensor:", new_X.shape)
print("Loss tensor:", new_y.shape)


# ### Splitting the data, 80/10/10

# In[35]:


train_size = int(len(new_X) *0.8)
#I remove the first batch because it is just full of zeros,
X_train, X_test = new_X[:train_size], new_X[train_size:]
y_train, y_test = new_y[:train_size], new_y[train_size:]

# Print the shapes to verify the split
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 16  # You can adjust the batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Example of accessing a batch of data
for inputs, targets in train_loader:
    print(f'Inputs: {inputs.shape}, Targets: {targets.shape}')

    break  # Only print the first batch for verification


# ###LSTM Model used as the NN.

# In[36]:


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


# ### I will implement sliding window next time, but for now, it will only predict the next value.

# In[37]:


def train(model, dataloader, num_epochs, learning_rate):
    print("Training the model")
    # Set the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Add L2 regularization
    
    # Create a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, verbose=True)
    
    model.train()  # Set the model to training mode
    loss_values = []
    best_loss = float('inf')
    patience_counter = 0
    
    # Weight different TCP types differently to emphasize their importance
    def get_tcp_weights(inputs):
        # Extract TCP type column - assuming it's at index 10 based on earlier analysis
        tcp_column = inputs[:, :, 10]
        # Create weights: 2.0 for transitions between algorithms, 1.0 for stable periods
        # This emphasizes learning at points where the algorithm changes
        weights = torch.ones_like(tcp_column)
        for i in range(tcp_column.size(0)):
            for j in range(1, tcp_column.size(1)):
                if tcp_column[i, j] != tcp_column[i, j-1]:
                    weights[i, j] = 2.0  # Put twice as much weight on algorithm transitions
        return weights.unsqueeze(-1)  # Match output dimensions

    for epoch in range(num_epochs):
        total_loss = 0  # Track total loss for averaging
        hidden_state, cell_state = None, None  # Reset hidden states for each epoch

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx == len(dataloader) - 1:  
                break  # Skip the last batch

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            output, cell_state, hidden_state = model(inputs, cell_state, hidden_state)
            
            # Get weights for different TCP types
            tcp_weights = get_tcp_weights(inputs)
            
            # Compute weighted loss to emphasize TCP algorithm transitions
            loss = criterion(output * tcp_weights, targets * tcp_weights)
            total_loss += loss.item()  # Sum up the loss for averaging

            # Backpropagation
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Detach hidden states to prevent memory buildup
            hidden_state = hidden_state.detach()
            cell_state = cell_state.detach()

        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        loss_values.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "lstm_model_state_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= 50:  # Stop after 50 epochs without improvement
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')

    # Make sure we use the best model for final save
    model.load_state_dict(torch.load("lstm_model_state_best.pth"))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_values)), loss_values, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()
    print("Loss plot saved to training_loss.png")


# In[38]:


# Define the model parameters
# Increased complexity for better learning
input_size = 12
hidden_size = 64  # Larger hidden size
num_layers = 2    # Multiple layers
output_size = 1
dataloader = train_loader

#Create the model
model = LSTM_pt(input_size, hidden_size, num_layers, output_size).to(device)
train(model, dataloader, num_epochs = 500, learning_rate = 0.001)  # Train longer
print("done training")
# Save the model state
model_save_path = "lstm_model_state.pth"
print("saving model")
torch.save(model.state_dict(), model_save_path)
print(f"Model state saved to {model_save_path}")

# Add a section to visualize feature importance
print("\n--- Analyzing Feature Importance ---")
# Extract LSTM weight matrices to analyze feature importance
lstm_weight_ih = model.lstm.weight_ih_l0.detach().cpu().numpy()  # Input-hidden weights
feature_names = df_normalized.columns.tolist()

# Calculate average magnitude of weights for each feature
feature_importance = []
for i in range(input_size):
    weights = lstm_weight_ih[:, i]
    importance = np.mean(np.abs(weights))
    feature_importance.append((feature_names[i], importance))

feature_importance.sort(key=lambda x: x[1], reverse=True)
print("\n--- Feature Importance Based on Weight Magnitude ---")
for feature, importance in feature_importance:
    print(f"{feature}: {importance:.6f}")

# Create a bar chart of feature importance
plt.figure(figsize=(12, 6))
features = [x[0] for x in feature_importance]
importances = [x[1] for x in feature_importance]
plt.bar(features, importances)
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("Feature importance plot saved to feature_importance.png")


# ### Testing the model

# In[39]:


for batch_idx, (inputs, targets) in enumerate(test_loader):
    print(f"Batch {batch_idx}: Inputs Shape: {inputs.shape}, Targets Shape: {targets.shape}")

print("Total test batches:", len(test_loader))


# In[40]:


model.eval()

# Initialize variables to track loss
total_val_loss = 0
num_batches = 0
total_lost_values = []

# Define the loss function
criterion = torch.nn.MSELoss()

# Initialize hidden state and cell state
hidden_state, cell_state = None, None  

# Disable gradient computation for validation
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):

        # Initialize hidden state and cell state for each batch
        if hidden_state is not None:
            hidden_state = hidden_state.detach()
        if cell_state is not None:
            cell_state = cell_state.detach()

        # Forward pass
        output, cell_state, hidden_state = model(inputs, cell_state, hidden_state)

        output_denorm = output *(data_max["loss_ratio"] - data_min["loss_ratio"]) + data_min["loss_ratio"]
        target_denorm = targets *(data_max["loss_ratio"] - data_min["loss_ratio"]) + data_min["loss_ratio"]

        if batch_idx % 5 == 0:
            print("Predicted output: ", output_denorm[0])
            print("True Output hello: ", target_denorm[0])

        # Compute loss for this batch
        loss = criterion(output, targets)

        # Accumulate loss
        total_val_loss += loss.item()  # Add the batch loss to total
        total_lost_values.append(total_val_loss / (batch_idx + 1))  # Average loss so far

        num_batches += 1

# Compute average loss for the entire validation set
avg_loss = total_val_loss / num_batches

# Print validation results
print(f'Average Validation Loss: {avg_loss:.10f}')

# Plot the loss curve
plt.plot(range(num_batches), total_lost_values, label='Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss Over Batches (Validation)')
plt.legend()
plt.show()


# In[34]:


print("output size", output.shape)
print("targets size", targets.shape)


# In[ ]:


# # Example raw input: [seq_len, num_features] (unnormalized)
# #To be used for algo
# raw_input = np.array([
#     [100, 0.5, 0.1],  # time step 1
#     [110, 0.6, 0.2],  # time step 2
#     [120, 0.7, 0.3],  # time step 3
# ], dtype=np.float32)

# # Normalize using your original Min-Max scalers
# normalized_input = (raw_input - data_min.values) / (data_max.values - data_min.values)
# normalized_input = torch.tensor(normalized_input, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, num_features]

# model.eval()
# with torch.no_grad():
#     hidden_state, cell_state = None, None
#     output, _, _ = model(normalized_input, cell_state, hidden_state)

#     # Denormalize the output
#     min_val = data_min["Taxi Available in Selected Box Area"]
#     max_val = data_max["Taxi Available in Selected Box Area"]
#     output_denorm = output * (max_val - min_val) + min_val

#     print("Normalized model output:", output)
#     print("Denormalized prediction:", output_denorm)

