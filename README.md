# TCP Congestion Control Algorithm Predictor

This tool predicts which TCP congestion control algorithm (Reno or Cubic) is likely to have a lower loss ratio based on recent network performance data.

## Prerequisites

- Python 3.6+
- PyTorch
- NumPy
- Pandas

## Installation

1. Ensure you have all the required model files in the `model_files` directory:

   - `Reno_weights.pth`
   - `my_model_weights.pth` (Cubic model)
   - `RenoCubic_weights.pth`

2. Install required Python packages:
   ```
   pip install torch numpy pandas
   ```

## Usage

Import and use the `predict_best_algorithm` function from `predict_loss.py`:

```python
from predict_loss import predict_best_algorithm

# Current TCP congestion control algorithm ('reno' or 'cubic')
current_algorithm = 'reno'

# Last 15 data points with network metrics (15 time steps, 15 features)
# Features should match those used during model training
last_15_data_points = [
    # Array of 15 data points, each with 15 features
    # ...
]

# Get the recommended algorithm with default switching threshold (0.01)
recommended_algorithm = predict_best_algorithm(current_algorithm, last_15_data_points)
print(f"Recommended algorithm: {recommended_algorithm}")

# Or with a custom switching threshold
recommended_algorithm = predict_best_algorithm(current_algorithm, last_15_data_points, switching_threshold=0.05)
print(f"Recommended algorithm: {recommended_algorithm}")
```

## Data Format

The `last_15_data_points` should be a list or numpy array with shape `[15, 15]`:

- 15 consecutive time steps
- 15 features per time step (the model expects 15 features for both Reno and Cubic models)

The function will automatically handle the different input format required by the RenoCubic model (which needs 16 features) by padding the input data.

## Model Details

The system uses three different LSTM models:

1. Reno model - Predicts loss ratio when using the Reno algorithm
2. Cubic model - Predicts loss ratio when using the Cubic algorithm
3. RenoCubic model - Predicts loss ratio when switching between algorithms

When you provide the current algorithm and recent network metrics, the function compares the predicted loss ratios to recommend the algorithm that's likely to have a lower loss ratio.

## How It Works

### Model Loading and Normalization

The system loads pre-trained models with correct normalization parameters:

1. When first run, it attempts to load the original training data to extract the actual min/max values used for normalizing the loss ratios
2. These parameters are cached for future use
3. If the original data is not available, it uses reasonable default values

### Algorithm Selection

1. The function predicts the expected loss ratio with both algorithms
2. It applies a non-negativity constraint to ensure all predictions are positive
3. To avoid unnecessary switching, it only recommends changing algorithms when the improvement exceeds the specified threshold

### Switching Threshold

The `switching_threshold` parameter (default 0.01) represents the minimum improvement in loss ratio required to justify switching algorithms. This helps prevent frequent switching when the benefit is negligible.

Increasing this value:

- Reduces the likelihood of switching
- Provides more stability in the network

Decreasing this value:

- Makes switching more likely
- May potentially optimize performance more aggressively

## Example

See the example at the bottom of `predict_loss.py` for a demonstration using random data with different switching thresholds.
