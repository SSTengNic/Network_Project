# TCP Congestion Control Optimization using Deep Learning - PRD

## Project Overview
Develop a system that uses deep learning to dynamically select the optimal TCP congestion control algorithm based on real-time network conditions, improving overall network performance.

## Research Question
How can Deep Learning be leveraged to optimize TCP congestion control by dynamically selecting the most suitable congestion control algorithm based on network conditions?

## Objectives
1. Generate comprehensive datasets for different TCP congestion control algorithms (Reno, Cubic)
2. Train neural network models to predict network performance metrics and loss ratios
3. Implement a dynamic algorithm selector based on these predictions
4. Evaluate and validate the system performance

## Key Components

### 1. Data Generation via Mininet
- Simulate network environments with varying conditions (bandwidth, delay, loss, etc.)
- Collect performance data for multiple TCP congestion control algorithms (Reno, Cubic, etc.)
- Record metrics compatible with existing dataset: rto, rtt, cwnd, segs_in, data_segs_out, lastrcv, delivered
- Calculate loss ratio per time step for training prediction models

### 2. Data Processing
- Clean and normalize collected data
- Feature extraction and selection aligned with existing models
- Prepare training/validation/test datasets
- Ensure compatibility with existing datasets (reno1.log.csv and cubic1.log.csv)

### 3. Model Development
- Leverage existing work from Loss_Predict_V2.ipynb 
- Enhance/retrain models with new comprehensive datasets
- Design neural network architecture for loss prediction across different congestion control algorithms
- Implement algorithm selection mechanism based on predictions

### 4. Implementation & Testing
- Integrate prediction models with dynamic selection system
- Benchmark against static algorithm implementations
- Validate in various network scenarios

## Technical Requirements

### Mininet Simulation Environment
- Network topologies: dumbbell, parking lot, etc.
- Configurable parameters: bandwidth, delay, queue size, loss rate
- Data collection framework for TCP metrics with focus on:
  - RTT (Round-trip Time)
  - RTO (Retransmission Timeout)
  - CWND (Congestion Window)
  - SSTHRESH (Slow Start Threshold)
  - Bytes Retransmitted (bytes_retrans)
  - Data Segments Out (data_segs_out)
  - Delivered
  - Segs In (segs_in)

### ML Model Architecture
- Input: Network condition metrics
- Output: Predicted loss ratio for each algorithm
- Real-time inference capability
- Selection logic for optimal algorithm based on predictions

## Success Metrics
- Improved throughput compared to static algorithm selection
- Reduced latency and packet loss
- Faster adaptation to changing network conditions
- Better performance than either Reno or Cubic alone in diverse network environments

## Project Phases
1. **Setup Phase**: Mininet environment configuration and simulation design
2. **Data Collection Phase**: Generate datasets for different congestion control algorithms
3. **Model Training Phase**: Train and validate prediction models
4. **Integration Phase**: Implement algorithm selection system
5. **Evaluation Phase**: Comprehensive testing and performance analysis

## Additional Considerations
- Ensure model works in real-time conditions
- Consider computational overhead for practical deployment
- Document methodology for reproducibility 