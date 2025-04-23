import matplotlib.pyplot as plt
import ast
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

def read_tcp_log(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    timestamps = []
    cwnd_values = []
    
    for line in lines:
        try:
            data = ast.literal_eval(line)
            timestamps.append(data['timestamp'])
            cwnd_values.append(data['cwnd'])
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing line in {filename}: {e}")
            continue
    
    # Convert to numpy arrays for easier manipulation
    timestamps = np.array(timestamps)
    cwnd_values = np.array(cwnd_values)
    
    # Normalize timestamps to start from 0
    if len(timestamps) > 0:
        timestamps = timestamps - timestamps[0]
    
    return timestamps, cwnd_values

def read_prototype_log(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    timestamps = []
    cwnd_values = []
    tcp_types = []
    
    for line in lines:
        try:
            data = ast.literal_eval(line)
            timestamps.append(data['timestamp'])
            cwnd_values.append(data['cwnd'])
            tcp_types.append(data['tcp_type'])
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing line in {filename}: {e}")
            continue
    
    # Convert to numpy arrays for easier manipulation
    timestamps = np.array(timestamps)
    cwnd_values = np.array(cwnd_values)
    
    # Normalize timestamps to start from 0
    if len(timestamps) > 0:
        timestamps = timestamps - timestamps[0]
    
    return timestamps, cwnd_values, tcp_types

def calculate_avg_cwnd(cwnd_values, skip_first=0, cutoff=None):
    """Calculate average CWND value from an array of values, optionally skipping first N values and cutting off at a time."""
    if skip_first >= len(cwnd_values):
        return 0  # Return 0 if not enough values
    
    if cutoff is None:
        return np.mean(cwnd_values[skip_first:])
    else:
        return np.mean(cwnd_values[skip_first:cutoff])

def plot_cwnd_comparison():
    # Number of initial datapoints to skip
    skip_first = 10
    
    # Time cutoff in seconds
    time_cutoff = 100
    
    # Read data from the standard log files
    cubic_time, cubic_cwnd = read_tcp_log('src/data/cubic_log.txt')
    reno_time, reno_cwnd = read_tcp_log('src/data/reno_log.txt')
    
    # Read prototype data with tcp_type info
    prototype_time, prototype_cwnd, prototype_tcp_types = read_prototype_log('src/data/prototype_log.txt')
    
    # Find the cutoff index for each dataset (where time exceeds time_cutoff)
    cubic_cutoff = np.searchsorted(cubic_time, time_cutoff, side='right') if len(cubic_time) > 0 else 0
    reno_cutoff = np.searchsorted(reno_time, time_cutoff, side='right') if len(reno_time) > 0 else 0
    prototype_cutoff = np.searchsorted(prototype_time, time_cutoff, side='right') if len(prototype_time) > 0 else 0
    
    # Calculate average CWND values with all datapoints within time cutoff
    cubic_avg_all = calculate_avg_cwnd(cubic_cwnd, 0, cubic_cutoff)
    reno_avg_all = calculate_avg_cwnd(reno_cwnd, 0, reno_cutoff)
    prototype_avg_all = calculate_avg_cwnd(prototype_cwnd, 0, prototype_cutoff)
    
    # Calculate average CWND values without first 10 datapoints and within time cutoff
    cubic_avg = calculate_avg_cwnd(cubic_cwnd, skip_first, cubic_cutoff)
    reno_avg = calculate_avg_cwnd(reno_cwnd, skip_first, reno_cutoff)
    prototype_avg = calculate_avg_cwnd(prototype_cwnd, skip_first, prototype_cutoff)
    
    # Print average CWND values
    print(f"Average CWND for Cubic (all data within {time_cutoff}s): {cubic_avg_all:.2f}")
    print(f"Average CWND for Reno (all data within {time_cutoff}s): {reno_avg_all:.2f}")
    print(f"Average CWND for Prototype (all data within {time_cutoff}s): {prototype_avg_all:.2f}")
    print(f"\nExcluding first {skip_first} datapoints and within {time_cutoff}s:")
    print(f"Average CWND for Cubic: {cubic_avg:.2f}")
    print(f"Average CWND for Reno: {reno_avg:.2f}")
    print(f"Average CWND for Prototype: {prototype_avg:.2f}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot standard cubic and reno data
    if cubic_cutoff > 0:
        # First 10 points in lighter color
        if skip_first > 0:
            plt.plot(cubic_time[:min(skip_first, cubic_cutoff)], cubic_cwnd[:min(skip_first, cubic_cutoff)], 
                    color='lightblue', linewidth=1.5, linestyle='-')
        
        # Regular color after skip_first
        if cubic_cutoff > skip_first:
            plt.plot(cubic_time[skip_first:cubic_cutoff], cubic_cwnd[skip_first:cubic_cutoff], 
                    label=f'Cubic (avg: {cubic_avg:.2f})', color='blue', linewidth=2)
    
    if reno_cutoff > 0:
        # First 10 points in lighter color
        if skip_first > 0:
            plt.plot(reno_time[:min(skip_first, reno_cutoff)], reno_cwnd[:min(skip_first, reno_cutoff)], 
                    color='lightcoral', linewidth=1.5, linestyle='-')
        
        # Regular color after skip_first
        if reno_cutoff > skip_first:
            plt.plot(reno_time[skip_first:reno_cutoff], reno_cwnd[skip_first:reno_cutoff], 
                    label=f'Reno (avg: {reno_avg:.2f})', color='red', linewidth=2)
    
    # Plot prototype data with changing colors based on tcp_type
    if prototype_cutoff > 0:
        # First handle the first 10 points in lighter colors if needed
        if skip_first > 0:
            plt.plot(prototype_time[:min(skip_first, prototype_cutoff)], 
                    prototype_cwnd[:min(skip_first, prototype_cutoff)], 
                    color='lightgray', linewidth=1.5, linestyle='-')
        
        # Now handle the main data with color changes
        if prototype_cutoff > skip_first:
            # Get the data to plot (after skip_first)
            plot_times = prototype_time[skip_first:prototype_cutoff]
            plot_cwnds = prototype_cwnd[skip_first:prototype_cutoff]
            plot_types = prototype_tcp_types[skip_first:prototype_cutoff]
            
            if len(plot_times) > 1:
                # Create line segments
                points = np.array([plot_times, plot_cwnds]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create a color array based on tcp_type
                colors = []
                for i in range(len(plot_types)-1):
                    # Green for cubic, orange for reno
                    colors.append(0 if plot_types[i] == 'cubic' else 1)
                
                # Create a colormap with just two colors
                cmap = ListedColormap(['green', 'orange'])
                
                # Create the line collection
                lc = LineCollection(segments, cmap=cmap)
                lc.set_array(np.array(colors))
                lc.set_linewidth(2)
                
                # Add to plot
                line = plt.gca().add_collection(lc)
                
                # Add custom legend entries
                plt.plot([], [], color='green', linewidth=2, label='Prototype (Cubic)')
                plt.plot([], [], color='orange', linewidth=2, label='Prototype (Reno)')
                plt.plot([], [], ' ', label=f'Prototype (avg: {prototype_avg:.2f})')
    
    # Add vertical line at the cutoff point
    if skip_first > 0 and len(cubic_time) > skip_first:
        plt.axvline(x=cubic_time[skip_first], color='black', linestyle='--', alpha=0.5, 
                  label=f'First {skip_first} points excluded')
    
    # Add horizontal lines for average values
    plt.axhline(y=cubic_avg, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=reno_avg, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=prototype_avg, color='green', linestyle='--', alpha=0.5)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Congestion Window (cwnd)', fontsize=12)
    plt.title(f'Comparison of TCP Congestion Window Sizes (Skip {skip_first}, Cutoff {time_cutoff}s)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Set x-axis limit to 100 seconds
    plt.xlim(0, time_cutoff)
    
    # Save the plot to a file
    plt.savefig(f'cwnd_comparison_skip{skip_first}_cutoff{time_cutoff}.png', dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    plot_cwnd_comparison() 