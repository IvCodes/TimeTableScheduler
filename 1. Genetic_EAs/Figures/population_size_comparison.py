"""
Population Size Comparison Visualization Script

This script creates visualizations to compare the Pareto fronts between
different population sizes and generation counts (50/50 vs 100/100)
to analyze the trade-off between computational cost and solution quality.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.patches import Patch

# Constants for consistent labeling
CONFIG_NAMES = ["50 Pop/Gen", "100 Pop/Gen"]
# Algorithm name constants
ALG_NSGA2 = "NSGA-II"
ALG_SPEA2 = "SPEA2"
ALG_MOEAD = "MOEA/D"
ALGORITHM_NAMES = [ALG_NSGA2, ALG_SPEA2, ALG_MOEAD]
COLORS = ['blue', 'green', 'red']
CONFIG_MARKERS = ['o', 'x']  # Circle for 50/50, X for 100/100
CONFIG_ALPHA = [0.7, 0.9]    # Lower alpha for 50/50, higher for 100/100
LINE_STYLES = ['-', '--', '-.']  # Different styles for algorithms

# Handle key name differences between JSON files
KEY_MAPPINGS = {
    "50": {
        "unassigned": "unassigned",
        "soft_score": "soft_score"
    },
    "100": {
        "unassigned": "unassigned",
        "soft_score": "soft_score"
    }
}

# Execution time data (extracted from the console output)
EXECUTION_TIMES = {
    "50": {
        ALG_NSGA2: 17.05,
        ALG_SPEA2: 14.45,
        ALG_MOEAD: 25.36
    },
    "100": {
        ALG_NSGA2: 74.30,
        ALG_SPEA2: 82.15,
        ALG_MOEAD: 138.58
    }
}

# Pareto front sizes (extracted from the console output)
PARETO_SIZES = {
    "50": {
        ALG_NSGA2: 50,
        ALG_SPEA2: 50,
        ALG_MOEAD: 93
    },
    "100": {
        ALG_NSGA2: 100,
        ALG_SPEA2: 100,
        ALG_MOEAD: 171
    }
}

def load_json_data(json_path):
    """Load JSON data from the specified path."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {os.path.basename(json_path)}")
        return data
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
        return None

def create_population_size_comparison(data_paths, output_dir):
    """
    Create visualizations comparing results from different population sizes.
    
    Args:
        data_paths: Dictionary with config keys and paths to data files
        output_dir: Directory to save output figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data for each configuration
    datasets = {}
    for config_key, path in data_paths.items():
        data = load_json_data(path)
        if data:
            datasets[config_key] = data
    
    if len(datasets) < 2:
        print("Not enough data to compare. Need both 50/50 and 100/100 configurations.")
        return
    
    # Create comparison figures for each algorithm
    for alg_idx, alg_name in enumerate(ALGORITHM_NAMES):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot data for each configuration
        for config_idx, (config_key, data) in enumerate(datasets.items()):
            if alg_name in data:
                solutions = data[alg_name]
                
                # Extract the coordinates for Professor Conflicts vs Unassigned Activities
                x_values = [sol["professor_conflicts"] for sol in solutions]
                y_values = [sol[KEY_MAPPINGS[config_key]["unassigned"]] for sol in solutions]
                
                # Plot scatter points
                ax.scatter(x_values, y_values, 
                        color=COLORS[alg_idx], 
                        marker=CONFIG_MARKERS[config_idx],
                        s=100, alpha=CONFIG_ALPHA[config_idx],
                        label=f"{CONFIG_NAMES[config_idx]} ({len(solutions)} solutions)")
                
                # Connect points with lines to visualize the Pareto front
                points = sorted(zip(x_values, y_values), key=lambda p: p[0])
                if points:
                    x_sorted, y_sorted = zip(*points)
                    ax.plot(x_sorted, y_sorted, 
                           color=COLORS[alg_idx], 
                           linestyle=LINE_STYLES[config_idx],
                           alpha=0.5, linewidth=1.5)
        
        # Add execution time and Pareto front size to the legend
        pop50_time = EXECUTION_TIMES["50"][alg_name]
        pop100_time = EXECUTION_TIMES["100"][alg_name]
        pop50_size = PARETO_SIZES["50"][alg_name]
        pop100_size = PARETO_SIZES["100"][alg_name]
        
        time_legend = [
            Patch(color='none', label=f"50/50: {pop50_time:.2f}s, {pop50_size} solutions"),
            Patch(color='none', label=f"100/100: {pop100_time:.2f}s, {pop100_size} solutions"),
            Patch(color='none', label=f"Time increase: {pop100_time/pop50_time:.2f}x")
        ]
        
        # Add legend
        handles, _ = ax.get_legend_handles_labels()
        handles.extend(time_legend)
        
        # Set title and labels
        ax.set_title(f"{alg_name}: Population Size Comparison (4-Room Dataset)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Professor Conflicts (Lower is Better)", fontsize=12)
        ax.set_ylabel("Unassigned Activities (Lower is Better)", fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Force integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend
        ax.legend(handles=handles, fontsize=10, loc='upper right')
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{alg_name.replace('/', '_')}_population_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Comparison for {alg_name} saved to: {output_path}")
    
    # Create a combined visualization with all algorithms
    create_combined_comparison(datasets, output_dir)
    
    # Create performance metrics visualization
    create_performance_metrics_visualization(output_dir)
    
    return True

def create_combined_comparison(datasets, output_dir):
    """Create a single figure showing all algorithms and both configurations."""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # For each algorithm and configuration
    for alg_idx, alg_name in enumerate(ALGORITHM_NAMES):
        for config_idx, (config_key, data) in enumerate(datasets.items()):
            if alg_name in data:
                solutions = data[alg_name]
                
                # Extract the coordinates for Professor Conflicts vs Unassigned Activities
                x_values = [sol["professor_conflicts"] for sol in solutions]
                y_values = [sol[KEY_MAPPINGS[config_key]["unassigned"]] for sol in solutions]
                
                # Plot scatter points
                ax.scatter(x_values, y_values, 
                        color=COLORS[alg_idx], 
                        marker=CONFIG_MARKERS[config_idx],
                        s=80, alpha=CONFIG_ALPHA[config_idx],
                        label=f"{alg_name} - {CONFIG_NAMES[config_idx]}")
    
    # Set title and labels
    ax.set_title("Population Size Comparison Across All Algorithms (4-Room Dataset)", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Professor Conflicts (Lower is Better)", fontsize=12)
    ax.set_ylabel("Unassigned Activities (Lower is Better)", fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Force integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add legend
    ax.legend(fontsize=10, loc='upper right')
    
    # Save the figure
    output_path = os.path.join(output_dir, "combined_population_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined comparison saved to: {output_path}")

def create_performance_metrics_visualization(output_dir):
    """Create bar charts comparing execution time and solution quality metrics."""
    # Execution time comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data for bar chart
    algorithms = list(ALGORITHM_NAMES)
    pop50_times = [EXECUTION_TIMES["50"][alg] for alg in algorithms]
    pop100_times = [EXECUTION_TIMES["100"][alg] for alg in algorithms]
    
    # Positions for the bars
    x = np.arange(len(algorithms))
    width = 0.35
    
    # Plot execution times
    ax1.bar(x - width/2, pop50_times, width, label='50 Pop/Gen', color='skyblue')
    ax1.bar(x + width/2, pop100_times, width, label='100 Pop/Gen', color='navy')
    
    # Add time increase factors
    for i, alg in enumerate(algorithms):
        increase = EXECUTION_TIMES["100"][alg] / EXECUTION_TIMES["50"][alg]
        ax1.text(i, EXECUTION_TIMES["100"][alg] + 5, f"{increase:.2f}x", 
                ha='center', va='bottom', fontweight='bold')
    
    # Customize the chart
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Algorithm Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Plot Pareto front sizes
    pop50_sizes = [PARETO_SIZES["50"][alg] for alg in algorithms]
    pop100_sizes = [PARETO_SIZES["100"][alg] for alg in algorithms]
    
    ax2.bar(x - width/2, pop50_sizes, width, label='50 Pop/Gen', color='lightgreen')
    ax2.bar(x + width/2, pop100_sizes, width, label='100 Pop/Gen', color='darkgreen')
    
    # Add size increase factors
    for i, alg in enumerate(algorithms):
        increase = PARETO_SIZES["100"][alg] / PARETO_SIZES["50"][alg]
        ax2.text(i, PARETO_SIZES["100"][alg] + 5, f"{increase:.2f}x", 
                ha='center', va='bottom', fontweight='bold')
    
    # Customize the chart
    ax2.set_ylabel('Number of Pareto Solutions', fontsize=12)
    ax2.set_title('Pareto Front Size Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add an overall title
    plt.suptitle("Performance Metrics Comparison: 50 vs 100 Population/Generation", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, "performance_metrics_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Performance metrics comparison saved to: {output_path}")

if __name__ == "__main__":
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Define paths to data
    data_paths = {
        "50": os.path.join(parent_dir, "ga_results_4room", "pareto_front_data.json"),
        "100": os.path.join(parent_dir, "ga_results_4room_100", "pareto_front_data.json")
    }
    
    # Output directory
    output_dir = os.path.join(parent_dir, "visualization_results")
    
    # Create population size comparison visualizations
    success = create_population_size_comparison(data_paths, output_dir)
    
    if success:
        print("\nPopulation size comparison visualizations created successfully.")
    else:
        print("\nFailed to create population size comparison visualizations.")
