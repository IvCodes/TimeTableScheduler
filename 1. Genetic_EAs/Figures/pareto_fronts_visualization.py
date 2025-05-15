"""
Pareto Front Visualization Script for Timetable Scheduling

This script creates research-quality Pareto front visualizations
for both 4-room and 7-room datasets to directly address reviewer comments
about missing Pareto front visualizations in the paper.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# Constants for better readability
ALGORITHM_NAMES = ["NSGA-II", "SPEA2", "MOEA/D"]
DATASET_NAMES = ["4-Room Dataset", "7-Room Dataset"]
COLORS = ['blue', 'green', 'red']
MARKERS = ['o', 's', '^']  # Circle, Square, Triangle
STYLES = ['-', '--', '-.']  # Different line styles for algorithms

# Key mappings for different datasets
KEY_MAPPINGS = {
    "4room": {
        "unassigned": "unassigned_activities",
        "soft_score": "soft_constraints",
    },
    "7room": {
        "unassigned": "unassigned",
        "soft_score": "soft_score",
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

def get_key_mapping(dataset_type):
    """Get the key mapping for a dataset type."""
    if "4" in dataset_type:
        return KEY_MAPPINGS["4room"]
    else:
        return KEY_MAPPINGS["7room"]

def create_pareto_front_visualization(data_paths, output_dir):
    """
    Create research-quality Pareto front visualizations for both datasets.
    
    Args:
        data_paths: Dictionary with dataset keys and paths
        output_dir: Directory to save the output figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a 1x2 figure layout (one figure per dataset)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # For each dataset
    for i, (dataset_key, json_path) in enumerate(data_paths.items()):
        # Load data
        data = load_json_data(json_path)
        if not data:
            continue
            
        # Get the correct key mapping
        key_mapping = get_key_mapping(dataset_key)
        
        # Set the current axis for this dataset
        ax = axes[i]
        
        # For each algorithm
        for alg_idx, alg_name in enumerate(ALGORITHM_NAMES):
            if alg_name in data:
                # Extract data for this algorithm
                solutions = data[alg_name]
                
                # Extract the coordinates for Professor Conflicts vs Unassigned Activities
                x_values = [sol["professor_conflicts"] for sol in solutions]
                y_values = [sol[key_mapping["unassigned"]] for sol in solutions]
                
                # Plot scatter points
                ax.scatter(x_values, y_values, 
                          color=COLORS[alg_idx], 
                          marker=MARKERS[alg_idx],
                          s=80, alpha=0.6, 
                          label=f"{alg_name} ({len(solutions)} solutions)")
                
                # Sort points by x-value for connecting lines
                points = sorted(zip(x_values, y_values), key=lambda p: p[0])
                if points:
                    x_sorted, y_sorted = zip(*points)
                    # Plot connecting lines for visualization of Pareto front shape
                    ax.plot(x_sorted, y_sorted, 
                           color=COLORS[alg_idx], 
                           linestyle=STYLES[alg_idx],
                           alpha=0.5, linewidth=1.5)
        
        # Set title and labels
        ax.set_title(f"Pareto Front: {DATASET_NAMES[i]}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Professor Conflicts (Lower is Better)", fontsize=12)
        ax.set_ylabel("Unassigned Activities (Lower is Better)", fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Force integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend with cleaner labels
        ax.legend(fontsize=10, loc='upper right')
    
    # Add an overall title
    plt.suptitle("Pareto Front Comparison for Multi-Objective Timetable Scheduling", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, "pareto_fronts_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Pareto front visualization saved to: {output_path}")
    return output_path

def create_triple_objective_visualization(data_paths, output_dir):
    """
    Create a visualization showing three objectives simultaneously.
    
    Uses Professor Conflicts on X-axis, Unassigned Activities on Y-axis,
    and bubble size to represent Room Conflicts.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a 1x2 figure layout (one figure per dataset)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # For each dataset
    for i, (dataset_key, json_path) in enumerate(data_paths.items()):
        # Load data
        data = load_json_data(json_path)
        if not data:
            continue
            
        # Get the correct key mapping
        key_mapping = get_key_mapping(dataset_key)
        
        # Set the current axis for this dataset
        ax = axes[i]
        
        # For each algorithm
        for alg_idx, alg_name in enumerate(ALGORITHM_NAMES):
            if alg_name in data:
                # Extract data for this algorithm
                solutions = data[alg_name]
                
                # Extract the coordinates for three objectives
                x_values = [sol["professor_conflicts"] for sol in solutions]
                y_values = [sol[key_mapping["unassigned"]] for sol in solutions]
                z_values = [sol["room_conflicts"] for sol in solutions]
                
                # Scale the size based on room conflicts (larger bubbles = more conflicts)
                # Add a minimum size to ensure visibility of small values
                sizes = [max(20, 20 + z * 3) for z in z_values]
                
                # Plot bubble chart
                scatter = ax.scatter(x_values, y_values, 
                           s=sizes, 
                           c=[COLORS[alg_idx]] * len(x_values),
                           alpha=0.6,
                           marker=MARKERS[alg_idx],
                           edgecolors='white', linewidth=0.5,
                           label=f"{alg_name}")
        
        # Set title and labels
        ax.set_title(f"Triple Objective View: {DATASET_NAMES[i]}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Professor Conflicts (Lower is Better)", fontsize=12)
        ax.set_ylabel("Unassigned Activities (Lower is Better)", fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Force integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Create custom legend elements
        legend_elements = [
            Line2D([0], [0], marker=MARKERS[i], color='w', markerfacecolor=COLORS[i], 
                   markersize=10, label=ALGORITHM_NAMES[i])
            for i in range(len(ALGORITHM_NAMES))
        ]
        
        # Add custom legend explaining bubble size
        sizes_legend = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=5, alpha=0.6, label='Small Room Conflicts'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=10, alpha=0.6, label='Medium Room Conflicts'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=15, alpha=0.6, label='Large Room Conflicts')
        ]
        
        # Combine legends
        all_legend_elements = legend_elements + sizes_legend
        
        # Add the combined legend
        ax.legend(handles=all_legend_elements, fontsize=9, loc='upper right')
    
    # Add an overall title
    plt.suptitle("Triple Objective Visualization (Bubble Size = Room Conflicts)", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, "triple_objective_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Triple objective visualization saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Paths to JSON data files
    data_paths = {
        "4room": os.path.join(parent_dir, "ga_results_4room", "pareto_front_data.json"),
        "7room": os.path.join(parent_dir, "ga_results_7room", "pareto_front_data.json")
    }
    
    # Output directory
    output_dir = os.path.join(parent_dir, "visualization_results")
    
    # Create visualizations
    pareto_path = create_pareto_front_visualization(data_paths, output_dir)
    triple_obj_path = create_triple_objective_visualization(data_paths, output_dir)
    
    print("\nVisualizations created successfully:")
    print(f"1. Pareto Front Comparison: {pareto_path}")
    print(f"2. Triple Objective View: {triple_obj_path}")
