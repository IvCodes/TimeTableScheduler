"""
Enhanced Research-Quality Pareto Front Visualization

Creates a publication-ready Pareto front visualization with:
1. Clean scatter plot (no connecting lines)
2. Color-coded algorithms
3. Reference line connecting ideal and nadir points
4. Clear labels and consistent styling
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines

# Constants for consistent styling
ALG_NSGA2 = "NSGA-II"
ALG_SPEA2 = "SPEA2"
ALG_MOEAD = "MOEA/D"
ALGORITHM_NAMES = [ALG_NSGA2, ALG_SPEA2, ALG_MOEAD]

# Visual styling with brighter colors for better visibility
COLORS = ['#0072BD', '#00A651', '#FF0000']  # Bright Blue (NSGA-II), Bright Green (SPEA2), Bright Red (MOEA/D)
MARKERS = ['o', 'o', 'o']  # All circles as requested
DATASET_NAMES = ["4-Room Dataset", "7-Room Dataset"]
MARKER_SIZE = 140  # Much larger markers for publication
ALPHA = 0.9  # Higher opacity for better contrast
REFERENCE_COLOR = "#666666"  # Gray for reference line
LINE_WIDTH = 1.5

# Key mappings for different datasets
KEY_MAPPINGS = {
    "4room": {
        "unassigned": "unassigned", 
        "soft_score": "soft_score"
    },
    "7room": {
        "unassigned": "unassigned",
        "soft_score": "soft_score"
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

def create_enhanced_pareto_visualization(data_paths, output_dir, output_filename="enhanced_pareto_fronts.png"):
    """
    Create a publication-quality Pareto front visualization.
    
    Args:
        data_paths: Dictionary with dataset keys and paths
        output_dir: Directory to save output figures
        output_filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a 1x2 figure layout (one plot per dataset)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # For each dataset
    for i, (dataset_key, json_path) in enumerate(data_paths.items()):
        # Load data
        data = load_json_data(json_path)
        if not data:
            continue
            
        # Get the correct key mapping
        key_mapping = KEY_MAPPINGS[dataset_key]
        
        # Set the current axis
        ax = axes[i]
        
        # Track min/max values for reference line
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        # For each algorithm
        for alg_idx, alg_name in enumerate(ALGORITHM_NAMES):
            if alg_name in data:
                # Extract data for this algorithm
                solutions = data[alg_name]
                
                # Extract the coordinates for Professor Conflicts vs Unassigned Activities
                x_values = [sol["professor_conflicts"] for sol in solutions]
                y_values = [sol[key_mapping["unassigned"]] for sol in solutions]
                
                # Update min/max values
                min_x = min(min_x, min(x_values))
                max_x = max(max_x, max(x_values))
                min_y = min(min_y, min(y_values))
                max_y = max(max_y, max(y_values))
                
                # Plot scatter points without connecting lines
                ax.scatter(x_values, y_values, 
                          color=COLORS[alg_idx], 
                          marker=MARKERS[alg_idx],
                          s=MARKER_SIZE, alpha=ALPHA, 
                          edgecolors='white', linewidth=0.5,
                          label=f"{alg_name} ({len(solutions)} solutions)")
        
        # Add reference line from ideal to nadir point
        if min_x != float('inf') and min_y != float('inf'):
            # Add some padding to the reference line
            pad_x = (max_x - min_x) * 0.05
            pad_y = (max_y - min_y) * 0.05
            
            # Draw the reference line with arrows
            ax.plot([min_x - pad_x, max_x + pad_x], [max_y + pad_y, min_y - pad_y], 
                   color=REFERENCE_COLOR, linestyle='--', linewidth=LINE_WIDTH, alpha=0.7)
            
            # Only annotate the nadir point (removed ideal point annotation)
            ax.annotate("Nadir\nPoint", xy=(max_x, max_y), xytext=(max_x + pad_x*2, max_y + pad_y*2),
                       arrowprops=dict(arrowstyle='->', color=REFERENCE_COLOR, lw=1),
                       fontsize=10, color=REFERENCE_COLOR)
        
        # Set title and labels with research-quality typography
        ax.set_title(f"Pareto Front: {DATASET_NAMES[i]}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Professor Conflicts (Lower is Better)", fontsize=12)
        ax.set_ylabel("Unassigned Activities (Lower is Better)", fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Force integer ticks for better interpretability
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Ensure axes start at 0 or slightly below for clarity
        ax.set_xlim(left=max(0, min_x - pad_x*3))
        ax.set_ylim(bottom=max(0, min_y - pad_y*3))
        
        # Add legend with research-quality styling
        # Create custom legend elements
        legend_elements = [
            mlines.Line2D([0], [0], marker=MARKERS[i], color='w', markerfacecolor=COLORS[i], 
                        markersize=10, label=f"{name} ({len(data[name]) if name in data else 0} solutions)")
            for i, name in enumerate(ALGORITHM_NAMES)
        ]
        
        # Add the legend
        ax.legend(handles=legend_elements, fontsize=10, framealpha=0.9, 
                 edgecolor='lightgray', loc='upper right')
    
    # Add an overall title
    plt.suptitle("Pareto Front Comparison: Multi-Objective Timetable Scheduling", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout for optimal spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure in high resolution
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Enhanced Pareto front visualization saved to: {output_path}")
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
    
    # Create enhanced visualization
    output_path = create_enhanced_pareto_visualization(data_paths, output_dir, "publication_pareto_fronts.png")
    
    print(f"\nPublication-quality visualization created successfully: {output_path}")
