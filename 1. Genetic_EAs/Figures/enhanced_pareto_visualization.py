"""
Enhanced Pareto front visualization for publication - showing only nondominated solutions
with connected lines and better visual elements for clearer algorithm comparison.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
from scipy.spatial import ConvexHull

# Constants for consistent labeling and visualization
SCENARIO_NAMES = ["4-Room", "7-Room"]
ALGORITHM_NAMES = ["NSGA-II", "SPEA2", "MOEA/D"]
COLORS = ['#0072BD', '#00A651', '#FF0000']  # Bright Blue, Bright Green, Bright Red
MARKERS = ['o', 'o', 'o']  # All circles as per previous preference
LINE_STYLES = ['-', '--', ':']  # Different line styles for algorithms

# Updated key mappings based on actual JSON structure
KEY_MAPPINGS = {
    0: {  # 4-room
        "unassigned": "unassigned",
        "soft_score": "soft_score",
        "invert_soft": False
    },
    1: {  # 7-room
        "unassigned": "unassigned",
        "soft_score": "soft_score",
        "invert_soft": False  # Check if this needs to be True for 7-room
    }
}

def load_pareto_data(json_paths):
    """Load Pareto front data from JSON files."""
    all_data = []
    
    for scenario_idx, json_path in enumerate(json_paths):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        scenario_data = []
        for alg_idx, alg_name in enumerate(ALGORITHM_NAMES):
            if alg_name in data:
                # Extract mapping keys for this scenario
                unassigned_key = KEY_MAPPINGS[scenario_idx]["unassigned"]
                soft_key = KEY_MAPPINGS[scenario_idx]["soft_score"]
                invert_soft = KEY_MAPPINGS[scenario_idx]["invert_soft"]
                
                points = []
                for solution in data[alg_name]:
                    x = solution[unassigned_key]
                    # Handle soft score inversion if needed
                    y = solution[soft_key]
                    if invert_soft:
                        y = 1 - y
                    points.append((x, y))
                
                # Only keep nondominated points (true Pareto front)
                pareto_points = get_pareto_front(points)
                
                scenario_data.append({
                    "algorithm": alg_name,
                    "points": pareto_points,
                    "color": COLORS[alg_idx],
                    "marker": MARKERS[alg_idx],
                    "line_style": LINE_STYLES[alg_idx]
                })
        
        all_data.append({
            "scenario": SCENARIO_NAMES[scenario_idx],
            "data": scenario_data
        })
    
    return all_data

def get_pareto_front(points):
    """Extract nondominated points (Pareto front) from a set of points.
    For minimization of both objectives."""
    points = np.array(points)
    is_efficient = np.ones(points.shape[0], dtype=bool)
    
    for i, point in enumerate(points):
        # Keep any point with at least one value better than all others
        if is_efficient[i]:
            # Find all points strictly dominated by this point
            dominated = np.all(point <= points, axis=1) & np.any(point < points, axis=1)
            is_efficient[dominated] = False
    
    return points[is_efficient].tolist()

def calculate_hypervolume(points, reference_point):
    """Calculate hypervolume metric for a Pareto front."""
    # Simple 2D implementation - area dominated by the Pareto front
    if len(points) == 0:
        return 0
    
    # Add reference point for hull calculation (makes it work for small sets)
    points_with_ref = np.vstack([points, reference_point])
    
    # Calculate convex hull
    try:
        hull = ConvexHull(points_with_ref)
        return hull.volume
    except Exception:
        # Fallback if hull calculation fails (e.g., for < 3 points)
        return 0

def create_enhanced_pareto_visualization(data_paths, save_path):
    """Create an enhanced Pareto front visualization."""
    data = load_pareto_data(data_paths)
    
    # Set up the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reference point for hypervolume calculation (upper right corner + margin)
    reference_point = np.array([200, 1.5])
    hypervolume_metrics = {}
    
    # Plot each scenario
    for i, scenario in enumerate(data):
        ax = axes[i]
        ax.set_title(f"{scenario['scenario']} Scenario")
        ax.set_xlabel("Unassigned Activities")
        ax.set_ylabel("Constraint Violations Score")
        
        # Plot data for each algorithm
        for alg_data in scenario['data']:
            points = np.array(alg_data['points'])
            
            if len(points) > 0:
                # Sort points for connecting lines (by x-coordinate)
                sorted_indices = np.argsort(points[:, 0])
                sorted_points = points[sorted_indices]
                
                # Plot points and connecting lines
                ax.plot(sorted_points[:, 0], sorted_points[:, 1], 
                       linestyle=alg_data['line_style'],
                       color=alg_data['color'], 
                       marker=alg_data['marker'],
                       markersize=7,  # Reasonable marker size
                       label=alg_data['algorithm'])
                
                # Calculate hypervolume
                hv = calculate_hypervolume(points, reference_point)
                if scenario['scenario'] not in hypervolume_metrics:
                    hypervolume_metrics[scenario['scenario']] = {}
                hypervolume_metrics[scenario['scenario']][alg_data['algorithm']] = hv
                
                # Find and annotate extreme points (only for MOEA/D as it's reportedly best)
                if alg_data['algorithm'] == "MOEA/D":
                    # Find nadir point (max in both dimensions)
                    nadir_idx = np.argmax(np.sum(points, axis=1))
                    nadir_point = points[nadir_idx]
                    ax.annotate('Nadir', 
                               xy=(nadir_point[0], nadir_point[1]),
                               xytext=(nadir_point[0]+5, nadir_point[1]+0.05),
                               arrowprops=dict(facecolor='black', shrink=0.05, width=1))
                    
                    # Find ideal point (min in both dimensions)
                    min_x_idx = np.argmin(points[:, 0])
                    min_y_idx = np.argmin(points[:, 1])
                    ax.plot(points[min_x_idx, 0], points[min_y_idx, 1], 'k*', markersize=8)
                    ax.annotate('Ideal', 
                               xy=(points[min_x_idx, 0], points[min_y_idx, 1]),
                               xytext=(points[min_x_idx, 0]-15, points[min_y_idx, 1]-0.1),
                               arrowprops=dict(facecolor='black', shrink=0.05, width=1))
        
        # Add diagonal reference line suggesting equal weighting
        ax.plot([0, 100], [0, 1], 'k--', alpha=0.3)
        ax.text(60, 0.7, 'Equal\nWeighting', rotation=45, alpha=0.5)
        
        # Set reasonable axis limits
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        
        # Add hypervolume metrics as an inset box
        textbox = '\n'.join([f"{alg}: {hypervolume_metrics[scenario['scenario']][alg]:.2f}" 
                            for alg in ALGORITHM_NAMES if alg in hypervolume_metrics[scenario['scenario']]])
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, f"Hypervolume:\n{textbox}", transform=ax.transAxes, 
               fontsize=8, verticalalignment='top', bbox=props)
    
    # Add a single legend for both subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0))
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced Pareto visualization saved to {save_path}")

if __name__ == "__main__":
    # Updated paths to match actual file locations
    json_paths = [
        "c:/Users/Easara_200005/Desktop/Projects/Research_conference/TimeTableScheduler/1. Genetic_EAs/ga_results_4room_100/pareto_front_data.json",
        "c:/Users/Easara_200005/Desktop/Projects/Research_conference/TimeTableScheduler/1. Genetic_EAs/ga_results_7room/pareto_front_data.json"
    ]
    
    output_path = "c:/Users/Easara_200005/Desktop/Projects/Research_conference/TimeTableScheduler/1. Genetic_EAs/visualization_results/enhanced_pareto_fronts.png"
    
    create_enhanced_pareto_visualization(json_paths, output_path)
