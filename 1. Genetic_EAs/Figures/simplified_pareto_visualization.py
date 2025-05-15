"""
Simplified Pareto front visualization for publication - focusing on clarity and readability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

# Constants for consistent labeling and visualization
SCENARIO_NAMES = ["4-Room", "7-Room"]
ALGORITHM_NAMES = ["NSGA-II", "SPEA2", "MOEA/D"]
COLORS = ['#0072BD', '#00A651', '#FF0000']  # Bright Blue, Bright Green, Bright Red
MARKERS = ['o', 'o', 'o']  # All circles as per previous preference
MARKER_SIZE = 140  # Much larger markers for better visibility

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
        "invert_soft": False
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
                    "marker": MARKERS[alg_idx]
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

def create_simplified_pareto_visualization(data_paths, save_path):
    """Create a simplified Pareto front visualization."""
    data = load_pareto_data(data_paths)
    
    # Set up the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot each scenario
    for i, scenario in enumerate(data):
        ax = axes[i]
        ax.set_title(f"{scenario['scenario']} Scenario", fontsize=12)
        ax.set_xlabel("Unassigned Activities", fontsize=11)
        ax.set_ylabel("Constraint Violations Score", fontsize=11)
        
        # Plot data for each algorithm
        for alg_data in scenario['data']:
            points = np.array(alg_data['points'])
            
            if len(points) > 0:
                # Just plot the scatter points without connecting lines for clarity
                ax.scatter(points[:, 0], points[:, 1], 
                         color=alg_data['color'], 
                         marker=alg_data['marker'],
                         s=MARKER_SIZE,  # Much larger markers
                         label=alg_data['algorithm'])
                
        # Set reasonable axis limits with some padding
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    
    # Add a single legend for both subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0))
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simplified Pareto visualization saved to {save_path}")

if __name__ == "__main__":
    # Updated paths to match actual file locations
    json_paths = [
        "c:/Users/Easara_200005/Desktop/Projects/Research_conference/TimeTableScheduler/1. Genetic_EAs/ga_results_4room_100/pareto_front_data.json",
        "c:/Users/Easara_200005/Desktop/Projects/Research_conference/TimeTableScheduler/1. Genetic_EAs/ga_results_7room/pareto_front_data.json"
    ]
    
    output_path = "c:/Users/Easara_200005/Desktop/Projects/Research_conference/TimeTableScheduler/1. Genetic_EAs/visualization_results/simplified_pareto_fronts.png"
    
    create_simplified_pareto_visualization(json_paths, output_path)
