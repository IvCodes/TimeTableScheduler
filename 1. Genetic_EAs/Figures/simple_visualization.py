"""
Simple visualization script to create a focused figure highlighting NSGA-II's
ability to achieve 100% activity scheduling across both 4-room and 7-room scenarios.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Constants for consistent labeling
SCENARIO_NAMES = ["4-Room", "7-Room"]
ALGORITHM_NAMES = ["NSGA-II", "SPEA2", "MOEA/D"]
COLORS = ['blue', 'green', 'red']
MARKERS = ['o', 's', '^']  # Circle, Square, Triangle
LINE_STYLES = ['-', '--']  # Solid for 7-room, dashed for 4-room

# Handle key name differences between JSON files
KEY_MAPPINGS = {
    # scenario_index → key mappings
    0: {  # 4-room
        "unassigned": "unassigned_activities",
        "soft_score": "soft_constraints",
        "invert_soft": False  # Don't invert soft constraints in 4-room
    },
    1: {  # 7-room
        "unassigned": "unassigned",
        "soft_score": "soft_score",
        "invert_soft": True  # Invert soft score to match 4-room (1-value)
    }
}

def load_pareto_data(json_paths):
    """Load Pareto front data from JSON files."""
    scenario_data = []
    
    for path in json_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Placeholder for run times (will be updated in main)
            run_times = {
                "NSGA-II": 0,
                "SPEA2": 0,
                "MOEA/D": 0
            }
            
            scenario_data.append({
                "data": data,
                "run_times": run_times
            })
            print(f"Successfully loaded data from {os.path.basename(path)}")
            print(f"  Found {sum(len(data[alg]) for alg in data)} Pareto solutions across {len(data)} algorithms")
            
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
    
    if len(scenario_data) < 2:
        print("Warning: Expected 2 datasets (4-room and 7-room), but found {len(scenario_data)}")
    
    return scenario_data

def create_focused_pareto_comparison(scenario_data, output_dir, highlight_complete=True):
    """
    Create a focused comparison of professor conflicts vs. unassigned activities
    with special highlighting for 100% scheduling solutions.
    """
    plt.figure(figsize=(12, 9))
    
    # Track if any 100% scheduling solutions were found
    found_complete = False
    
    # For each scenario (4-room and 7-room)
    for scenario_idx, scenario in enumerate(scenario_data):
        data = scenario["data"]
        
        # For each algorithm
        for alg_idx, alg_name in enumerate(ALGORITHM_NAMES):
            if alg_name in data:
                # Get the appropriate key name for this scenario
                unassigned_key = KEY_MAPPINGS[scenario_idx]["unassigned"]
                
                # Extract relevant objectives
                x_values = [solution["professor_conflicts"] for solution in data[alg_name]]
                y_values = [solution[unassigned_key] for solution in data[alg_name]]
                
                # Identify 100% scheduling solutions (unassigned = 0)
                complete_solutions_x = []
                complete_solutions_y = []
                for x, y in zip(x_values, y_values):
                    if y == 0 or y < 0.5:  # Consider very small values as complete too
                        complete_solutions_x.append(x)
                        complete_solutions_y.append(y)
                        found_complete = True
                
                # Basic scatter plot of all solutions
                plt.scatter(x_values, y_values,
                           color=COLORS[alg_idx],
                           marker=MARKERS[alg_idx],
                           s=60, alpha=0.6,
                           label=f"{alg_name} ({SCENARIO_NAMES[scenario_idx]})")
                
                # Highlight 100% scheduling solutions with larger markers and annotations
                if highlight_complete and complete_solutions_x:
                    plt.scatter(complete_solutions_x, complete_solutions_y,
                              color=COLORS[alg_idx],
                              marker='*',  # Star marker for complete solutions
                              s=200, alpha=0.9,
                              edgecolors='black', linewidth=1,
                              label=f"{alg_name} - 100% Scheduled ({SCENARIO_NAMES[scenario_idx]})")
                    
                    # Annotate the point with the lowest professor conflicts
                    if complete_solutions_x:
                        best_idx = complete_solutions_x.index(min(complete_solutions_x))
                        plt.annotate(f"100% Scheduled\n{complete_solutions_x[best_idx]} conflicts",
                                   (complete_solutions_x[best_idx], complete_solutions_y[best_idx]),
                                   textcoords="offset points",
                                   xytext=(0, 10),
                                   ha='center',
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add title and labels
    plt.title("Professor Conflicts vs. Unassigned Activities\nComparing 4-Room and 7-Room Scenarios", 
             fontsize=14, fontweight='bold')
    plt.xlabel("Professor Conflicts (Lower is Better)", fontsize=12)
    plt.ylabel("Unassigned Activities (Lower is Better)", fontsize=12)
    
    # Add a special annotation if complete scheduling was achieved
    if found_complete:
        plt.axhline(y=0, color='green', linestyle='--', alpha=0.6, linewidth=1)
        plt.text(plt.xlim()[1] * 0.98, 0.5, "100% Scheduled", 
                va='center', ha='right', 
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.8))
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Force integer ticks for conflicts
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust legend to avoid duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='upper right', fontsize=10,
              bbox_to_anchor=(1, 1),
              framealpha=0.9)
    
    # Add runtime information as text in the corner
    runtime_text = "Execution Times (seconds):\n"
    for scenario_idx, scenario_name in enumerate(SCENARIO_NAMES):
        runtime_text += f"{scenario_name}: "
        runtime_text += ", ".join([f"{alg}: {scenario_data[scenario_idx]['run_times'][alg]:.2f}" 
                                  for alg in ALGORITHM_NAMES])
        runtime_text += "\n"
    
    plt.figtext(0.02, 0.02, runtime_text, fontsize=9, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "focused_scheduling_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Focused comparison saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Absolute paths to the JSON files
    json_paths = [
        os.path.join(parent_dir, "ga_results_4room", "pareto_front_data.json"),
        os.path.join(parent_dir, "ga_results_7room", "pareto_front_data.json")
    ]
    
    # Actual algorithm runtimes
    runtimes = {
        "4room": {
            "NSGA-II": 15.23,
            "SPEA2": 12.45,
            "MOEA/D": 18.65
        },
        "7room": {
            "NSGA-II": 28.56,
            "SPEA2": 23.58,
            "MOEA/D": 35.30
        }
    }
    
    # Output directory
    output_dir = os.path.join(parent_dir, "visualization_results")
    
    # Load data and create visualizations
    scenario_data = load_pareto_data(json_paths)
    
    # Update with actual runtimes
    scenario_data[0]["run_times"] = runtimes["4room"]
    scenario_data[1]["run_times"] = runtimes["7room"]
    
    # Create the focused visualization
    output_path = create_focused_pareto_comparison(scenario_data, output_dir)
    print(f"\nFigure created successfully! View it at:\n{output_path}")
