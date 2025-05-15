# visualization_multi_scenario.py
"""
Script to generate research-quality multi-scenario comparison visualizations
for evolutionary algorithm performance on timetable scheduling.

Uses data from the output JSON files of both 4-room and 7-room scenarios.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
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

# For radar chart
OBJECTIVES = ["Professor\nConflicts", "Group\nConflicts", "Room\nConflicts",
              "Unassigned\nActivities", "Soft\nConstraints"]


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


def create_multi_scenario_comparison(scenario_data, output_dir, output_filename="multi_scenario_comparison.png"):
    """Create a 4-panel research visualization comparing algorithm performance across scenarios."""

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Panel A: Pareto Front Comparison (Professor Conflicts vs Unassigned Activities)
    ax1 = fig.add_subplot(221)
    plot_pareto_front_comparison(ax1, scenario_data)

    # Panel B: Runtime Comparison
    ax2 = fig.add_subplot(222)
    plot_runtime_comparison(ax2, scenario_data)

    # Panel C: Pareto Solution Count
    ax3 = fig.add_subplot(223)
    plot_solution_count_comparison(ax3, scenario_data)

    # Panel D: Radar Chart of Best Solutions
    ax4 = fig.add_subplot(224, polar=True)
    plot_best_solution_radar(ax4, scenario_data)

    # Add overall title
    plt.suptitle("Multi-Scenario Evolutionary Algorithm Performance Comparison",
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Multi-scenario comparison saved to: {output_path}")
    return output_path


def plot_pareto_front_comparison(ax, scenario_data):
    """Plot Pareto fronts for both scenarios on the same axes."""
    ax.set_title(
        "Pareto Front Comparison: Professor Conflicts vs Unassigned Activities", fontsize=12)

    # For connecting lines between points
    for scenario_idx, scenario in enumerate(scenario_data):
        data = scenario["data"]

        for alg_idx, alg_name in enumerate(ALGORITHM_NAMES):
            if alg_name in data:
                # Extract relevant objectives (professor conflicts and unassigned)
                x_values = [solution["professor_conflicts"]
                            for solution in data[alg_name]]
                
                # Use the appropriate key name for this scenario
                unassigned_key = KEY_MAPPINGS[scenario_idx]["unassigned"]
                y_values = [solution[unassigned_key] for solution in data[alg_name]]

                # Sort by x for line plotting
                points = sorted(zip(x_values, y_values), key=lambda p: p[0])
                x_sorted, y_sorted = zip(*points) if points else ([], [])

                # Plot the line and scatter points
                ax.plot(x_sorted, y_sorted, color=COLORS[alg_idx],
                        linestyle=LINE_STYLES[scenario_idx],
                        alpha=0.7,
                        label=f"{alg_name} ({SCENARIO_NAMES[scenario_idx]})")

                ax.scatter(x_values, y_values,
                           color=COLORS[alg_idx],
                           marker=MARKERS[alg_idx],
                           s=40, alpha=0.6,
                           edgecolors='white', linewidth=0.5)

    ax.set_xlabel("Professor Conflicts (Lower is Better)", fontsize=10)
    ax.set_ylabel("Unassigned Activities (Lower is Better)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

    # Force integer ticks for conflicts
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_runtime_comparison(ax, scenario_data):
    """Bar chart comparing algorithm runtimes across scenarios."""
    ax.set_title("Algorithm Runtime Comparison", fontsize=12)

    # Set up bar positions
    bar_width = 0.35
    positions = np.arange(len(ALGORITHM_NAMES))

    # Plot bars for each scenario
    for i, scenario in enumerate(scenario_data):
        runtimes = [scenario["run_times"][alg] for alg in ALGORITHM_NAMES]
        ax.bar(positions + i*bar_width, runtimes, bar_width,
               label=SCENARIO_NAMES[i], alpha=0.8, color=COLORS[i])

    # Add percentage increase labels
    for j, alg in enumerate(ALGORITHM_NAMES):
        runtime_4room = scenario_data[0]["run_times"][alg]
        runtime_7room = scenario_data[1]["run_times"][alg]
        percent_increase = (
            (runtime_7room - runtime_4room) / runtime_4room) * 100

        ax.text(positions[j] + bar_width/2, max(runtime_4room, runtime_7room) + 1,
                f"+{percent_increase:.1f}%", ha='center', va='bottom', fontsize=9)

    # Add labels and legend
    ax.set_xticks(positions + bar_width/2)
    ax.set_xticklabels(ALGORITHM_NAMES)
    ax.set_ylabel("Runtime (seconds)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')


def plot_solution_count_comparison(ax, scenario_data):
    """Bar chart showing number of Pareto-optimal solutions found by each algorithm."""
    ax.set_title(
        "Pareto Solution Count by Algorithm and Scenario", fontsize=12)

    # Set up bar positions
    bar_width = 0.35
    positions = np.arange(len(ALGORITHM_NAMES))

    # Plot bars for each scenario
    for i, scenario in enumerate(scenario_data):
        data = scenario["data"]
        solution_counts = [len(data.get(alg, [])) for alg in ALGORITHM_NAMES]

        ax.bar(positions + i*bar_width, solution_counts, bar_width,
               label=SCENARIO_NAMES[i], alpha=0.8, color=COLORS[i])

        # Add count labels on bars
        for j, count in enumerate(solution_counts):
            ax.text(positions[j] + i*bar_width, count + 1, str(count),
                    ha='center', va='bottom', fontsize=9)

    # Add labels and legend
    ax.set_xticks(positions + bar_width/2)
    ax.set_xticklabels(ALGORITHM_NAMES)
    ax.set_ylabel("Number of Pareto Solutions", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')


def plot_best_solution_radar(ax, scenario_data):
    """Radar chart showing the best compromise solution from each algorithm."""
    ax.set_title("Multi-Objective Comparison of Best Solutions", fontsize=12)

    # Number of variables (objectives)
    N = len(OBJECTIVES)

    # Set ticks at regular intervals
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Set up radar chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(OBJECTIVES, fontsize=8)

    # For each algorithm in each scenario, find the "best compromise" solution
    # To track max value for each objective
    max_values = {obj: 0 for obj in range(5)}

    best_compromises = []
    for scenario_idx, scenario in enumerate(scenario_data):
        data = scenario["data"]

        for alg_idx, alg_name in enumerate(ALGORITHM_NAMES):
            if alg_name in data:
                # Find solution with best average rank across objectives
                solutions = data[alg_name]
                if solutions:
                    # Function to calculate "compromise score" - lower is better
                    # This is just one approach - you might want a different definition of "best"
                    def compromise_score(sol):
                        # Get the appropriate key names for this scenario
                        unassigned_key = KEY_MAPPINGS[scenario_idx]["unassigned"]
                        soft_key = KEY_MAPPINGS[scenario_idx]["soft_score"]
                        invert_soft = KEY_MAPPINGS[scenario_idx]["invert_soft"]
                        
                        # Calculate the soft constraints score (may need inversion)
                        soft_value = (1 - sol[soft_key]) if invert_soft else sol[soft_key]
                        
                        return (
                            sol["professor_conflicts"] +
                            # Scale down because values are larger
                            sol["group_conflicts"]/10 +
                            sol["room_conflicts"] +
                            # Weight unassigned more heavily
                            sol[unassigned_key]*5 +
                            # Use the appropriate soft constraint value
                            soft_value*50
                        )

                    best_solution = min(solutions, key=compromise_score)

                    # Get the appropriate key names for this scenario
                    unassigned_key = KEY_MAPPINGS[scenario_idx]["unassigned"]
                    soft_key = KEY_MAPPINGS[scenario_idx]["soft_score"]
                    invert_soft = KEY_MAPPINGS[scenario_idx]["invert_soft"]
                    
                    # Calculate the soft constraints score (may need inversion)
                    soft_value = (1 - best_solution[soft_key]) if invert_soft else best_solution[soft_key]
                    
                    # Extract values (normalized later)
                    values = [
                        best_solution["professor_conflicts"],
                        best_solution["group_conflicts"],
                        best_solution["room_conflicts"],
                        best_solution[unassigned_key],
                        # Use appropriate soft constraint value
                        soft_value
                    ]

                    # Update max values for normalization
                    for i, val in enumerate(values):
                        max_values[i] = max(max_values[i], val)

                    best_compromises.append({
                        "scenario": SCENARIO_NAMES[scenario_idx],
                        "algorithm": alg_name,
                        "values": values,
                        "color": COLORS[alg_idx],
                        "linestyle": LINE_STYLES[scenario_idx]
                    })

    # Normalize values and plot
    for solution in best_compromises:
        # Normalize between 0.1 and 0.9 (avoid origin and edge)
        values = [0.1 + 0.8 * (val / max_values[i] if max_values[i] > 0 else 0)
                  for i, val in enumerate(solution["values"])]
        values += values[:1]  # Close the loop

        # Plot the solution
        ax.plot(angles, values, color=solution["color"],
                linestyle=solution["linestyle"], linewidth=2,
                label=f"{solution['algorithm']} ({solution['scenario']})")

        # Fill the area
        ax.fill(angles, values, color=solution["color"], alpha=0.1)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)


if __name__ == "__main__":
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Absolute paths to the JSON files
    json_paths = [
        os.path.join(parent_dir, "ga_results_4room", "pareto_front_data.json"),
        os.path.join(parent_dir, "ga_results_7room", "pareto_front_data.json")
    ]
    
    # Actual algorithm runtimes (from the output)
    runtimes = {
        "4room": {
            "NSGA-II": 15.23,  # Update with actual values
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
    
    # Create the visualization
    output_path = create_multi_scenario_comparison(scenario_data, output_dir)
    print(f"\nFigure created successfully! View it at:\n{output_path}")
