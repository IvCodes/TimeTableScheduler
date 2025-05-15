"""
Visualization module for Evolutionary Algorithm results.

This module handles visualization of EA algorithm results, keeping it separate from the 
core algorithm implementation for better modularity.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# Add the root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import common visualization module
from common.visualization import (
    create_output_dir,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_parallel_coordinates,
    plot_metrics_comparison,
    plot_constraint_breakdown,
    plot_resource_utilization
)

def visualize_ea_results(results, output_dir="ea_results"):
    """
    Generate visualizations from EA algorithm results.
    
    Args:
        results: Dictionary containing all algorithm results
        output_dir: Base output directory
    """
    # Create output directory with room information
    room_count = results.get("room_count", "unknown")
    output_dir = f"{output_dir}_{room_count}room"
    os.makedirs(output_dir, exist_ok=True)

    # Define constants for objective labels to avoid repetition
    PROF_CONFLICTS = "Professor Conflicts"
    UNASSIGNED_ACTIVITIES = "Unassigned Activities"
    SOFT_SCORE = "Soft Score" # Used in one of the 2D plots
    
    # Extract algorithm names and results
    algorithm_names = list(results["pareto_fronts"].keys())
    
    # Export pareto front data to JSON
    print("\n--- Exporting Pareto Front Data to JSON ---")
    with open(os.path.join(output_dir, "pareto_front_data.json"), "w") as f:
        json.dump(results["pareto_fronts"], f, indent=2)
    
    # Generate simple 2D Pareto front plots for selected objectives
    print("\n--- Generating Simple Pareto Front Plots ---")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # Plot first set of objectives (0 vs 3)
    plot_pareto_front_2d(
        [list(results["pareto_fronts"][algo]) for algo in algorithm_names],
        algorithm_names,
        colors,
        markers,
        0, 3,
        PROF_CONFLICTS, UNASSIGNED_ACTIVITIES,
        "Pareto Front: Professor Conflicts vs. Unassigned Activities",
        output_dir,
        "simple_pareto_0v3.png"
    )
    print(f"Simple plot saved to {output_dir}/simple_pareto_0v3.png")
    
    # Plot second set of objectives (3 vs 4)
    plot_pareto_front_2d(
        [list(results["pareto_fronts"][algo]) for algo in algorithm_names],
        algorithm_names,
        colors,
        markers,
        3, 4,
        UNASSIGNED_ACTIVITIES, SOFT_SCORE,
        "Pareto Front: Unassigned Activities vs. Soft Score",
        output_dir,
        "simple_pareto_3v4.png"
    )
    print(f"Simple plot saved to {output_dir}/simple_pareto_3v4.png")
    
    # Generate combined Pareto front plots
    plot_pareto_front_2d(
        [list(results["pareto_fronts"][algo]) for algo in algorithm_names],
        algorithm_names,
        colors,
        markers,
        3, 4,
        UNASSIGNED_ACTIVITIES, SOFT_SCORE,
        "Combined EA Pareto Front: Unassigned vs. Soft Score",
        output_dir,
        "combined_ea_pareto_3v4.png"
    )
    print(f"Combined Pareto plot saved to {output_dir}/combined_ea_pareto_3v4.png")
    
    plot_pareto_front_2d(
        [list(results["pareto_fronts"][algo]) for algo in algorithm_names],
        algorithm_names,
        colors,
        markers,
        0, 3,
        PROF_CONFLICTS, UNASSIGNED_ACTIVITIES,
        "Combined EA Pareto Front: Prof. Conflicts vs. Unassigned",
        output_dir,
        "combined_ea_pareto_0v3.png"
    )
    print(f"Combined Pareto plot saved to {output_dir}/combined_ea_pareto_0v3.png")
    
    # Perform statistical analysis
    print("\n--- Performing Statistical Analysis ---")
    metrics = results["metrics"]
    
    # Find best algorithm for each metric
    best_algorithms = {}
    higher_better = ["hypervolume", "convergence_speed"]
    
    for metric in metrics[algorithm_names[0]]:
        if metric in ["pareto_size", "execution_time", "memory_usage"]:
            continue
            
        metric_values = {algo: metrics[algo].get(metric, 0) for algo in algorithm_names}
        
        if metric in higher_better:
            best_algo = max(metric_values, key=metric_values.get)
        else:
            best_algo = min(metric_values, key=metric_values.get)
            
        best_algorithms[metric] = best_algo
    
    # Add execution time
    execution_times = {algo: metrics[algo].get("execution_time", 0) for algo in algorithm_names}
    best_algorithms["execution_time"] = min(execution_times, key=execution_times.get)
    
    # Add convergence speed
    if "convergence_speed" in metrics[algorithm_names[0]]:
        conv_speeds = {algo: metrics[algo].get("convergence_speed", 0) for algo in algorithm_names}
        best_algorithms["convergence_speed"] = max(conv_speeds, key=conv_speeds.get)
    
    # Write statistical analysis to file
    with open(os.path.join(output_dir, "statistical_analysis.txt"), "w") as f:
        f.write("Best performing algorithm per metric:\n")
        for metric, algo in best_algorithms.items():
            qualifier = "higher is better" if metric in higher_better else "lower is better"
            f.write(f"  {metric.capitalize()} ({qualifier}): {algo}\n")
    
    print(f"Statistical analysis saved to {output_dir}/statistical_analysis.txt")
    
    # Generate research-quality visualizations with metrics
    print("\n--- Generating Research-Quality Visualizations with Metrics ---")
    
    # Research-quality Pareto front - Professor conflicts vs. Unassigned
    try:
        plot_pareto_front_2d(
            [list(results["pareto_fronts"][algo]) for algo in algorithm_names],
            algorithm_names,
            colors,
            markers,
            0, 3,
            PROF_CONFLICTS, UNASSIGNED_ACTIVITIES,
            "Research-Quality Pareto Front: EA Algorithms",
            output_dir,
            "research_pareto_0v3.png"
        )
        print(f"Research-quality Pareto front saved to {output_dir}/research_pareto_0v3.png")
    except Exception as e_plot1:
        print(f"Error generating research_pareto_0v3.png: {e_plot1} (Type: {type(e_plot1)})")

    # Line-only variant
    try:
        plot_pareto_front_2d(
            [list(results["pareto_fronts"][algo]) for algo in algorithm_names],
            algorithm_names,
            colors,
            markers,
            0, 3,
            PROF_CONFLICTS, UNASSIGNED_ACTIVITIES,
            "Pareto Front Lines: EA Algorithms",
            output_dir,
            "pareto_lines_0v3.png"
        )
        print(f"Lines-only variant saved to {output_dir}/pareto_lines_0v3.png")
    except Exception as e_plot2:
        print(f"Error generating pareto_lines_0v3.png: {e_plot2} (Type: {type(e_plot2)})")

    # Parallel coordinates plot
    try:
        pareto_fronts_for_plot = [list(results["pareto_fronts"][algo]) for algo in algorithm_names]
        print(f"DEBUG: visualize_ea_results: algorithm_names: {algorithm_names}")
        print(f"DEBUG: visualize_ea_results: colors: {colors}")
        print(f"DEBUG: visualize_ea_results: shapes of pareto_fronts_for_plot: {[np.array(pf).shape if pf else 'Empty' for pf in pareto_fronts_for_plot]}")

        plot_parallel_coordinates(
            [list(results["pareto_fronts"][algo]) for algo in algorithm_names],
            algorithm_names,
            colors,
            output_dir,
            "parallel_coordinates.png"
        )
        print(f"Parallel coordinates plot saved to {output_dir}/parallel_coordinates.png")
    except Exception as e_plot3:
        print(f"DEBUG ON ERROR: visualize_ea_results: algorithm_names: {algorithm_names}")
        print(f"DEBUG ON ERROR: visualize_ea_results: colors: {colors}")
        # pareto_fronts_for_plot might not be defined if error was in its creation, handle this
        try:
            print(f"DEBUG ON ERROR: visualize_ea_results: shapes of pareto_fronts_for_plot: {[np.array(pf).shape if pf else 'Empty' for pf in pareto_fronts_for_plot]}")
        except NameError:
            print("DEBUG ON ERROR: visualize_ea_results: pareto_fronts_for_plot was not defined.")

        print(f"Error generating parallel_coordinates.png: {e_plot3} (Type: {type(e_plot3)})")
    
    # Final comparison report
    print("\n--- Final Comparison ---")
    print("Algorithm execution times:")
    for algo in algorithm_names:
        print(f"{algo}: {metrics[algo]['execution_time']:.2f} seconds")
    
    print("\n--- Final Pareto Front Sizes ---")
    for algo in algorithm_names:
        print(f"{algo}: {metrics[algo]['pareto_size']}")
    
    print(f"\nPlots saved to '{output_dir}' directory:")
    print("1. Research-quality Pareto front: research_pareto_0v3.png")
    print("2. Clean Pareto front with reference line: pareto_lines_0v3.png")
    print("3. Parallel coordinates plot: parallel_coordinates.png")
    print("4. Simple objective comparison plots: 2 plots")


if __name__ == "__main__":
    # Check if results file exists
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize EA algorithm results")
    parser.add_argument("--results", type=str, default="ea_results.json", 
                        help="Path to results JSON file")
    parser.add_argument("--output_dir", type=str, default="ea_results",
                        help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Load results from file
    if os.path.exists(args.results):
        with open(args.results, "r") as f:
            results = json.load(f)
        
        visualize_ea_results(results, args.output_dir)
    else:
        print(f"Results file {args.results} not found.")
        sys.exit(1)
