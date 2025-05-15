"""
Common visualization module for timetable scheduling algorithms.
Provides standardized plotting functions for comparing algorithms.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
from mpl_toolkits.mplot3d import Axes3D

from common.metrics import OBJECTIVE_LABELS


def create_output_dir(base_dir="results", algorithm_name=None):
    """
    Create and return output directory path.
    
    Args:
        base_dir: Base directory for results
        algorithm_name: Name of algorithm (optional)
        
    Returns:
        str: Path to output directory
    """
    if algorithm_name:
        output_dir = os.path.join(base_dir, algorithm_name)
    else:
        output_dir = base_dir
        
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_convergence(history, title, output_dir, filename=None):
    """
    Plot convergence history of an optimization algorithm.
    
    Args:
        history: List of best fitness values per iteration
        title: Plot title
        output_dir: Directory to save the plot
        filename: Filename for the plot (optional)
        
    Returns:
        str: Path to saved plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if not filename:
        filename = f"convergence_{title.lower().replace(' ', '_')}.png"
        
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_pareto_front_2d(pareto_fronts, algorithm_names, colors, markers, 
                        obj_x_index, obj_y_index, obj_x_label, obj_y_label,
                        title, output_dir, filename=None):
    """
    Plot 2D Pareto front for comparing multiple algorithms.
    
    Args:
        pareto_fronts: List of pareto fronts (each a list of solutions)
        algorithm_names: List of algorithm names
        colors: List of colors for each algorithm
        markers: List of markers for each algorithm
        obj_x_index: Index of x-axis objective
        obj_y_index: Index of y-axis objective
        obj_x_label: Label for x-axis
        obj_y_label: Label for y-axis
        title: Plot title
        output_dir: Directory to save the plot
        filename: Filename for the plot (optional)
        
    Returns:
        str: Path to saved plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot reference lines
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    # Plot each Pareto front
    for i, (front, name, color, marker) in enumerate(zip(pareto_fronts, algorithm_names, colors, markers)):
        # Extract the two objectives we want to plot
        if front:
            x_values = [solution[obj_x_index] for solution in front]
            y_values = [solution[obj_y_index] for solution in front]
            
            # Update min/max values
            min_x = min(min_x, min(x_values))
            max_x = max(max_x, max(x_values))
            min_y = min(min_y, min(y_values))
            max_y = max(max_y, max(y_values))
            
            # Plot the front
            plt.scatter(x_values, y_values, c=color, marker=marker, s=80, 
                       label=name, alpha=0.8, edgecolor='w')
    
    # Add ideal point area
    plt.axhspan(0, min_y, 0, min_x, alpha=0.1, color='green', label='Ideal Region')
    
    # Add reference line
    if min_x < max_x and min_y < max_y:
        plt.plot([min_x, max_x], [min_y, min_y], 'k--', alpha=0.3)
        plt.plot([min_x, min_x], [min_y, max_y], 'k--', alpha=0.3)
    
    plt.xlabel(obj_x_label, fontsize=12)
    plt.ylabel(obj_y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Algorithm', fontsize=10, title_fontsize=12)
    
    if not filename:
        safe_title = title.lower().replace(' ', '_').replace(':', '').replace(',', '')
        filename = f"pareto_2d_{safe_title}.png"
        
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_pareto_front_3d(pareto_fronts, algorithm_names, colors, markers,
                         obj_x_index, obj_y_index, obj_z_index,
                         obj_x_label, obj_y_label, obj_z_label,
                         title, output_dir, filename=None):
    """
    Plot 3D Pareto front for comparing multiple algorithms.
    
    Args:
        pareto_fronts: List of pareto fronts (each a list of solutions)
        algorithm_names: List of algorithm names
        colors: List of colors for each algorithm
        markers: List of markers for each algorithm
        obj_x_index: Index of x-axis objective
        obj_y_index: Index of y-axis objective
        obj_z_index: Index of z-axis objective
        obj_x_label: Label for x-axis
        obj_y_label: Label for y-axis
        obj_z_label: Label for z-axis
        title: Plot title
        output_dir: Directory to save the plot
        filename: Filename for the plot (optional)
        
    Returns:
        str: Path to saved plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each Pareto front
    for front, name, color, marker in zip(pareto_fronts, algorithm_names, colors, markers):
        if front:
            x_values = [solution[obj_x_index] for solution in front]
            y_values = [solution[obj_y_index] for solution in front]
            z_values = [solution[obj_z_index] for solution in front]
            
            ax.scatter(x_values, y_values, z_values, c=color, marker=marker, s=70,
                      label=name, alpha=0.8, edgecolor='w')
    
    ax.set_xlabel(obj_x_label, fontsize=12)
    ax.set_ylabel(obj_y_label, fontsize=12)
    ax.set_zlabel(obj_z_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title='Algorithm', fontsize=10, title_fontsize=12)
    
    if not filename:
        safe_title = title.lower().replace(' ', '_').replace(':', '').replace(',', '')
        filename = f"pareto_3d_{safe_title}.png"
        
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_parallel_coordinates(pareto_fronts, algorithm_names, colors, output_dir, filename=None):
    print(f"DEBUG: plot_parallel_coordinates: received algorithm_names: {algorithm_names}")
    print(f"DEBUG: plot_parallel_coordinates: received colors: {colors}")
    print(f"DEBUG: plot_parallel_coordinates: shapes of received pareto_fronts: {[np.array(pf).shape if pf else 'Empty' for pf in pareto_fronts]}")
    """
    Plot parallel coordinates visualization of Pareto fronts.
    
    Args:
        pareto_fronts: List of pareto fronts (each a list of solutions)
        algorithm_names: List of algorithm names
        colors: List of colors for each algorithm
        output_dir: Directory to save the plot
        filename: Filename for the plot (optional)
        
    Returns:
        str: Path to saved plot
    """
    _ = plt.figure(figsize=(14, 8)) # Assign to _ as fig object not used directly
    
    # Prepare data for the plot
    all_solutions = []
    plot_specific_labels = OBJECTIVE_LABELS
    num_objectives_for_plot = len(plot_specific_labels)

    for i, front in enumerate(pareto_fronts): 
        for solution_objectives in front: 
            if isinstance(solution_objectives, dict):
                continue
            
            objectives_to_add = list(solution_objectives[:num_objectives_for_plot])
            while len(objectives_to_add) < num_objectives_for_plot:
                objectives_to_add.append(np.nan)
            all_solutions.append(objectives_to_add + [i])
    
    print(f"DEBUG: plot_parallel_coordinates: all_solutions count: {len(all_solutions)}")
    if not all_solutions:
        print(f"DEBUG: plot_parallel_coordinates: No solutions to plot in {filename}. Skipping graph generation.")
        # We might still want to save an empty plot or a placeholder if required by calling function's expectation of a filepath return
        # For now, just returning early if no data makes sense.
        return None # Or handle as appropriate

    # Convert to DataFrame
    print(f"DEBUG: plot_parallel_coordinates: plot_specific_labels for df columns: {plot_specific_labels}")
    df_columns = plot_specific_labels + ['Algorithm']
    df = pd.DataFrame(all_solutions, columns=df_columns)
    print(f"DEBUG: plot_parallel_coordinates: df.head():\n{df.head()}")
    print(f"DEBUG: plot_parallel_coordinates: df.shape: {df.shape}")
    print(f"DEBUG: plot_parallel_coordinates: df.columns set to: {df.columns.tolist()}")
    
    # Plot
    plot_colors = [colors[i % len(colors)] for i in range(len(algorithm_names))]
    print(f"DEBUG: plot_parallel_coordinates: plot_colors for pd.plotting: {plot_colors}")

    try:
        pd.plotting.parallel_coordinates(df, 'Algorithm', color=plot_colors, alpha=0.5)
    except KeyError as e_pd_plotting:
        print(f"DEBUG: plot_parallel_coordinates: KeyError INSIDE pd.plotting.parallel_coordinates: {e_pd_plotting}")
        print(f"DEBUG: plot_parallel_coordinates: df.columns at error: {df.columns.tolist()}")
        if 'Algorithm' in df:
            print(f"DEBUG: plot_parallel_coordinates: 'Algorithm' column unique values: {df['Algorithm'].unique()}")
        else:
            print("DEBUG: plot_parallel_coordinates: 'Algorithm' column NOT FOUND in df at error point.")
        raise # re-raise the error
    except Exception as e_general_plotting:
        print(f"DEBUG: plot_parallel_coordinates: Non-KeyError Exception INSIDE pd.plotting.parallel_coordinates: {e_general_plotting} (Type: {type(e_general_plotting)})")
        raise
    
    plt.title('Parallel Coordinates Plot of Pareto Fronts', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=30, ha='right')
    
    # This check is now done earlier, before DataFrame creation.
    # if not all_solutions:
    #     print(f"No solutions to plot in {filename}. Skipping.")
    #     return
    
    if not filename:
        filename = "parallel_coordinates.png"
        
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_metrics_comparison(algorithm_names, metrics_dict, metric_names, output_dir, filename=None):
    """
    Plot comparison of performance metrics across algorithms.
    
    Args:
        algorithm_names: List of algorithm names
        metrics_dict: Dictionary mapping algorithm names to metric dictionaries
        metric_names: List of metric names to plot
        output_dir: Directory to save the plot
        filename: Filename for the plot (optional)
        
    Returns:
        str: Path to saved plot
    """
    n_metrics = len(metric_names)
    n_algos = len(algorithm_names)
    
    _, axes = plt.subplots(n_metrics, 1, figsize=(10, 3*n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]
    
    # Create bar plots for each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        
        # Extract values for this metric from each algorithm
        values = []
        for algo in algorithm_names:
            value = metrics_dict.get(algo, {}).get(metric_name, 0)
            values.append(value)
        
        # Create color map based on metric (some higher is better, some lower is better)
        higher_is_better = metric_name in ["hypervolume", "convergence_speed", "resource_efficiency"]
        
        # Sort algorithms by performance
        sorted_indices = np.argsort(values)
        if higher_is_better:
            sorted_indices = sorted_indices[::-1]  # Reverse for higher is better
            
        sorted_algos = [algorithm_names[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        # Create colormap (green = best, yellow = mid, red = worst)
        cmap = cm.get_cmap('RdYlGn')
        colors = cmap(np.linspace(0, 1, n_algos))
        
        # Plot
        bars = ax.barh(sorted_algos, sorted_values, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01  # Offset for label position
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                   va='center', fontsize=8)
        
        # Format
        ax.set_title(f"{metric_name.replace('_', ' ').title()}", fontsize=12)
        ax.set_xlim(0, max(values) * 1.15)  # Add margin for labels
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    plt.tight_layout()
    
    if not filename:
        filename = "metrics_comparison.png"
        
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_learning_curve(episode_rewards, window_size=10, title="Learning Curve", 
                       output_dir="results", filename=None):
    """
    Plot learning curve for RL algorithms.
    
    Args:
        episode_rewards: List of rewards for each episode
        window_size: Window size for moving average (smoothing)
        title: Plot title
        output_dir: Directory to save the plot
        filename: Filename for the plot (optional)
        
    Returns:
        str: Path to saved plot
    """
    plt.figure(figsize=(10, 6))
    
    # Raw rewards
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Smoothed rewards (moving average)
    if len(episode_rewards) >= window_size:
        smoothed_rewards = []
        for i in range(len(episode_rewards) - window_size + 1):
            window_reward = np.mean(episode_rewards[i:i+window_size])
            smoothed_rewards.append(window_reward)
        
        # Plot smoothed rewards
        plt.plot(range(window_size-1, len(episode_rewards)), 
                smoothed_rewards, linewidth=2, color='red', 
                label=f'Moving Average (window={window_size})')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if not filename:
        safe_title = title.lower().replace(' ', '_').replace(':', '').replace(',', '')
        filename = f"learning_curve_{safe_title}.png"
        
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_constraint_breakdown(constraint_values, algorithm_names, constraint_types,
                             is_hard_constraint, output_dir, filename=None):
    """
    Plot breakdown of constraint violations.
    
    Args:
        constraint_values: Dictionary mapping (algorithm, constraint) to violation count
        algorithm_names: List of algorithm names
        constraint_types: List of constraint types
        is_hard_constraint: Boolean indicating if these are hard constraints
        output_dir: Directory to save the plot
        filename: Filename for the plot (optional)
        
    Returns:
        str: Path to saved plot
    """
    # Prepare data for stacked bar chart
    n_algos = len(algorithm_names)
    n_constraints = len(constraint_types)
    
    data = np.zeros((n_algos, n_constraints))
    for i, algo in enumerate(algorithm_names):
        for j, constraint in enumerate(constraint_types):
            data[i, j] = constraint_values.get((algo, constraint), 0)
    
    # Create colormap
    cmap = cm.get_cmap('tab20', n_constraints)
    colors = [cmap(i) for i in range(n_constraints)]
    
    # Plot
    plt.figure(figsize=(12, 7))
    
    bottoms = np.zeros(n_algos)
    for j in range(n_constraints):
        plt.bar(algorithm_names, data[:, j], bottom=bottoms, 
               label=constraint_types[j], color=colors[j])
        bottoms += data[:, j]
    
    # Add total labels on top of each bar
    for i, algo in enumerate(algorithm_names):
        total = sum(data[i, :])
        plt.text(i, total + max(bottoms)*0.02, f'Total: {total:.1f}', 
                ha='center', va='bottom', fontsize=10)
    
    constraint_type = "Hard" if is_hard_constraint else "Soft"
    plt.title(f"{constraint_type} Constraint Violations Breakdown", fontsize=14)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Violation Count', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Constraint Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    if not filename:
        constraint_str = "hard" if is_hard_constraint else "soft"
        filename = f"constraint_breakdown_{constraint_str}.png"
        
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_resource_utilization(algorithm_names, execution_times, peak_memories,
                             output_dir, filename=None):
    """
    Plot resource utilization comparison.
    
    Args:
        algorithm_names: List of algorithm names
        execution_times: List of execution times (seconds)
        peak_memories: List of peak memory usages (MB)
        output_dir: Directory to save the plot
        filename: Filename for the plot (optional)
        
    Returns:
        str: Path to saved plot
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot execution times
    ax1.bar(algorithm_names, execution_times, color='skyblue')
    ax1.set_title('Execution Time Comparison', fontsize=14)
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_xticks(range(len(algorithm_names)))
    ax1.set_xticklabels(algorithm_names, rotation=15, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels
    for i, v in enumerate(execution_times):
        ax1.text(i, v + max(execution_times)*0.02, f'{v:.2f}s', 
                ha='center', va='bottom', fontsize=10)
    
    # Plot peak memory usage
    ax2.bar(algorithm_names, peak_memories, color='lightgreen')
    ax2.set_title('Peak Memory Usage Comparison', fontsize=14)
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_xticks(range(len(algorithm_names)))
    ax2.set_xticklabels(algorithm_names, rotation=15, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels
    for i, v in enumerate(peak_memories):
        ax2.text(i, v + max(peak_memories)*0.02, f'{v:.1f} MB', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if not filename:
        filename = "resource_utilization.png"
        
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath
