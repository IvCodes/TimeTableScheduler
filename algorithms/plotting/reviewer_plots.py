"""
Specialized visualization module to address reviewer comments.
Generates plots specifically requested: Pareto fronts, learning curves, and comparative analyses.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import get_cmap

# Constants
PAPER_FIGURE_DPI = 600  # High DPI for publication-quality figures
FIGURE_SIZE = (10, 6)   # Standard figure size for paper

def load_result_file(file_path):
    """
    Load results from a JSON file and extract metrics data.
    
    Args:
        file_path: Path to the JSON result file
        
    Returns:
        Dictionary containing results data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if the metrics are nested within the 'metrics' key
        if 'metrics' in data:
            metrics_data = data['metrics']
            # Also include top-level metadata like algorithm, dataset, runtime
            metrics_data.update({k: v for k, v in data.items() if k != 'metrics'})
            return metrics_data
        else:
            # For older result files or different structure, look for metrics at root level
            # and inside the 'timetable' key as in our current structure
            metrics_data = {}
            # Check if metrics data is nested inside 'timetable'
            if 'timetable' in data and isinstance(data['timetable'], dict):
                for key in ['final_pareto_front', 'pareto_front_size', 'hypervolume', 'spacing', 'igd', 
                          'execution_time', 'solution_diversity']:
                    if key in data['timetable']:
                        metrics_data[key] = data['timetable'][key]
            
            # Also add top-level data
            metrics_data.update({k: v for k, v in data.items() if k != 'timetable' and isinstance(v, (dict, list, int, float, str))})
            return metrics_data
    except Exception as e:
        print(f"Error loading result file {file_path}: {e}")
        return None

def plot_pareto_front_comparison(result_files, output_dir='figures', dataset_name='4-room'):
    """
    Plot Pareto fronts for multi-objective genetic algorithms.
    
    Args:
        result_files: Dictionary of {algorithm_name: file_path} for GA results
        output_dir: Directory to save the output figure
        dataset_name: Name of the dataset (e.g., '4-room', '7-room')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=FIGURE_SIZE)
    
    # Style configuration
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', '*']
    
    legend_items = []
    
    for i, (algo_name, file_path) in enumerate(result_files.items()):
        data = load_result_file(file_path)
        if not data or 'final_pareto_front' not in data:
            print(f"No Pareto front data found in {file_path}")
            continue
            
        # Extract Pareto front points
        pareto_points = data['final_pareto_front']
        
        if not pareto_points:
            print(f"Empty Pareto front for {algo_name}")
            continue
            
        # Convert to numpy arrays for easier handling
        hard_violations = np.array([point[0] for point in pareto_points])
        soft_scores = np.array([point[1] for point in pareto_points])
        
        # Plot Pareto front
        scatter = plt.scatter(
            hard_violations, 
            soft_scores,
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=60,
            alpha=0.7,
            label=algo_name
        )
        legend_items.append(scatter)
    
    plt.title(f'Pareto Front Comparison - {dataset_name} Dataset', fontsize=14)
    plt.xlabel('Hard Constraint Violations', fontsize=12)
    plt.ylabel('Soft Constraint Score (lower is better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Ensure axis uses whole numbers for hard violations
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save figure
    plt.tight_layout()
    output_path = f"{output_dir}/pareto_front_{dataset_name}.png"
    plt.savefig(output_path, dpi=PAPER_FIGURE_DPI)
    print(f"Pareto front plot saved to {output_path}")
    plt.close()

def plot_learning_curves(result_files, output_dir='figures', dataset_name='4-room'):
    """
    Plot learning curves for reinforcement learning algorithms.
    
    Args:
        result_files: Dictionary of {algorithm_name: file_path} for RL results
        output_dir: Directory to save the output figure
        dataset_name: Name of the dataset (e.g., '4-room', '7-room')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Style configuration
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':', '-']
    
    for i, (algo_name, file_path) in enumerate(result_files.items()):
        data = load_result_file(file_path)
        if not data:
            continue
            
        # Extract episode/epoch data
        rewards = data.get('episode_rewards', [])
        violations = data.get('episode_violations', [])
        
        if not rewards and not violations:
            print(f"No learning curve data found for {algo_name}")
            continue
            
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        # Plot rewards
        if rewards:
            episodes = range(1, len(rewards) + 1)
            ax1.plot(episodes, rewards, linestyle=line_style, color=color, linewidth=2, label=algo_name)
        
        # Plot violations
        if violations:
            episodes = range(1, len(violations) + 1)
            ax2.plot(episodes, violations, linestyle=line_style, color=color, linewidth=2, label=algo_name)
    
    # Configure reward plot
    ax1.set_title('Reward vs. Episode', fontsize=14)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Configure violations plot
    ax2.set_title('Hard Violations vs. Episode', fontsize=14)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Hard Constraint Violations', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Ensure axis uses whole numbers for episodes
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save figure
    plt.tight_layout()
    output_path = f"{output_dir}/learning_curves_{dataset_name}.png"
    plt.savefig(output_path, dpi=PAPER_FIGURE_DPI)
    print(f"Learning curves plot saved to {output_path}")
    plt.close()

def plot_convergence_comparison(result_files, output_dir='figures', dataset_name='4-room', is_ga=True):
    """
    Plot convergence comparison for multiple algorithms.
    
    Args:
        result_files: Dictionary of {algorithm_name: file_path} for algorithm results
        output_dir: Directory to save the output figures
        dataset_name: Name of the dataset (e.g., '4-room', '7-room')
        is_ga: Flag indicating if the algorithms are GA-based (affects metrics extraction)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create two figures, one for hard violations and one for soft scores
    fig_hard, ax_hard = plt.subplots(figsize=FIGURE_SIZE)
    fig_soft, ax_soft = plt.subplots(figsize=FIGURE_SIZE)
    
    # Style configuration
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    for i, (algo_name, file_path) in enumerate(result_files.items()):
        data = load_result_file(file_path)
        if not data:
            continue
            
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        # Extract convergence data based on algorithm type
        if is_ga:
            hard_violations = data.get('best_hard_violations', [])
            soft_scores = data.get('best_soft_score', [])
            x_label = 'Generation'
        else:  # For RL
            hard_violations = data.get('episode_violations', [])
            soft_scores = data.get('episode_soft_scores', [])
            x_label = 'Episode/Iteration'
        
        # Plot hard violations
        if hard_violations:
            x = range(1, len(hard_violations) + 1)
            ax_hard.plot(x, hard_violations, linestyle=line_style, color=color, linewidth=2, label=algo_name)
        
        # Plot soft scores
        if soft_scores:
            x = range(1, len(soft_scores) + 1)
            ax_soft.plot(x, soft_scores, linestyle=line_style, color=color, linewidth=2, label=algo_name)
    
    # Configure hard violations plot
    ax_hard.set_title(f'Hard Constraint Violations Convergence - {dataset_name}', fontsize=14)
    ax_hard.set_xlabel(x_label, fontsize=12)
    ax_hard.set_ylabel('Hard Constraint Violations', fontsize=12)
    ax_hard.grid(True, linestyle='--', alpha=0.7)
    ax_hard.legend()
    
    # Configure soft scores plot
    ax_soft.set_title(f'Soft Constraint Score Convergence - {dataset_name}', fontsize=14)
    ax_soft.set_xlabel(x_label, fontsize=12)
    ax_soft.set_ylabel('Soft Constraint Score (lower is better)', fontsize=12)
    ax_soft.grid(True, linestyle='--', alpha=0.7)
    ax_soft.legend()
    
    # Save figures
    fig_hard.tight_layout()
    fig_soft.tight_layout()
    
    hard_output_path = f"{output_dir}/hard_violations_convergence_{dataset_name}.png"
    soft_output_path = f"{output_dir}/soft_score_convergence_{dataset_name}.png"
    
    fig_hard.savefig(hard_output_path, dpi=PAPER_FIGURE_DPI)
    fig_soft.savefig(soft_output_path, dpi=PAPER_FIGURE_DPI)
    
    print(f"Hard violations convergence plot saved to {hard_output_path}")
    print(f"Soft score convergence plot saved to {soft_output_path}")
    
    plt.close(fig_hard)
    plt.close(fig_soft)

def plot_comparative_analysis(four_room_results, seven_room_results, output_dir='figures'):
    """
    Plot comparative analysis between 4-room and 7-room datasets across algorithms.
    
    Args:
        four_room_results: Dictionary of {algorithm_name: data} for 4-room results
        seven_room_results: Dictionary of {algorithm_name: data} for 7-room results
        output_dir: Directory to save the output figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract algorithm names that are common to both datasets
    common_algos = set(four_room_results.keys()) & set(seven_room_results.keys())
    
    if not common_algos:
        print("No common algorithms found between 4-room and 7-room results")
        return
    
    # Prepare data for plotting
    algo_names = list(common_algos)
    hard_violations_4 = []
    hard_violations_7 = []
    soft_scores_4 = []
    soft_scores_7 = []
    
    for algo in algo_names:
        # Extract final hard violations and soft scores
        data_4 = four_room_results.get(algo, {})
        data_7 = seven_room_results.get(algo, {})
        
        # For hard violations, use the final best value
        violations_4 = data_4.get('best_hard_violations', [])
        violations_7 = data_7.get('best_hard_violations', [])
        
        hard_violations_4.append(violations_4[-1] if violations_4 else 0)
        hard_violations_7.append(violations_7[-1] if violations_7 else 0)
        
        # For soft scores, use the final best value
        scores_4 = data_4.get('best_soft_score', [])
        scores_7 = data_7.get('best_soft_score', [])
        
        soft_scores_4.append(scores_4[-1] if scores_4 else 0)
        soft_scores_7.append(scores_7[-1] if scores_7 else 0)
    
    # Create bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set up indices for bar positioning
    x = np.arange(len(algo_names))
    width = 0.35
    
    # Plot hard violations comparison
    bars1 = ax1.bar(x - width/2, hard_violations_4, width, label='4-Room Dataset', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, hard_violations_7, width, label='7-Room Dataset', color='#ff7f0e')
    
    ax1.set_title('Hard Constraint Violations Comparison', fontsize=14)
    ax1.set_ylabel('Hard Constraint Violations', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algo_names, rotation=45, ha='right')
    ax1.legend()
    
    # Add value labels on the bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Plot soft scores comparison
    bars3 = ax2.bar(x - width/2, soft_scores_4, width, label='4-Room Dataset', color='#1f77b4')
    bars4 = ax2.bar(x + width/2, soft_scores_7, width, label='7-Room Dataset', color='#ff7f0e')
    
    ax2.set_title('Soft Constraint Score Comparison', fontsize=14)
    ax2.set_ylabel('Soft Constraint Score (lower is better)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algo_names, rotation=45, ha='right')
    ax2.legend()
    
    # Add value labels on the bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = f"{output_dir}/4room_vs_7room_comparison.png"
    plt.savefig(output_path, dpi=PAPER_FIGURE_DPI)
    print(f"Comparative analysis plot saved to {output_path}")
    plt.close()

def generate_all_paper_plots(four_room_dir, seven_room_dir=None, output_dir='paper_figures'):
    """
    Generate all plots required for the paper.
    
    Args:
        four_room_dir: Directory containing 4-room experiment results
        seven_room_dir: Directory containing 7-room experiment results (if available)
        output_dir: Directory to save the output figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect GA result files from 4-room dataset
    ga_results_4room = {}
    for algo in ['nsga2', 'moead', 'spea2']:
        path = os.path.join(four_room_dir, f'GA/results_{algo}_sliit_computing_dataset.json')
        if os.path.exists(path):
            ga_results_4room[algo.upper()] = path
    
    # Collect RL result files from 4-room dataset
    rl_results_4room = {}
    for algo in ['dqn', 'sarsa', 'qlearning']:
        path = os.path.join(four_room_dir, f'RL/results_{algo}_sliit_computing_dataset.json')
        if os.path.exists(path):
            rl_results_4room[algo.upper()] = path
    
    # Generate Pareto front for GA algorithms (4-room)
    if ga_results_4room:
        plot_pareto_front_comparison(ga_results_4room, output_dir, dataset_name='4-room')
    
    # Generate learning curves for RL algorithms (4-room)
    if rl_results_4room:
        plot_learning_curves(rl_results_4room, output_dir, dataset_name='4-room')
    
    # Generate convergence plots for GA algorithms (4-room)
    if ga_results_4room:
        plot_convergence_comparison(ga_results_4room, output_dir, dataset_name='4-room', is_ga=True)
    
    # Generate convergence plots for RL algorithms (4-room)
    if rl_results_4room:
        plot_convergence_comparison(rl_results_4room, output_dir, dataset_name='4-room', is_ga=False)
    
    # If 7-room dataset results are available, generate those plots too
    if seven_room_dir:
        # Collect GA result files from 7-room dataset
        ga_results_7room = {}
        for algo in ['nsga2', 'moead', 'spea2']:
            path = os.path.join(seven_room_dir, f'GA/results_{algo}_sliit_computing_dataset_7.json')
            if os.path.exists(path):
                ga_results_7room[algo.upper()] = path
        
        # Collect RL result files from 7-room dataset
        rl_results_7room = {}
        for algo in ['dqn', 'sarsa', 'qlearning']:
            path = os.path.join(seven_room_dir, f'RL/results_{algo}_sliit_computing_dataset_7.json')
            if os.path.exists(path):
                rl_results_7room[algo.upper()] = path
        
        # Generate 7-room dataset plots
        if ga_results_7room:
            plot_pareto_front_comparison(ga_results_7room, output_dir, dataset_name='7-room')
            plot_convergence_comparison(ga_results_7room, output_dir, dataset_name='7-room', is_ga=True)
        
        if rl_results_7room:
            plot_learning_curves(rl_results_7room, output_dir, dataset_name='7-room')
            plot_convergence_comparison(rl_results_7room, output_dir, dataset_name='7-room', is_ga=False)
        
        # Generate comparative analysis between 4-room and 7-room datasets
        if ga_results_4room and ga_results_7room:
            # Load result data
            four_room_data = {algo: load_result_file(path) for algo, path in ga_results_4room.items()}
            seven_room_data = {algo: load_result_file(path) for algo, path in ga_results_7room.items()}
            
            # Generate comparison plots
            plot_comparative_analysis(four_room_data, seven_room_data, output_dir)

if __name__ == "__main__":
    # Example usage (uncomment when running as script)
    # generate_all_paper_plots(
    #     four_room_dir="c:/Users/Easara_200005/Desktop/Projects/Research_conference/TimeTableScheduler/output",
    #     seven_room_dir="c:/Users/Easara_200005/Desktop/Projects/Research_conference/TimeTableScheduler/output_7room",
    #     output_dir="c:/Users/Easara_200005/Desktop/Projects/Research_conference/TimeTableScheduler/paper_figures"
    # )
    pass
