"""
TimeTable Scheduler - Azure Fabric Notebook Script (Part 5: Experiments and Visualization)
This script is formatted for easy conversion to a Jupyter notebook.
Each cell is marked with '# CELL: {description}' comments.
"""

# CELL: Import Libraries and Components
import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Set paths for importing from other notebook parts
sys.path.append(".")

# Import from previous notebook parts
# In Jupyter, these would already be loaded from previous cells

# Import data loading functions
from timetable_fabric_notebook_part1_setup import (
    load_data_from_local, load_data_from_adls, process_data, 
    display_dataset_info, Activity, Group, Space, Lecturer
)

# Import evaluation and visualization functions
from timetable_fabric_notebook_part2_evaluator import (
    evaluate_schedule, multi_objective_evaluator, 
    visualize_schedule, visualize_pareto_front, visualize_convergence,
    parse_ga_schedule, parse_rl_schedule
)

# Import GA algorithms
from timetable_fabric_notebook_part3_ga import (
    nsga2, spea2, moead
)

# Import RL algorithms
from timetable_fabric_notebook_part4_rl_base import (
    q_learning, initialize_state, get_available_actions, 
    apply_action, sarsa_state_to_schedule
)
from timetable_fabric_notebook_part4_rl_sarsa import sarsa
from timetable_fabric_notebook_part4_rl_dqn import dqn_scheduling

# CELL: Configuration Parameters
# Define experiment parameters
GA_PARAMS = {
    '4room': {
        'pop_size': 150,
        'generations': 150,
        'crossover_rate': 0.9,
        'mutation_rate': 0.1,
        'archive_size': 150,  # For SPEA2
        'neighborhood_size': 20,  # For MOEA/D
    },
    '7room': {
        'pop_size': 200,
        'generations': 250,
        'crossover_rate': 0.9,
        'mutation_rate': 0.1,
        'archive_size': 200,  # For SPEA2
        'neighborhood_size': 25,  # For MOEA/D
    }
}

RL_PARAMS = {
    '4room': {
        'alpha': 0.1,
        'gamma': 0.6,
        'epsilon': 0.1,
        'n_episodes': 800,
        'epsilon_min': 0.01,  # For DQN
        'epsilon_decay': 0.995,  # For DQN
        'batch_size': 32,  # For DQN
    },
    '7room': {
        'alpha': 0.1,
        'gamma': 0.6,
        'epsilon': 0.1,
        'n_episodes': 1200,
        'epsilon_min': 0.01,  # For DQN
        'epsilon_decay': 0.995,  # For DQN
        'batch_size': 32,  # For DQN
    }
}

# Define dataset paths
DATASETS = {
    '4room': 'sliit_computing_dataset.json',
    '7room': 'sliit_computing_dataset_7.json'
}

# Define experiment settings
EXPERIMENTS = {
    'ga': ['nsga2', 'spea2', 'moead'],
    'rl': ['q_learning', 'sarsa', 'dqn']
}

# CELL: Create Experiment Results Directory
def create_results_dir():
    """Create directory for experiment results"""
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    
    try:
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")
        return results_dir
    except Exception as e:
        print(f"Error creating results directory: {str(e)}")
        return "results"  # Fallback name

# CELL: Run GA Experiment
def run_ga_experiment(algorithm, dataset_type, data_tuple, results_dir):
    """
    Run a Genetic Algorithm experiment.
    
    Args:
        algorithm: Algorithm to run ('nsga2', 'spea2', 'moead')
        dataset_type: Type of dataset ('4room', '7room')
        data_tuple: Tuple containing all the data structures
        results_dir: Directory to save results
        
    Returns:
        Dict: Results of the experiment
    """
    print(f"\n=== Running {algorithm.upper()} on {dataset_type} dataset ===")
    
    # Unpack data tuple
    (
        activities_dict, groups_dict, spaces_dict, lecturers_dict,
        activities_list, groups_list, spaces_list, lecturers_list,
        activity_types, timeslots_list, days_list, periods_list, slots
    ) = data_tuple
    
    # Get parameters for this experiment
    params = GA_PARAMS[dataset_type]
    
    # Create experiment directory
    exp_dir = f"{results_dir}/{dataset_type}/{algorithm}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    # Run algorithm
    if algorithm == 'nsga2':
        population, fitness = nsga2(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            pop_size=params['pop_size'], 
            generations=params['generations'],
            crossover_rate=params['crossover_rate'],
            mutation_rate=params['mutation_rate']
        )
    elif algorithm == 'spea2':
        population, fitness = spea2(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            pop_size=params['pop_size'], 
            generations=params['generations'],
            archive_size=params['archive_size'],
            crossover_rate=params['crossover_rate'],
            mutation_rate=params['mutation_rate']
        )
    elif algorithm == 'moead':
        population, fitness = moead(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            pop_size=params['pop_size'], 
            generations=params['generations'],
            neighborhood_size=params['neighborhood_size'],
            crossover_rate=params['crossover_rate'],
            mutation_rate=params['mutation_rate']
        )
    else:
        raise ValueError(f"Unknown GA algorithm: {algorithm}")
    
    # End timer
    duration = time.time() - start_time
    
    # Find best individual
    best_idx = min(range(len(fitness)), key=lambda i: fitness[i][-1])
    best_individual = population[best_idx]
    best_fitness = fitness[best_idx]
    
    # Create best schedule
    best_schedule = parse_ga_schedule(best_individual, activities_dict, spaces_dict, slots)
    
    # Evaluate best schedule
    best_reward, hard_violations, soft_violations = evaluate_schedule(
        best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
    )
    
    # Create result object
    result = {
        'algorithm': algorithm,
        'dataset': dataset_type,
        'params': params,
        'duration': duration,
        'population': population,
        'fitness': fitness,
        'best_individual': best_individual,
        'best_fitness': best_fitness,
        'best_schedule': best_schedule,
        'best_reward': best_reward,
        'hard_violations': hard_violations,
        'soft_violations': soft_violations
    }
    
    # Save result
    with open(f"{exp_dir}/result.pkl", 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Experiment completed in {duration:.2f} seconds")
    print(f"Best fitness: {best_fitness[-1]}")
    print(f"Hard violations: {hard_violations}, Soft violations: {soft_violations}")
    
    # Create visualizations
    
    # Pareto front
    fig = visualize_pareto_front(
        fitness, 
        metrics_names=[
            'Space Capacity Violations', 
            'Lecturer/Group Clashes', 
            'Space Clashes', 
            'Soft Constraint Violations', 
            'Total Fitness'
        ],
        title=f"{algorithm.upper()} Pareto Front ({dataset_type} Dataset)"
    )
    fig.savefig(f"{exp_dir}/pareto_front.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Best schedule
    fig = visualize_schedule(
        best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
        title=f"Best Schedule - {algorithm.upper()} ({dataset_type} Dataset)"
    )
    fig.savefig(f"{exp_dir}/best_schedule.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return result

# CELL: Run RL Experiment
def run_rl_experiment(algorithm, dataset_type, data_tuple, results_dir):
    """
    Run a Reinforcement Learning experiment.
    
    Args:
        algorithm: Algorithm to run ('q_learning', 'sarsa', 'dqn')
        dataset_type: Type of dataset ('4room', '7room')
        data_tuple: Tuple containing all the data structures
        results_dir: Directory to save results
        
    Returns:
        Dict: Results of the experiment
    """
    print(f"\n=== Running {algorithm.upper()} on {dataset_type} dataset ===")
    
    # Unpack data tuple
    (
        activities_dict, groups_dict, spaces_dict, lecturers_dict,
        activities_list, groups_list, spaces_list, lecturers_list,
        activity_types, timeslots_list, days_list, periods_list, slots
    ) = data_tuple
    
    # Get parameters for this experiment
    params = RL_PARAMS[dataset_type]
    
    # Create experiment directory
    exp_dir = f"{results_dir}/{dataset_type}/{algorithm}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    # Run algorithm
    if algorithm == 'q_learning':
        history, q_table = q_learning(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=params['alpha'],
            gamma=params['gamma'],
            epsilon=params['epsilon'],
            n_episodes=params['n_episodes']
        )
    elif algorithm == 'sarsa':
        history, q_table = sarsa(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=params['alpha'],
            gamma=params['gamma'],
            epsilon=params['epsilon'],
            n_episodes=params['n_episodes']
        )
    elif algorithm == 'dqn':
        history, model = dqn_scheduling(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            gamma=params['gamma'],
            epsilon=params['epsilon'],
            epsilon_min=params['epsilon_min'],
            epsilon_decay=params['epsilon_decay'],
            batch_size=params['batch_size'],
            n_episodes=params['n_episodes']
        )
    else:
        raise ValueError(f"Unknown RL algorithm: {algorithm}")
    
    # End timer
    duration = time.time() - start_time
    
    # Create result object
    result = {
        'algorithm': algorithm,
        'dataset': dataset_type,
        'params': params,
        'duration': duration,
        'history': history,
        'best_schedule': history['best_schedule'],
        'best_reward': history['best_reward']
    }
    
    # Add algorithm-specific results
    if algorithm in ['q_learning', 'sarsa']:
        result['q_table'] = q_table
    elif algorithm == 'dqn':
        # Save model if TensorFlow was available
        try:
            model.save(f"{exp_dir}/dqn_model")
            result['model_saved'] = True
        except:
            result['model_saved'] = False
    
    # Save result
    with open(f"{exp_dir}/result.pkl", 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Experiment completed in {duration:.2f} seconds")
    print(f"Best reward: {history['best_reward']}")
    
    # Create visualizations
    
    # Learning curve
    fig = visualize_convergence(
        history,
        title=f"{algorithm.upper()} Learning Curve ({dataset_type} Dataset)"
    )
    fig.savefig(f"{exp_dir}/learning_curve.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Best schedule
    fig = visualize_schedule(
        history['best_schedule'], activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
        title=f"Best Schedule - {algorithm.upper()} ({dataset_type} Dataset)"
    )
    fig.savefig(f"{exp_dir}/best_schedule.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return result

# CELL: Run All Experiments
def run_all_experiments(results_dir=None):
    """
    Run all experiments.
    
    Args:
        results_dir: Directory to save results
    """
    if results_dir is None:
        results_dir = create_results_dir()
    
    # Results storage
    results = {}
    
    # For each dataset
    for dataset_type, dataset_file in DATASETS.items():
        results[dataset_type] = {}
        
        print(f"\n===== Running experiments for {dataset_type} dataset =====")
        
        # Load dataset
        try:
            # Look in current directory and Dataset directory
            for path in [dataset_file, f"Dataset/{dataset_file}", f"data/{dataset_file}"]:
                try:
                    data_tuple = load_data_from_local(path)
                    display_dataset_info(data_tuple)
                    break
                except:
                    continue
            else:
                raise FileNotFoundError(f"Dataset not found: {dataset_file}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            continue
        
        # Run GA experiments
        for algorithm in EXPERIMENTS['ga']:
            try:
                result = run_ga_experiment(algorithm, dataset_type, data_tuple, results_dir)
                results[dataset_type][algorithm] = result
            except Exception as e:
                print(f"Error running {algorithm} on {dataset_type}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Run RL experiments
        for algorithm in EXPERIMENTS['rl']:
            try:
                result = run_rl_experiment(algorithm, dataset_type, data_tuple, results_dir)
                results[dataset_type][algorithm] = result
            except Exception as e:
                print(f"Error running {algorithm} on {dataset_type}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Save overall results
    with open(f"{results_dir}/all_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return results, results_dir

# CELL: Comparative Analysis
def comparative_analysis(results, results_dir):
    """
    Perform comparative analysis of all algorithms.
    
    Args:
        results: Results from all experiments
        results_dir: Directory to save analysis
    """
    print("\n===== Performing Comparative Analysis =====")
    
    # Comparative analysis for each dataset
    for dataset_type in results.keys():
        print(f"\n--- Analysis for {dataset_type} dataset ---")
        
        # Create dataset analysis directory
        analysis_dir = f"{results_dir}/{dataset_type}/analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Gather statistics
        stats = {
            'algorithm': [],
            'duration': [],
            'hard_violations': [],
            'soft_violations': [],
            'final_fitness': []
        }
        
        for alg_type in ['ga', 'rl']:
            for algorithm in EXPERIMENTS[alg_type]:
                if algorithm in results[dataset_type]:
                    result = results[dataset_type][algorithm]
                    stats['algorithm'].append(algorithm)
                    stats['duration'].append(result['duration'])
                    
                    # Get hard and soft violations
                    if 'hard_violations' in result:
                        hard_v = result['hard_violations']
                        soft_v = result['soft_violations']
                    else:
                        # Calculate violations from best schedule
                        hard_v = result['history']['hard_violations'][-1]
                        soft_v = result['history']['soft_violations'][-1]
                    
                    stats['hard_violations'].append(hard_v)
                    stats['soft_violations'].append(soft_v)
                    
                    # Get final fitness
                    if 'best_fitness' in result:
                        fitness = result['best_fitness'][-1]
                    else:
                        fitness = result['best_reward']
                    
                    stats['final_fitness'].append(fitness)
        
        # Create comparison bar chart for violations
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(stats['algorithm']))
        width = 0.35
        
        ax.bar(x - width/2, stats['hard_violations'], width, label='Hard Violations')
        ax.bar(x + width/2, stats['soft_violations'], width, label='Soft Violations')
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Violations')
        ax.set_title(f'Constraint Violations by Algorithm ({dataset_type} Dataset)')
        ax.set_xticks(x)
        ax.set_xticklabels(stats['algorithm'])
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(f"{analysis_dir}/violations_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create comparison bar chart for execution time
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.bar(stats['algorithm'], stats['duration'])
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'Execution Time by Algorithm ({dataset_type} Dataset)')
        
        plt.tight_layout()
        fig.savefig(f"{analysis_dir}/time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create comparison table
        comparison_table = {
            'Algorithm': stats['algorithm'],
            'Execution Time (s)': [f"{t:.2f}" for t in stats['duration']],
            'Hard Violations': stats['hard_violations'],
            'Soft Violations': [f"{v:.2f}" for v in stats['soft_violations']],
            'Final Fitness': [f"{f:.2f}" for f in stats['final_fitness']]
        }
        
        # Save as CSV
        with open(f"{analysis_dir}/comparison_table.csv", 'w') as f:
            # Write header
            f.write(','.join(comparison_table.keys()) + '\n')
            
            # Write rows
            for i in range(len(comparison_table['Algorithm'])):
                row = [str(comparison_table[key][i]) for key in comparison_table.keys()]
                f.write(','.join(row) + '\n')
        
        print(f"Comparative analysis saved to {analysis_dir}")
    
    print("\nComparative Analysis Complete!")
    return analysis_dir

# CELL: Run a Quick Test (Minimal Parameters)
def run_quick_test():
    """Run a quick test with minimal parameters to verify functionality."""
    print("\n===== Running Quick Test =====")
    
    # Override parameters for quick test
    GA_PARAMS['4room']['pop_size'] = 10
    GA_PARAMS['4room']['generations'] = 5
    RL_PARAMS['4room']['n_episodes'] = 5
    
    # Only run on 4-room dataset
    test_datasets = {'4room': DATASETS['4room']}
    
    # Create test results directory
    test_dir = "test_results"
    os.makedirs(test_dir, exist_ok=True)
    
    # Only run one algorithm from each type
    test_experiments = {
        'ga': ['nsga2'],
        'rl': ['q_learning']
    }
    
    # Save original settings
    orig_datasets = DATASETS.copy()
    orig_experiments = EXPERIMENTS.copy()
    
    # Override settings for test
    global DATASETS, EXPERIMENTS
    DATASETS = test_datasets
    EXPERIMENTS = test_experiments
    
    try:
        # Run test
        results, _ = run_all_experiments(test_dir)
        
        # Check results
        if results and '4room' in results:
            if 'nsga2' in results['4room'] and 'q_learning' in results['4room']:
                print("\nQuick test successfully completed!")
                return True
        
        print("\nQuick test failed!")
        return False
    
    finally:
        # Restore original settings
        DATASETS = orig_datasets
        EXPERIMENTS = orig_experiments

# CELL: Main Execution
# Uncomment this cell to run the experiments
# 
# # Run a quick test first
# if run_quick_test():
#     # Run all experiments
#     results, results_dir = run_all_experiments()
#     
#     # Perform comparative analysis
#     analysis_dir = comparative_analysis(results, results_dir)
#     
#     print(f"\nAll experiments completed. Results saved to {results_dir}")
#     print(f"Comparative analysis saved to {analysis_dir}")
# else:
#     print("\nQuick test failed. Please check the code and try again.")

# CELL: Individual Algorithm Execution
# Uncomment this cell to run specific algorithms
# 
# # Create results directory
# results_dir = create_results_dir()
# 
# # Load 4-room dataset
# data_tuple = load_data_from_local("Dataset/sliit_computing_dataset.json")
# display_dataset_info(data_tuple)
# 
# # Run NSGA-II on 4-room dataset
# result = run_ga_experiment('nsga2', '4room', data_tuple, results_dir)
# 
# # Run Q-Learning on 4-room dataset
# result = run_rl_experiment('q_learning', '4room', data_tuple, results_dir)
