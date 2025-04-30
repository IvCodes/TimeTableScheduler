"""
Run experiments on both 4-room and 7-room datasets to generate results for paper.
This script automates the execution of optimization algorithms and saves results
in the required format for visualization.
"""

import os
import sys
import json
import time
import importlib
import traceback
from pathlib import Path
from json import JSONEncoder
import random

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from algorithms.data.loader import load_data
from algorithms.metrics.tracker import MetricsTracker
from algorithms.evaluation.evaluator import evaluate_soft_constraints

class ActivityEncoder(JSONEncoder):
    """
    Custom JSON encoder to handle Activity objects.
    """
    def default(self, obj):
        if hasattr(obj, 'id') and hasattr(obj, 'subject') and hasattr(obj, 'teacher_id'):
            # Likely an Activity object
            return {
                'id': obj.id,
                'subject': obj.subject,
                'teacher_id': obj.teacher_id,
                'group_ids': obj.group_ids,
                'duration': obj.duration
            }
        # Let the base class handle it
        return JSONEncoder.default(self, obj)


def prepare_result_for_serialization(result):
    """
    Prepare the result dictionary for serialization by ensuring all values are JSON-serializable.
    
    Args:
        result: The result dictionary to prepare
        
    Returns:
        dict: A deep copy of the result dictionary with all values converted to JSON-serializable formats
    """
    # Create a new dict with only serializable elements
    clean_result = {}
    
    for key, value in result.items():
        # Skip the best_solution key, as we can't directly serialize Activity objects
        if key == 'best_solution':
            continue
        
        # Handle numpy values
        if hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
            clean_result[key] = value.tolist()
        elif isinstance(value, dict):
            # Recursively clean nested dictionaries
            clean_result[key] = prepare_result_for_serialization(value)
        else:
            clean_result[key] = value
    
    return clean_result


def calculate_room_utilization(timetable, spaces_dict, slots):
    """
    Calculate room utilization statistics for the timetable.
    
    Parameters:
        timetable (dict): The timetable to analyze
        spaces_dict (dict): Dictionary of spaces
        slots (list): List of time slots
        
    Returns:
        dict: Room utilization statistics
    """
    # Initialize room usage counters
    room_usage = {room_id: 0 for room_id in spaces_dict.keys()}
    total_slots = len(slots)
    
    # Count usage for each room
    for slot in slots:
        for room_id in spaces_dict.keys():
            if room_id in timetable.get(slot, {}) and timetable[slot][room_id] is not None:
                room_usage[room_id] += 1
    
    # Calculate statistics
    zero_usage_rooms = [room_id for room_id, usage in room_usage.items() if usage == 0]
    utilization_percentages = [100.0 * usage / total_slots for usage in room_usage.values()]
    avg_utilization = sum(utilization_percentages) / len(utilization_percentages) if utilization_percentages else 0.0
    
    return {
        'room_usage': room_usage,
        'avg_utilization': avg_utilization,
        'zero_usage_count': len(zero_usage_rooms),
        'zero_usage_rooms': zero_usage_rooms,
        'utilization_by_room': {room_id: 100.0 * usage / total_slots for room_id, usage in room_usage.items()}
    }

def evaluate_solution(timetable, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots):
    """
    Evaluate a timetable solution against the dataset constraints.
    
    Parameters:
        timetable (dict): The timetable to evaluate
        activities_dict (dict): Dictionary of activities
        groups_dict (dict): Dictionary of groups
        spaces_dict (dict): Dictionary of spaces
        lecturers_dict (dict): Dictionary of lecturers
        slots (list): List of time slots
        
    Returns:
        tuple: (hard_violations, unassigned_activities, soft_score)
    """
    # Calculate hard violations
    vacant_room = 0
    prof_conflicts = 0
    room_size_conflicts = 0
    sub_group_conflicts = 0
    unassigned_activities = len(activities_dict)
    activities_set = set()
    
    # Evaluate each slot in the timetable
    for slot in slots:
        prof_set = set()  # Track professors in this slot
        sub_group_set = set()  # Track student groups in this slot
        
        for room_id, activity in timetable.get(slot, {}).items():
            if activity is None:  # Empty slot
                vacant_room += 1
                continue
                
            if not hasattr(activity, 'id'):  # Not a valid activity object
                continue
                
            # Count this activity as assigned
            activities_set.add(activity.id)
            
            # Check for professor conflicts
            if activity.teacher_id in prof_set:
                prof_conflicts += 1
            prof_set.add(activity.teacher_id)
            
            # Check for student group conflicts
            for group_id in activity.group_ids:
                if group_id in sub_group_set:
                    sub_group_conflicts += 1
                sub_group_set.add(group_id)
            
            # Check room capacity constraints
            if room_id in spaces_dict:
                group_size = sum(groups_dict[group_id].size for group_id in activity.group_ids if group_id in groups_dict)
                if group_size > spaces_dict[room_id].size:
                    room_size_conflicts += 1
    
    # Calculate unassigned activities
    unassigned_activities -= len(activities_set)
    
    # Calculate total hard violations with higher weights for critical constraints
    hard_violations = (
        vacant_room * 1 +            # Vacant rooms (lower priority)
        prof_conflicts * 100 +        # Lecturer conflicts (high priority)
        sub_group_conflicts * 100 +   # Student group conflicts (high priority)
        room_size_conflicts * 50 +    # Room capacity (medium priority)
        unassigned_activities * 300   # Unassigned activities (highest priority - increased from 200)
    )
    
    # Calculate soft constraints based on student and lecturer metrics
    individual_soft_scores, soft_score = evaluate_soft_constraints(timetable, groups_dict, lecturers_dict, slots)

    # Print detailed evaluation results
    print("\n--- Hard Constraint Evaluation Results ---")
    print(f"Vacant Rooms Count: {vacant_room}")
    print(f"Lecturer Conflict Violations: {prof_conflicts}")
    print(f"Student Group Conflict Violations: {sub_group_conflicts}")
    print(f"Room Capacity Violations: {room_size_conflicts}")
    print(f"Unassigned Activity Violations: {unassigned_activities}")
    
    print("\n--- Soft Constraint Evaluation Results ---")
    # Check if individual_soft_scores is a tuple and has the expected length
    if isinstance(individual_soft_scores, tuple) and len(individual_soft_scores) == 7:
        print(f"Student Fatigue Factor: {individual_soft_scores[0]:.2f}")
        print(f"Student Idle Time Factor: {individual_soft_scores[1]:.2f}")
        print(f"Student Lecture Spread Factor: {individual_soft_scores[2]:.2f}")
        print(f"Lecturer Fatigue Factor: {individual_soft_scores[3]:.2f}")
        print(f"Lecturer Idle Time Factor: {individual_soft_scores[4]:.2f}")
        print(f"Lecturer Lecture Spread Factor: {individual_soft_scores[5]:.2f}")
        print(f"Lecturer Workload Balance: {individual_soft_scores[6]:.2f}")
        print(f"\nOverall Soft Score: {soft_score:.4f}") # Already correct
    else:
        print("Could not unpack individual soft scores. Received:", individual_soft_scores)
        print(f"\nOverall Soft Score: {soft_score:.4f}") # Still print the overall score

    # Return the results
    return hard_violations, unassigned_activities, soft_score

def save_results(timetable, metrics, filename):
    """
    Save timetable and metrics to a JSON file.
    
    Parameters:
        timetable (dict): Timetable solution
        metrics (dict): Performance metrics
        filename (str): Path to save the results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Prepare a serializable version of the timetable
    serializable_timetable = {}
    for slot in timetable:
        serializable_timetable[slot] = {}
        for room, activity in timetable[slot].items():
            if activity is None:
                serializable_timetable[slot][room] = None
            elif hasattr(activity, 'id'):
                serializable_timetable[slot][room] = activity.id
            else:
                serializable_timetable[slot][room] = str(activity)
    
    # Prepare result object
    result = {
        'timetable': serializable_timetable,
        'metrics': metrics
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)

def run_algorithm(algorithm_name, dataset_path, output_dir, algorithm_params=None):
    """
    Run the specified algorithm with the specified parameters.
    
    Parameters:
        algorithm_name (str): Name of the algorithm to run
        dataset_path (str): Path to the dataset file
        output_dir (str): Directory to save outputs
        algorithm_params (dict): Parameters for the algorithm
        
    Returns:
        dict: Results of the algorithm run
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the dataset name from the path
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    
    # Set default parameters if not provided
    if algorithm_params is None:
        algorithm_params = {}
        
    # Standardize the number of generations parameter
    algo_params = algorithm_params.copy() # Work with a copy
    algo_params.pop('generations', None)
    algo_params.pop('num_generations', None)
    # Removed default 'num_generations': 150
    
    # Conditionally add output_dir only if it's a GA algorithm
    ga_algorithms = {'nsga2', 'spea2', 'moead'}
    if algorithm_name in ga_algorithms:
        algo_params['output_dir'] = output_dir
    
    # Log the algorithm and parameters
    print(f"\nRunning {algorithm_name} on {os.path.basename(dataset_path)}...")
    
    # Load dataset without verbose logging
    print(f"Loading dataset {os.path.basename(dataset_path)}...")
    activities_dict, groups_dict, spaces_dict, lecturers_dict, slots = load_data(dataset_path)
    room_count = len(spaces_dict)
    activity_count = len(activities_dict)
    print(f"Dataset loaded: {activity_count} activities, {room_count} rooms")
    
    # Initialize module and function based on algorithm
    run_func = None
    if algorithm_name.upper() in ['NSGA2', 'SPEA2', 'MOEAD']:
        # Genetic algorithm
        if algorithm_name.upper() == 'NSGA2':
            from algorithms.ga.nsga2 import run_nsga2_optimizer as run_func
        elif algorithm_name.upper() == 'SPEA2':
            from algorithms.ga.spea2 import run_spea2_optimizer as run_func
        elif algorithm_name.upper() == 'MOEAD':
            from algorithms.ga.moead import run_moead_optimizer as run_func
            # MOEAD requires additional parameters
            algo_params['activities_dict'] = activities_dict
            algo_params['groups_dict'] = groups_dict
            algo_params['spaces_dict'] = spaces_dict
            algo_params['slots'] = slots
    elif algorithm_name.upper() in ['DQN', 'SARSA', 'QLEARNING']:
        # Reinforcement Learning algorithms
        if algorithm_name.upper() == 'DQN':
            from algorithms.rl.DQN_optimizer import run_dqn_optimizer as run_func
        elif algorithm_name.upper() == 'SARSA':
            from algorithms.rl.SARSA_optimizer import run_sarsa_optimizer as run_func
        elif algorithm_name.upper() == 'QLEARNING': # Assuming 'qlearning' maps to ImplicitQlearning
            from algorithms.rl.ImplicitQlearning_optimizer import run_implicit_qlearning_optimizer as run_func
        # RL algorithms might also need specific parameters passed, add them here if needed
        # Add data dictionaries required by RL optimizers
        algo_params['activities_dict'] = activities_dict
        algo_params['groups_dict'] = groups_dict
        algo_params['spaces_dict'] = spaces_dict
        algo_params['lecturers_dict'] = lecturers_dict
        algo_params['slots'] = slots

    else:
        print(f"Error: Algorithm '{algorithm_name}' is not recognized or supported.")
        return {
            'execution_time': 0.0,
            'hard_violations': float('inf'),
            'soft_score': 0.0,
            'room_utilization': 0.0,
            'success': False,
            'message': f"Algorithm '{algorithm_name}' not supported."
        }

    # Ensure run_func was assigned
    if run_func is None:
        print(f"Error: Could not find run function for algorithm '{algorithm_name}'.")
        return {
            'execution_time': 0.0,
            'hard_violations': float('inf'),
            'soft_score': 0.0,
            'room_utilization': 0.0,
            'success': False,
            'message': f"Run function for algorithm '{algorithm_name}' not found."
        }
    
    # Get the algorithm parameters
    params_to_log = {k: v for k, v in algo_params.items() if k != 'activities_dict'}
    print(f"Running algorithm with parameters (excluding activities_dict for brevity): {params_to_log}")
    
    if algorithm_params:
        algo_params.update(algorithm_params)
        
    # Run the algorithm with timing
    start_time = time.time()
    best_solution, metrics = run_func(**algo_params)
    execution_time = time.time() - start_time
    
    # Get metrics
    # Some functions directly return a metrics dict, others have a get_metrics method
    if hasattr(metrics, 'get_metrics'):
        metric_values = metrics.get_metrics()
    else:
        metric_values = metrics
            
    # Add execution time to metrics
    metric_values['execution_time'] = execution_time
    
    # Evaluate the best solution
    print("\nGenerating evaluation metrics for best solution...")
    hard_violations, unassigned_activities, soft_score = evaluate_solution(best_solution, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots)
    
    # Calculate room utilization statistics
    room_usage_stats = calculate_room_utilization(best_solution, spaces_dict, slots)
    print("\n--- Room Utilization Statistics ---")
    print(f"Average room utilization: {room_usage_stats['avg_utilization']:.2f}%")
    print(f"Rooms with zero usage: {room_usage_stats['zero_usage_count']} out of {room_count}")
    
    # Add room utilization to metrics
    metric_values['room_utilization'] = room_usage_stats
    
    # Save the results
    results_file = os.path.join(output_dir, "results_{}_{}.json".format(algorithm_name.lower(), dataset_name))
    save_results(best_solution, metric_values, results_file)
    
    # Log the results
    print("\n--- Overall Evaluation ---")
    print(f"Total Hard Constraint Violations: {hard_violations}")
    print(f"Soft Constraint Score: {soft_score:.2f}")
    print(f"Unassigned Activities: {unassigned_activities} out of {activity_count}")
    print(f"Room Utilization: {room_usage_stats['avg_utilization']:.2f}%")
    print(f"Final solution has {hard_violations} hard constraint violations and {soft_score:.4f} soft score")
    print(f"Results saved to {results_file}")
    
    # Return the best solution object and the metrics dictionary/object
    return best_solution, metrics 

def run_experiments(algorithms, datasets, output_dirs, params=None):
    """
    Run a batch of experiments across multiple algorithms and datasets.
    
    Args:
        algorithms (list): List of algorithm names
        datasets (list): List of dataset paths
        output_dirs (list): List of output directories matching datasets
        params (dict): Dictionary of algorithm parameters
    """
    results = {}
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\nRunning experiments on {dataset_name}\n================================================================================\n")
        results[dataset_name] = {}
        for algo_name in algorithms:
            algo_name_lower = algo_name.lower()
            print(f"\nRunning {algo_name_lower} on {dataset_name}.json...")
            
            # Determine output directory for this specific run (e.g., output/4rooms/nsga2)
            base_output_dir = output_dirs.get(dataset_name, 'output') # Get base dir for the dataset
            current_output_dir = os.path.join(base_output_dir, algo_name_lower) # Add algo subdir
            os.makedirs(current_output_dir, exist_ok=True)
            
            # Get parameters, ensuring output_dir is correctly set for this run
            algo_params = params.get(algo_name_lower, {}).copy() if params else {}
            algo_params['output_dir'] = current_output_dir # Override default output dir

            # Run the algorithm
            run_result = run_algorithm(
                algorithm_name=algo_name_lower, 
                dataset_path=dataset_path, 
                output_dir=current_output_dir,
                algorithm_params=algo_params
            )
            
            # Check if run_algorithm returned a valid result (tuple of length 2)
            if not isinstance(run_result, tuple) or len(run_result) != 2 or run_result[0] is None:
                # Treat None, (None, None), or anything else not like (solution, metrics) as failure
                error_msg = f"Execution failed or returned invalid result for {algo_name_lower} on {dataset_name}. Result: {run_result}"
                print(error_msg)
                results[dataset_name][algo_name_lower] = {'error': 'Execution failed or invalid result.', 'raw_result': str(run_result) }
                continue # Move to the next algorithm
                
            # --- Unpacking should be safe now ---
            _, metrics = load_results(full_path) 
            # print(f"DEBUG: Unpacked successfully.") # Optional debug print

            # Store results (only if execution was successful)
            results[dataset_name][algo_name_lower] = {
                'best_solution_summary': None, # Store a summary, not the whole object
                'metrics': metrics,
                'output_directory': current_output_dir
            }
            
            # Optional: Save detailed best solution if needed (can be large)
            # save_detailed_solution(best_solution, os.path.join(current_output_dir, f"{algo_name_lower}_best_solution.pkl"))

    # After all runs, aggregate or compare results if needed
    print("\n\n")
    print(f"{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, algorithms_results in results.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 40)
        
        for algorithm, result in algorithms_results.items():
            status = "Success" if 'error' not in result else "Failed" # Use plain text
            print(f"{algorithm.upper()}: {status}")
    
    print("\nAll experiments completed.")


if __name__ == "__main__":
    # Define algorithms to run
    ga_algorithms = ['nsga2', 'moead', 'spea2']
    rl_algorithms = ['dqn', 'sarsa', 'qlearning']
    co_algorithms = ['aco', 'bco', 'pso']
    
    # For the paper, we'll focus on GA and RL algorithms first
    paper_algorithms = ga_algorithms + rl_algorithms
    
    # Define datasets
    project_root = Path(__file__).resolve().parent.parent
    dataset_4rooms = str(project_root / "data/sliit_computing_dataset.json")
    dataset_7rooms = str(project_root / "data/sliit_computing_dataset_7.json")
    
    # Define output directories
    output_4rooms = str(project_root / "output")
    output_7rooms = str(project_root / "output_7room")
    
    # Algorithm parameters (can be customized)
    algorithm_params = {
        'nsga2': {
            'population_size': 50,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1
        },
        'moead': {
            'population_size': 50,
            'neighborhood_size': 10
        },
        'spea2': {
            'population_size': 50,
            'archive_size': 20
        },
        'dqn': {
            'episodes': 20,
            'epsilon': 0.1,
            'learning_rate': 0.001
        },
        'sarsa': {
            'episodes': 20,
            'alpha': 0.1,
            'gamma': 0.9
        },
        'qlearning': {
            'episodes': 100,
            'alpha': 0.1,
            'gamma': 0.9
        }
    }
    
    # Check if datasets exist
    if not os.path.exists(dataset_4rooms):
        print(f"Error: 4-room dataset not found at {dataset_4rooms}")
        print("Please ensure the dataset file exists before running experiments.")
        sys.exit(1)
    
    if not os.path.exists(dataset_7rooms):
        print(f"Error: 7-room dataset not found at {dataset_7rooms}")
        print("Please ensure the dataset file exists before running experiments.")
        sys.exit(1)
    
    # Run experiments
    print("Starting experiments for both 4-room and 7-room datasets...")
    
    run_experiments(
        paper_algorithms,
        {"4rooms": dataset_4rooms, "7rooms": dataset_7rooms},
        {"4rooms": output_4rooms, "7rooms": output_7rooms},
        algorithm_params
    )
    
    # Generate visualizations
    try:
        from algorithms.plotting.reviewer_plots import generate_all_paper_plots
        print("\nGenerating plots for paper...")
        generate_all_paper_plots(
            four_room_dir=output_4rooms,
            seven_room_dir=output_7rooms,
            output_dir=str(project_root / "paper_figures")
        )
        print("Plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
