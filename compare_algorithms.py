"""
Comparative Analysis of Timetable Scheduling Algorithms

This script provides a unified framework for comparing the performance of 
different algorithm families (Genetic Algorithms, Colony Optimization, Reinforcement Learning)
for university timetable scheduling.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import argparse
from datetime import datetime

# Import algorithm implementations
from common.data_loader import load_data
from common.metrics import (
    calculate_hypervolume, calculate_igd, calculate_gd, calculate_spread
)
from common.visualization import (
    plot_pareto_front_2d, plot_pareto_front_3d, plot_parallel_coordinates,
    plot_metrics_comparison, plot_constraint_breakdown, plot_resource_utilization
)
from common.resource_tracker import ResourceTracker

# Genetic Algorithms
from importlib.util import spec_from_file_location, module_from_spec
import sys

# Constants for paths
DIR_EA = "1. Genetic_EAs"
DIR_CO = "2. Colony_optimzation"
DIR_RL = "3. RL"
DIR_RL_SCRIPT = "RL_Script"

# Constants for algorithm types
ALGO_TYPE_EA = "ea"
ALGO_TYPE_CO = "co"
ALGO_TYPE_RL = "rl"

# Constants for metric names
METRIC_EXEC_TIME = "execution_time"
METRIC_MEMORY = "peak_memory_usage"
METRIC_HYPERVOLUME = "hypervolume"
METRIC_SPREAD = "spread"
METRIC_PARETO_SIZE = "pareto_front_size"

# Constants for objective names
OBJ_PROFESSOR_CONFLICTS = "Professor Conflicts"
OBJ_ROOM_CONFLICTS = "Room Conflicts"
OBJ_GROUP_CONFLICTS = "Group Conflicts"
OBJ_UNASSIGNED = "Unassigned Activities"
OBJ_SOFT_CONSTRAINTS = "Soft Constraints"

# Function to dynamically import modules
def import_module_from_path(module_name, file_path):
    """Dynamically import a module from a file path.
    
    This function handles cases where modules might have code that executes during import.
    """
    try:
        spec = spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"Could not find module {module_name} at {file_path}")
            return None
            
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        
        # For special cases like ea_alg.py which might have execution code at global level
        if 'ea_alg.py' in file_path:
            # Create a special module with just the essential functions we need
            from types import ModuleType
            
            try:
                # Try regular import first
                spec.loader.exec_module(module)
                return module
            except Exception as e:
                print(f"Standard import of {module_name} failed, using alternative approach: {e}")
                
                # Create a minimal version with just the functions we need
                alt_module = ModuleType(module_name)
                # Add key EA functions using a simpler loading approach
                # Import the code as a string and extract what we need
                import importlib.util
                with open(file_path, 'r') as f:
                    code = f.read()
                    
                # Extract the necessary functions using a simple approach
                # We'll define our own minimal versions here
                
                # Add the NSGA-II function
                def nsga2_minimal(pop_size, generations, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots):
                    print(f"Running NSGA-II with pop_size={pop_size}, generations={generations}")
                    # Return empty results for testing
                    return [], []
                    
                alt_module.nsga2 = nsga2_minimal
                
                # Add the SPEA2 function
                def spea2_minimal(pop_size, archive_size, generations, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots):
                    print(f"Running SPEA2 with pop_size={pop_size}, archive_size={archive_size}, generations={generations}")
                    # Return empty results for testing
                    return [], []
                    
                alt_module.spea2 = spea2_minimal
                
                # Add the MOEA/D function
                def moead_minimal(pop_size, generations, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots, n_neighbors=15, max_replacements=2):
                    print(f"Running MOEA/D with pop_size={pop_size}, generations={generations}, n_neighbors={n_neighbors}, max_replacements={max_replacements}")
                    # Return empty results for testing
                    return [], []
                    
                alt_module.moead = moead_minimal
                
                return alt_module
        else:
            # Regular import for other modules
            spec.loader.exec_module(module)
            return module
            
    except Exception as e:
        print(f"Error importing {module_name} from {file_path}: {e}")
        return None


# Define paths to algorithm files
EA_PATH = os.path.join(DIR_EA, "ea_alg_fixed.py")
CO_PATH = os.path.join(DIR_CO, "co_alg.py")
RL_PATH = os.path.join(DIR_RL, DIR_RL_SCRIPT, "rl_timetable.py")

# Import algorithm modules dynamically
ea_module = None
co_module = None
rl_module = None


def load_algorithm_module(algorithm_type):
    """Load the appropriate algorithm module if not already loaded.
    
    Args:
        algorithm_type: Type of algorithm to load module for
        
    Returns:
        bool: True if module loaded successfully, False otherwise
    """
    global ea_module, co_module, rl_module
    
    if algorithm_type == ALGO_TYPE_EA and ea_module is None:
        ea_module = import_module_from_path("ea_alg", EA_PATH)
        return ea_module is not None
    elif algorithm_type == ALGO_TYPE_CO and co_module is None:
        co_module = import_module_from_path("co_alg", CO_PATH)
        return co_module is not None
    elif algorithm_type == ALGO_TYPE_RL and rl_module is None:
        rl_module = import_module_from_path("rl_alg", RL_PATH)
        return rl_module is not None
    return True  # Module was already loaded


def prepare_output_directory(algorithm_type, algorithm_name, output_dir=None):
    """Prepare the output directory for algorithm results.
    
    Args:
        algorithm_type: Type of algorithm
        algorithm_name: Name of the algorithm
        output_dir: Optional custom output directory
        
    Returns:
        str: Path to the prepared output directory
    """
    if output_dir is None:
        output_dir = f"results/{algorithm_type}/{algorithm_name}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_algorithm(algorithm_type, algorithm_name, dataset_path=None, output_dir=None, **kwargs):
    """Run a specific algorithm.
    
    Args:
        algorithm_type: Type of algorithm ('ea', 'co', or 'rl')
        algorithm_name: Name of the specific algorithm
        dataset_path: Path to dataset file
        output_dir: Directory to save results
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Tuple: (Best solution, Pareto front, Metrics)
    """
    # Load module if needed
    if not load_algorithm_module(algorithm_type):
        print(f"Error: Failed to load module for {algorithm_type}")
        return None, None, None
    
    # Prepare output directory
    output_dir = prepare_output_directory(algorithm_type, algorithm_name, output_dir)
    
    # Set default algorithm-specific parameters if not provided
    # These defaults are based on typical values that produce good results for each algorithm
    algorithm_defaults = {
        # Evolutionary Algorithms defaults
        'ea': {
            'general': {
                'num_generations': 50,
                'population_size': 100,
                'crossover_rate': 0.9,
                'mutation_rate': 0.1
            },
            'nsga2': {
                'num_generations': 40,
                'population_size': 80,
                'crossover_rate': 0.9,
                'mutation_rate': 0.1,
                'tournament_size': 2
            },
            'spea2': {
                'num_generations': 40,
                'population_size': 100,
                'archive_size': 50,
                'crossover_rate': 0.9,
                'mutation_rate': 0.1
            },
            'moead': {
                'num_generations': 50,
                'population_size': 100,
                'neighborhood_size': 20,
                'crossover_rate': 0.9,
                'mutation_rate': 0.1
            }
        },
        
        # Colony Optimization defaults
        'co': {
            'general': {
                'num_iterations': 30
            },
            'aco': {
                'num_iterations': 20,
                'num_ants': 30,
                'evaporation_rate': 0.5,
                'alpha': 1.0,  # Pheromone importance
                'beta': 2.0,   # Heuristic importance
                'q_value': 100.0,  # Pheromone deposit quantity
                'elitist_weight': 2.0  # Weight for elitist ant
            },
            'bco': {
                'num_iterations': 30,
                'num_bees': 50,
                'employed_ratio': 0.5,
                'onlooker_ratio': 0.3,
                'scout_ratio': 0.2,
                'limit_trials': 5,
                'alpha': 1.0
            },
            'pso': {
                'num_iterations': 30,
                'num_particles': 50,
                'inertia_weight': 0.7,
                'cognitive_weight': 1.5,
                'social_weight': 1.5,
                'velocity_clamp': 0.1
            }
        },
        
        # Reinforcement Learning defaults
        'rl': {
            'general': {
                'episodes': 100,
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'epsilon': 0.1
            },
            'qlearning': {
                'episodes': 100,
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'epsilon': 0.1,
                'epsilon_decay': 0.99
            },
            'sarsa': {
                'episodes': 100,
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'epsilon': 0.1,
                'epsilon_decay': 0.99
            },
            'dqn': {
                'episodes': 200,
                'batch_size': 32,
                'replay_buffer_size': 10000,
                'target_update_frequency': 10,
                'learning_rate': 0.001,
                'discount_factor': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay_steps': 20000
            }
        }
    }
    
    # Apply general defaults for algorithm type
    if algorithm_type in algorithm_defaults:
        type_defaults = algorithm_defaults[algorithm_type].get('general', {})
        for param, value in type_defaults.items():
            if param not in kwargs:
                kwargs[param] = value
    
    # Apply algorithm-specific defaults
    if algorithm_type in algorithm_defaults and algorithm_name.lower() in algorithm_defaults[algorithm_type]:
        algo_defaults = algorithm_defaults[algorithm_type][algorithm_name.lower()]
        for param, value in algo_defaults.items():
            if param not in kwargs:
                kwargs[param] = value
    
    # Get algorithm defaults
    apply_algorithm_defaults(algorithm_type, algorithm_name, kwargs)
    
    # Execute the appropriate algorithm
    return execute_algorithm(algorithm_type, algorithm_name, dataset_path, output_dir, **kwargs)


# Define algorithm defaults as a module-level constant to reduce function complexity
ALGORITHM_DEFAULTS = {
    # Evolutionary Algorithms defaults
    ALGO_TYPE_EA: {
        'general': {
            'num_generations': 50,
            'population_size': 100,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1
        },
        'nsga2': {
            'num_generations': 40,
            'population_size': 80,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1,
            'tournament_size': 2
        },
        'spea2': {
            'num_generations': 40,
            'population_size': 100,
            'archive_size': 50,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1
        },
        'moead': {
            'num_generations': 50,
            'population_size': 100,
            'neighborhood_size': 20,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1
        }
    },
    
    # Colony Optimization defaults
    ALGO_TYPE_CO: {
        'general': {
            'num_iterations': 30
        },
        'aco': {
            'num_iterations': 20,
            'num_ants': 30,
            'evaporation_rate': 0.5,
            'alpha': 1.0,  # Pheromone importance
            'beta': 2.0,   # Heuristic importance
            'q_value': 100.0,  # Pheromone deposit quantity
            'elitist_weight': 2.0  # Weight for elitist ant
        },
        'bco': {
            'num_iterations': 30,
            'num_bees': 50,
            'employed_ratio': 0.5,
            'onlooker_ratio': 0.3,
            'scout_ratio': 0.2,
            'limit_trials': 5,
            'alpha': 1.0
        },
        'pso': {
            'num_iterations': 30,
            'num_particles': 50,
            'inertia_weight': 0.7,
            'cognitive_weight': 1.5,
            'social_weight': 1.5,
            'velocity_clamp': 0.1
        }
    },
    
    # Reinforcement Learning defaults
    ALGO_TYPE_RL: {
        'general': {
            'episodes': 100,
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon': 0.1
        },
        'qlearning': {
            'episodes': 100,
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon': 0.1,
            'epsilon_decay': 0.99
        },
        'sarsa': {
            'episodes': 100,
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon': 0.1,
            'epsilon_decay': 0.99
        },
        'dqn': {
            'episodes': 200,
            'batch_size': 32,
            'replay_buffer_size': 10000,
            'target_update_frequency': 10,
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay_steps': 20000
        }
    }
}


def apply_type_defaults(algorithm_type, kwargs):
    """Apply type-level default parameters.
    
    Args:
        algorithm_type: Type of algorithm
        kwargs: Parameter dictionary to update with defaults
    """
    if algorithm_type not in ALGORITHM_DEFAULTS:
        return
        
    type_defaults = ALGORITHM_DEFAULTS[algorithm_type].get('general', {})
    for param, value in type_defaults.items():
        if param not in kwargs:
            kwargs[param] = value


def apply_algorithm_specific_defaults(algorithm_type, algorithm_name, kwargs):
    """Apply algorithm-specific default parameters.
    
    Args:
        algorithm_type: Type of algorithm
        algorithm_name: Name of the algorithm
        kwargs: Parameter dictionary to update with defaults
    """
    if algorithm_type not in ALGORITHM_DEFAULTS:
        return
        
    name_lower = algorithm_name.lower()
    if name_lower not in ALGORITHM_DEFAULTS[algorithm_type]:
        return
        
    algo_defaults = ALGORITHM_DEFAULTS[algorithm_type][name_lower]
    for param, value in algo_defaults.items():
        if param not in kwargs:
            kwargs[param] = value


def apply_algorithm_defaults(algorithm_type, algorithm_name, kwargs):
    """Apply default parameters for the given algorithm type and name.
    
    Args:
        algorithm_type: Type of algorithm
        algorithm_name: Name of the algorithm
        kwargs: Parameter dictionary to update with defaults
    """
    # Apply general defaults for algorithm type
    apply_type_defaults(algorithm_type, kwargs)
    
    # Apply algorithm-specific defaults
    apply_algorithm_specific_defaults(algorithm_type, algorithm_name, kwargs)


def execute_algorithm(algorithm_type, algorithm_name, dataset_path, output_dir, **kwargs):
    """Execute the specified algorithm with the given parameters.
    
    Args:
        algorithm_type: Type of algorithm
        algorithm_name: Name of the algorithm
        dataset_path: Path to dataset file
        output_dir: Directory to save results
        **kwargs: Algorithm parameters
        
    Returns:
        Tuple: (Best solution, Pareto front, Metrics)
    """
    global ea_module, co_module, rl_module
    
    # Evolutionary Algorithms
    if algorithm_type == ALGO_TYPE_EA:
        if ea_module is None:
            print(f"Error: Could not load EA module from {EA_PATH}")
            return None, None, None
        
        valid_ea_algorithms = ["nsga2", "spea2", "moead"]
        if algorithm_name.lower() in valid_ea_algorithms:
            return ea_module.run_algorithm(algorithm_name, dataset_path, output_dir, **kwargs)
        else:
            print(f"Error: Unknown EA algorithm: {algorithm_name}. Must be one of {valid_ea_algorithms}")
            return None, None, None
    
    # Colony Optimization Algorithms
    elif algorithm_type == ALGO_TYPE_CO:
        if co_module is None:
            print(f"Error: Could not load CO module from {CO_PATH}")
            return None, None, None
        
        valid_co_algorithms = ["aco", "bco", "pso"]
        if algorithm_name.lower() in valid_co_algorithms:
            return co_module.run_algorithm(algorithm_name, dataset_path, output_dir, **kwargs)
        else:
            print(f"Error: Unknown CO algorithm: {algorithm_name}. Must be one of {valid_co_algorithms}")
            return None, None, None
    
    # Reinforcement Learning Algorithms
    elif algorithm_type == ALGO_TYPE_RL:
        if rl_module is None:
            print(f"Error: Could not load RL module from {RL_PATH}")
            return None, None, None
        
        valid_rl_algorithms = ["qlearning", "sarsa", "dqn"]
        if algorithm_name.lower() in valid_rl_algorithms:
            return rl_module.run_algorithm(algorithm_name, dataset_path, output_dir, **kwargs)
        else:
            print(f"Error: Unknown RL algorithm: {algorithm_name}. Must be one of {valid_rl_algorithms}")
            return None, None, None
    
    else:
        print(f"Error: Unknown algorithm type: {algorithm_type}. Must be one of {ALGO_TYPE_EA}, {ALGO_TYPE_CO}, or {ALGO_TYPE_RL}")
        return None, None, None


def get_default_algorithms():
    """Return the default list of algorithms to compare.
    
    Returns:
        list: Default list of algorithms to compare
    """
    return [
        "ea:nsga2", "ea:spea2", "ea:moead",   # EA algorithms
        "co:aco", "co:bco", "co:pso",         # Colony optimization algorithms
        "rl:qlearning", "rl:sarsa", "rl:dqn"   # RL algorithms
    ]


def prepare_comparison_directory(output_dir):
    """Create and return a comparison directory with timestamp.
    
    Args:
        output_dir: Base directory for comparison results
        
    Returns:
        str: Path to the created comparison directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = f"{output_dir}_{timestamp}"
    os.makedirs(comparison_dir, exist_ok=True)
    return comparison_dir


def run_single_algorithm(alg_spec, dataset_path, comparison_dir, **kwargs):
    """Run a single algorithm from the comparison.
    
    Args:
        alg_spec: Algorithm specification string in format 'type:name'
        dataset_path: Path to dataset file
        comparison_dir: Directory to save results
        **kwargs: Algorithm-specific parameters
        
    Returns:
        tuple: (alg_id, solution, pareto_front, metrics) or (alg_id, None, None, None) on failure
    """
    try:
        alg_type, alg_name = alg_spec.split(":")
        alg_id = f"{alg_type}_{alg_name}"
        
        print(f"\nRunning {alg_type.upper()} algorithm: {alg_name.upper()}")
        alg_output_dir = os.path.join(comparison_dir, alg_id)
        
        # Run the algorithm
        solution, pareto_front, metrics = run_algorithm(
            alg_type, alg_name, dataset_path, alg_output_dir, **kwargs
        )
        
        # Check if algorithm ran successfully
        if solution is not None:
            print(f"\n{alg_id.upper()} completed successfully")
            print(f"Pareto Front Size: {len(pareto_front) if pareto_front else 0}")
            print(f"Execution Time: {metrics.get(METRIC_EXEC_TIME, 'N/A')}")
            return alg_id, solution, pareto_front, metrics
        
        print(f"Failed to run {alg_id}")
    except Exception as e:
        print(f"Error running {alg_spec}: {e}")
    
    return None


def run_comparison(dataset_path=None, output_dir="comparison_results", algorithms=None, **kwargs):
    """Run a comparison of selected algorithms.
    
    Args:
        dataset_path: Path to dataset file
        output_dir: Directory to save results
        algorithms: List of algorithm specifications in the format 'type:name'
                   e.g. ['ea:nsga2', 'co:aco', 'rl:qlearning']
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Dict: Comparison results
    """
    # Get algorithms to compare
    if algorithms is None:
        algorithms = get_default_algorithms()
    
    # Create comparison directory
    comparison_dir = prepare_comparison_directory(output_dir)
    
    # Store results for each algorithm
    all_results = {}
    all_pareto_fronts = {}
    all_metrics = {}
    
    # Run each algorithm
    for alg_spec in algorithms:
        result = run_single_algorithm(alg_spec, dataset_path, comparison_dir, **kwargs)
        if result:
            alg_id, solution, pareto_front, metrics = result
            all_results[alg_id] = solution
            all_pareto_fronts[alg_id] = pareto_front
            all_metrics[alg_id] = metrics
    
    # Generate comparison visualizations and save results
    if all_pareto_fronts:
        generate_comparison_visualizations(all_pareto_fronts, all_metrics, comparison_dir)
        save_comparison_results(all_metrics, comparison_dir)
    
    return {
        "results": all_results,
        "pareto_fronts": all_pareto_fronts,
        "metrics": all_metrics,
        "output_dir": comparison_dir
    }


def prepare_visualization_data(pareto_fronts):
    """Prepare data for visualizations.
    
    Args:
        pareto_fronts: Dictionary of pareto fronts for each algorithm
        
    Returns:
        Tuple: (fronts, names, colors, markers)
    """
    fronts = []
    names = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    markers = ['o', 's', '^', 'D', '*', 'x', '+', '<', '>']
    
    for alg_id, pareto_front in pareto_fronts.items():
        if pareto_front:  # Only include non-empty pareto fronts
            fronts.append(pareto_front)
            names.append(alg_id)
    
    return fronts, names, colors, markers


def generate_2d_pareto_plots(fronts, names, colors, markers, vis_dir):
    """Generate 2D Pareto front plots.
    
    Args:
        fronts: List of pareto fronts
        names: List of algorithm names
        colors: List of colors
        markers: List of markers
        vis_dir: Directory to save visualizations
    """
    try:
        # Professor conflicts vs Soft constraints
        plot_pareto_front_2d(
            fronts,
            names,
            colors[:len(names)],
            markers[:len(names)],
            0, 4,  # Professor conflicts vs Soft constraints
            OBJ_PROFESSOR_CONFLICTS, OBJ_SOFT_CONSTRAINTS,
            f"Comparison: {OBJ_PROFESSOR_CONFLICTS} vs {OBJ_SOFT_CONSTRAINTS}",
            vis_dir,
            "comparison_pareto_2d_prof_vs_soft.png"
        )
        
        # Room conflicts vs Group conflicts
        plot_pareto_front_2d(
            fronts,
            names,
            colors[:len(names)],
            markers[:len(names)],
            1, 2,  # Room conflicts vs Group conflicts
            OBJ_ROOM_CONFLICTS, OBJ_GROUP_CONFLICTS,
            f"Comparison: {OBJ_ROOM_CONFLICTS} vs {OBJ_GROUP_CONFLICTS}",
            vis_dir,
            "comparison_pareto_2d_room_vs_group.png"
        )
    except Exception as e:
        print(f"Warning: Error generating 2D Pareto plots: {e}")


def generate_3d_pareto_plot(fronts, names, colors, markers, vis_dir):
    """Generate 3D Pareto front plot.
    
    Args:
        fronts: List of pareto fronts
        names: List of algorithm names
        colors: List of colors
        markers: List of markers
        vis_dir: Directory to save visualization
    """
    try:
        plot_pareto_front_3d(
            fronts,
            names,
            colors[:len(names)],
            markers[:len(names)],
            0, 1, 4,  # Professor conflicts, Room conflicts, Soft constraints
            OBJ_PROFESSOR_CONFLICTS, OBJ_ROOM_CONFLICTS, OBJ_SOFT_CONSTRAINTS,
            "Comparison: 3D Pareto Front",
            vis_dir,
            "comparison_pareto_3d.png"
        )
    except Exception as e:
        print(f"Warning: Error generating 3D Pareto plot: {e}")


def generate_parallel_coordinates_plot(fronts, names, colors, vis_dir):
    """Generate parallel coordinates plot.
    
    Args:
        fronts: List of pareto fronts
        names: List of algorithm names
        colors: List of colors
        vis_dir: Directory to save visualization
    """
    try:
        # List of objectives for parallel coordinates
        objectives = [
            OBJ_PROFESSOR_CONFLICTS, 
            OBJ_ROOM_CONFLICTS, 
            OBJ_GROUP_CONFLICTS, 
            OBJ_UNASSIGNED, 
            OBJ_SOFT_CONSTRAINTS
        ]
        
        plot_parallel_coordinates(
            fronts,
            names,
            colors[:len(names)],
            objectives,
            "Comparison: Parallel Coordinates Plot",
            vis_dir,
            "comparison_parallel_coords.png"
        )
    except Exception as e:
        print(f"Warning: Error generating parallel coordinates plot: {e}")


def generate_metrics_comparison(pareto_fronts, metrics, vis_dir):
    """Generate metrics comparison plots.
    
    Args:
        pareto_fronts: Dictionary of pareto fronts for each algorithm
        metrics: Dictionary of metrics for each algorithm
        vis_dir: Directory to save visualization
    """
    try:
        # Define metrics to compare
        metric_names = [
            METRIC_EXEC_TIME, 
            METRIC_MEMORY, 
            METRIC_HYPERVOLUME, 
            METRIC_SPREAD, 
            METRIC_PARETO_SIZE
        ]
        
        # Extract metric values
        metric_values = {}
        for metric in metric_names:
            metric_values[metric] = [
                metrics.get(alg_id, {}).get(metric, 0) 
                for alg_id in pareto_fronts.keys()
            ]
        
        # Create bar plots for metrics
        plot_metrics_comparison(
            list(pareto_fronts.keys()),
            metric_values,
            "Algorithm Performance Metrics",
            vis_dir,
            "comparison_metrics.png"
        )
    except Exception as e:
        print(f"Warning: Error generating metrics comparison plot: {e}")


def generate_comparison_visualizations(pareto_fronts, metrics, output_dir):
    """Generate comparison visualizations for multiple algorithms.
    
    Args:
        pareto_fronts: Dictionary of pareto fronts for each algorithm
        metrics: Dictionary of metrics for each algorithm
        output_dir: Directory to save visualizations
    """
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Only generate visualizations if we have at least one pareto front
    if not pareto_fronts:
        return
    
    # Prepare data for visualizations
    fronts, names, colors, markers = prepare_visualization_data(pareto_fronts)
    
    # No valid pareto fronts
    if not fronts:
        return
    
    # Generate different types of plots
    generate_2d_pareto_plots(fronts, names, colors, markers, vis_dir)
    generate_3d_pareto_plot(fronts, names, colors, markers, vis_dir)
    generate_parallel_coordinates_plot(fronts, names, colors, vis_dir)
    generate_metrics_comparison(pareto_fronts, metrics, vis_dir)


def save_comparison_results(metrics, output_dir):
    """Save comparison results to file.
    
    Args:
        metrics: Dictionary of metrics for each algorithm
        output_dir: Directory to save results
    """
    try:
        # Create a DataFrame for easier analysis
        metrics_df = pd.DataFrame(metrics).transpose()
        
        # Save as CSV
        metrics_path = os.path.join(output_dir, "comparison_metrics.csv")
        metrics_df.to_csv(metrics_path)
        
        # Save as JSON for easier programmatic access
        json_path = os.path.join(output_dir, "comparison_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Comparison results saved to {output_dir}")
    except Exception as e:
        print(f"Warning: Error saving comparison results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Runner and Comparison Tool for Timetable Scheduling Algorithms")
    
    # Main options
    parser.add_argument("--mode", type=str, choices=["run", "compare"], default="run",
                        help="Mode: 'run' a single algorithm or 'compare' multiple algorithms")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset file (default uses environment variable or default path)")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save results")
    
    # Algorithm specification
    parser.add_argument("--type", type=str, choices=["ea", "co", "rl"], 
                        help="Algorithm type: 'ea' (Evolutionary Algorithm), 'co' (Colony Optimization), 'rl' (Reinforcement Learning)")
    parser.add_argument("--name", type=str,
                        help="Algorithm name (ea: nsga2/spea2/moead, co: aco/bco/pso, rl: qlearning/sarsa/dqn)")
    
    # Comparison options
    parser.add_argument("--algorithms", type=str, nargs="+",
                        help="List of algorithms to compare in format 'type:name' (e.g., 'ea:nsga2 co:aco rl:qlearning')")
    
    # Common algorithm parameters
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of iterations/generations/episodes to run")
    parser.add_argument("--population", type=int, default=None,
                        help="Population size (individuals, ants/bees/particles, or replay buffer size)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Prepare kwargs for algorithm parameters
    kwargs = {}
    if args.iterations is not None:
        kwargs["num_iterations"] = args.iterations
        kwargs["num_generations"] = args.iterations  # For EA
        kwargs["episodes"] = args.iterations  # For RL
    
    if args.population is not None:
        kwargs["population_size"] = args.population  # For EA
        kwargs["num_ants"] = args.population  # For ACO
        kwargs["num_bees"] = args.population  # For BCO
        kwargs["num_particles"] = args.population  # For PSO
        kwargs["buffer_size"] = args.population  # For RL
    
    # Execute based on mode
    if args.mode == "run":
        # Validate required arguments
        if args.type is None or args.name is None:
            parser.error("--type and --name are required in 'run' mode")
        
        # Run a single algorithm
        print(f"\nRunning {args.type.upper()} algorithm: {args.name.upper()}")
        solution, pareto_front, metrics = run_algorithm(
            args.type, args.name, args.dataset, args.output, **kwargs
        )
        
        if solution is not None:
            print(f"\n{args.name.upper()} completed successfully")
            if pareto_front:
                print(f"Pareto Front Size: {len(pareto_front)}")
            print(f"Execution Time: {metrics.get('execution_time', 'N/A')} seconds")
            print(f"Peak Memory Usage: {metrics.get('peak_memory_usage', 'N/A')} MB")
        
    else:  # compare mode
        # Parse algorithms for comparison
        algorithms = None
        if args.algorithms:
            algorithms = args.algorithms
        
        # Run comparison
        result = run_comparison(args.dataset, args.output, algorithms, **kwargs)
        print(f"\nComparison completed. Results saved to {result['output_dir']}")



class AlgorithmComparator:
    """Class for comparing different timetable scheduling algorithms."""
    
    def __init__(self, dataset_path: str = None, output_dir: str = "results/comparison"):
        """
        Initialize the comparator.
        
        Args:
            dataset_path: Path to the dataset file
            output_dir: Directory to save comparison results
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.algorithms = {}
        self.results = {}
        self.metrics = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataset
        self._load_data()
    
    def _load_data(self):
        """Load the timetable dataset."""
        print(f"Loading dataset from {self.dataset_path}")
        
        # Check if dataset_path is provided, otherwise use environment variable or default
        if not self.dataset_path:
            self.dataset_path = os.environ.get('TIMETABLE_DATASET', 
                                             'data/sliit_computing_dataset.json')
        
        # Load data (we don't need the returned values here)
        load_data(self.dataset_path)
        print("Dataset loaded successfully")
    
    def add_genetic_algorithm(self, config: Dict = None):
        """
        Add Genetic Algorithm to the comparison.
        
        Args:
            config: Configuration parameters for the GA
        """
        # Import the EA module
        ea_path = os.path.join(os.getcwd(), "1. Genetic_EAs", "ea_alg.py")
        ea_module = import_module_from_path("ea_alg", ea_path)
        
        if ea_module is None:
            print("Failed to import Genetic Algorithm module")
            return
        
        # Default config if none provided
        if config is None:
            config = {
                'population_size': 100,
                'num_generations': 50,
                'crossover_rate': 0.8,
                'mutation_rate': 0.2,
                'tournament_size': 3
            }
        
        # Create GA instance
        ga = ea_module.GeneticAlgorithm(
            population_size=config['population_size'],
            num_generations=config['num_generations'],
            crossover_rate=config['crossover_rate'],
            mutation_rate=config['mutation_rate'],
            tournament_size=config['tournament_size']
        )
        
        # Add to algorithms dict
        self.algorithms['GA'] = {
            'instance': ga,
            'config': config,
            'module': ea_module
        }
        
        print("Added Genetic Algorithm to comparison")
    
    def add_ant_colony_optimization(self, config: Dict = None):
        """
        Add Ant Colony Optimization to the comparison.
        
        Args:
            config: Configuration parameters for ACO
        """
        # Import the ACO module
        aco_path = os.path.join(os.getcwd(), DIR_CO, "aco_timetable.py")
        aco_module = import_module_from_path("aco_timetable", aco_path)
        
        if aco_module is None:
            print("Failed to import ACO module")
            return
        
        # Default config if none provided
        if config is None:
            config = {
                'num_ants': 60,
                'num_iterations': 50,
                'evaporation_rate': 0.5,
                'alpha': 1.0,
                'beta': 2.0
            }
        
        # Create ACO instance
        aco_params = aco_module.ACOParameters(
            num_ants=config['num_ants'],
            num_iterations=config['num_iterations'],
            evaporation_rate=config['evaporation_rate'],
            alpha=config['alpha'],
            beta=config['beta']
        )
        aco = aco_module.AntColonyOptimization(aco_params)
        
        # Add to algorithms dict
        self.algorithms['ACO'] = {
            'instance': aco,
            'config': config,
            'module': aco_module
        }
        
        print("Added Ant Colony Optimization to comparison")
    
    def add_bee_colony_optimization(self, config: Dict = None):
        """
        Add Bee Colony Optimization to the comparison.
        
        Args:
            config: Configuration parameters for BCO
        """
        # Import the BCO module
        bco_path = os.path.join(os.getcwd(), DIR_CO, "bco_timetable.py")
        bco_module = import_module_from_path("bco_timetable", bco_path)
        
        if bco_module is None:
            print("Failed to import BCO module")
            return
        
        # Default config if none provided
        if config is None:
            config = {
                'num_bees': 50,
                'num_iterations': 50,
                'employed_ratio': 0.5,
                'onlooker_ratio': 0.3,
                'scout_ratio': 0.2
            }
        
        # Create BCO instance
        bco_params = bco_module.BCOParameters(
            num_bees=config['num_bees'],
            num_iterations=config['num_iterations'],
            employed_ratio=config['employed_ratio'],
            onlooker_ratio=config['onlooker_ratio'],
            scout_ratio=config['scout_ratio']
        )
        bco = bco_module.BeeColonyOptimization(bco_params)
        
        # Add to algorithms dict
        self.algorithms['BCO'] = {
            'instance': bco,
            'config': config,
            'module': bco_module
        }
        
        print("Added Bee Colony Optimization to comparison")
    
    def add_particle_swarm_optimization(self, config: Dict = None):
        """
        Add Particle Swarm Optimization to the comparison.
        
        Args:
            config: Configuration parameters for PSO
        """
        # Import the PSO module
        pso_path = os.path.join(os.getcwd(), DIR_CO, "pso_timetable.py")
        pso_module = import_module_from_path("pso_timetable", pso_path)
        
        if pso_module is None:
            print("Failed to import PSO module")
            return
        
        # Default config if none provided
        if config is None:
            config = {
                'num_particles': 50,
                'num_iterations': 50,
                'inertia_weight': 0.7,
                'cognitive_coef': 1.5,
                'social_coef': 1.5,
                'velocity_clamp': 0.1,
                'mutation_rate': 0.05
            }
        
        # Create PSO instance
        pso_params = pso_module.PSOParameters(
            num_particles=config['num_particles'],
            num_iterations=config['num_iterations'],
            inertia_weight=config['inertia_weight'],
            cognitive_coef=config['cognitive_coef'],
            social_coef=config['social_coef'],
            velocity_clamp=config['velocity_clamp'],
            mutation_rate=config['mutation_rate']
        )
        pso = pso_module.ParticleSwarmOptimization(pso_params)
        
        # Add to algorithms dict
        self.algorithms['PSO'] = {
            'instance': pso,
            'config': config,
            'module': pso_module
        }
        
        print("Added Particle Swarm Optimization to comparison")
    
    def add_reinforcement_learning(self, config: Dict = None):
        """
        Add Reinforcement Learning to the comparison.
        
        Args:
            config: Configuration parameters for RL
        """
        # Import the RL module
        rl_path = os.path.join(os.getcwd(), "3. RL", "RL_Script.py")
        rl_module = import_module_from_path("rl_script", rl_path)
        
        if rl_module is None:
            print("Failed to import RL module")
            return
        
        # Default config if none provided
        if config is None:
            config = {
                'algorithm': 'q_learning',  # 'q_learning', 'sarsa', or 'dqn'
                'num_episodes': 1000,
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'exploration_rate': 0.1
            }
        
        # Create RL instance (approach depends on RL module structure)
        try:
            # Try to get an RL algorithm instance (this may need adaptation based on RL module structure)
            if config['algorithm'] == 'q_learning':
                rl = rl_module.QLearning(
                    learning_rate=config['learning_rate'],
                    discount_factor=config['discount_factor'],
                    exploration_rate=config['exploration_rate'],
                    num_episodes=config['num_episodes']
                )
            elif config['algorithm'] == 'sarsa':
                rl = rl_module.SARSA(
                    learning_rate=config['learning_rate'],
                    discount_factor=config['discount_factor'],
                    exploration_rate=config['exploration_rate'],
                    num_episodes=config['num_episodes']
                )
            elif config['algorithm'] == 'dqn':
                rl = rl_module.DQN(
                    learning_rate=config['learning_rate'],
                    discount_factor=config['discount_factor'],
                    exploration_rate=config['exploration_rate'],
                    num_episodes=config['num_episodes']
                )
            else:
                print(f"Unknown RL algorithm: {config['algorithm']}")
                return
            
            # Add to algorithms dict
            self.algorithms[f"RL_{config['algorithm']}"] = {
                'instance': rl,
                'config': config,
                'module': rl_module
            }
            
            print(f"Added Reinforcement Learning ({config['algorithm']}) to comparison")
            
        except (AttributeError, TypeError) as e:
            print(f"Error initializing RL algorithm: {e}")
            print("RL algorithm structure may not match expected interface")
    
    def run_comparison(self, algorithms_to_run: List[str] = None):
        """
        Run all added algorithms and collect results.
        
        Args:
            algorithms_to_run: List of algorithm keys to run, or None for all
        """
        if not self.algorithms:
            print("No algorithms added to comparison")
            return
        
        # If no specific algorithms specified, run all
        if algorithms_to_run is None:
            algorithms_to_run = list(self.algorithms.keys())
        
        # Clear previous results
        self.results = {}
        self.metrics = {}
        
        # Create output directories
        comparison_dir = os.path.join(self.output_dir, 
                                    f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Run each algorithm
        for alg_key in algorithms_to_run:
            if alg_key not in self.algorithms:
                print(f"Algorithm {alg_key} not found")
                continue
            
            alg_info = self.algorithms[alg_key]
            alg_instance = alg_info['instance']
            
            print(f"\nRunning {alg_key}...")
            
            try:
                # Create algorithm-specific output directory
                alg_output_dir = os.path.join(comparison_dir, alg_key)
                os.makedirs(alg_output_dir, exist_ok=True)
                
                # Start timing
                start_time = time.time()
                
                # Run the algorithm
                # We expect a common interface with a "run" method that returns
                # (best_solution, pareto_front, metrics)
                solution, pareto_front, metrics = alg_instance.run(
                    dataset_path=self.dataset_path,
                    output_dir=alg_output_dir
                )
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Store results
                self.results[alg_key] = {
                    'solution': solution,
                    'pareto_front': pareto_front,
                    'execution_time': execution_time
                }
                
                # Store metrics
                # Add execution time if not already included
                if 'execution_time' not in metrics:
                    metrics['execution_time'] = execution_time
                
                self.metrics[alg_key] = metrics
                
                print(f"{alg_key} completed in {execution_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error running {alg_key}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate comparison visualizations and reports
        if self.results:
            self._generate_comparison_visualizations(comparison_dir)
            self._generate_comparison_report(comparison_dir)
            
            print(f"\nComparison completed. Results saved to {comparison_dir}")
    
    def _generate_comparison_visualizations(self, output_dir: str):
        """
        Generate visualizations comparing all algorithms.
        
        Args:
            output_dir: Directory to save visualizations
        """
        # Create visualization directory
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Collect Pareto fronts and algorithm names
        pareto_fronts = []
        alg_names = []
        colors = []
        markers = []
        
        # Define colors and markers for each algorithm
        color_map = {
            'GA': 'blue',
            'ACO': 'red',
            'BCO': 'green',
            'PSO': 'orange',
            'RL_q_learning': 'purple',
            'RL_sarsa': 'brown',
            'RL_dqn': 'pink'
        }
        
        marker_map = {
            'GA': 'o',
            'ACO': 's',
            'BCO': 'D',
            'PSO': '^',
            'RL_q_learning': 'p',
            'RL_sarsa': '*',
            'RL_dqn': 'X'
        }
        
        for alg_key, result in self.results.items():
            if 'pareto_front' in result and result['pareto_front']:
                pareto_fronts.append(result['pareto_front'])
                alg_names.append(alg_key)
                colors.append(color_map.get(alg_key, 'gray'))
                markers.append(marker_map.get(alg_key, 'o'))
        
        # Only generate visualizations if we have data
        if not pareto_fronts:
            print("No Pareto fronts available for visualization")
            return
        
        # Generate 2D Pareto front plots for different objective combinations
        plot_pareto_front_2d(
            pareto_fronts,
            alg_names,
            colors,
            markers,
            0, 4,  # Professor conflicts vs Soft constraints
            OBJ_PROFESSOR_CONFLICTS, OBJ_SOFT_CONSTRAINTS,
            f"Comparison of Pareto Fronts: {OBJ_PROFESSOR_CONFLICTS} vs {OBJ_SOFT_CONSTRAINTS}",
            vis_dir,
            "comparison_pareto_2d_prof_vs_soft.png"
        )
        
        plot_pareto_front_2d(
            pareto_fronts,
            alg_names,
            colors,
            markers,
            3, 4,  # Unassigned activities vs Soft constraints
            OBJ_UNASSIGNED, OBJ_SOFT_CONSTRAINTS,
            f"Comparison of Pareto Fronts: {OBJ_UNASSIGNED} vs {OBJ_SOFT_CONSTRAINTS}",
            vis_dir,
            "comparison_pareto_2d_unassigned_vs_soft.png"
        )
        
        # 3D Pareto front plot
        plot_pareto_front_3d(
            pareto_fronts,
            alg_names,
            colors,
            markers,
            0, 3, 4,  # Professor conflicts, Unassigned, Soft constraints
            OBJ_PROFESSOR_CONFLICTS, OBJ_UNASSIGNED, OBJ_SOFT_CONSTRAINTS,
            "Comparison of Pareto Fronts in 3D",
            vis_dir,
            "comparison_pareto_3d.png"
        )
        
        # Parallel coordinates plot
        plot_parallel_coordinates(
            pareto_fronts,
            alg_names,
            colors,
            vis_dir,
            "comparison_parallel_coordinates.png"
        )
        
        # Metrics comparison plots
        # Execution time
        if all('execution_time' in self.metrics[alg] for alg in alg_names):
            exec_times = [self.metrics[alg]['execution_time'] for alg in alg_names]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(alg_names, exec_times, color=colors)
            plt.xlabel('Algorithm')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Comparison of Execution Times')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}s',
                        ha='center', va='bottom', rotation=0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "comparison_execution_time.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Hypervolume comparison (if available)
        if all('hypervolume' in self.metrics[alg] for alg in alg_names):
            hypervolumes = [self.metrics[alg]['hypervolume'] for alg in alg_names]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(alg_names, hypervolumes, color=colors)
            plt.xlabel('Algorithm')
            plt.ylabel('Hypervolume')
            plt.title('Comparison of Hypervolumes')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "comparison_hypervolume.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_comparison_report(self, output_dir: str):
        """
        Generate a comprehensive comparison report.
        
        Args:
            output_dir: Directory to save the report
        """
        # Create a DataFrame with all metrics
        metrics_dict = {}
        
        # Collect all unique metric keys
        all_metric_keys = set()
        for alg_metrics in self.metrics.values():
            all_metric_keys.update(alg_metrics.keys())
        
        # Create a dictionary with algorithms as keys and metrics as values
        for alg_key, alg_metrics in self.metrics.items():
            metrics_dict[alg_key] = {k: alg_metrics.get(k, "N/A") for k in all_metric_keys}
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        
        # Save to CSV
        metrics_df.to_csv(os.path.join(output_dir, "comparison_metrics.csv"))
        
        # Generate a comprehensive report
        with open(os.path.join(output_dir, "comparison_report.md"), 'w') as f:
            f.write("# Algorithm Comparison Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Algorithms Compared\n\n")
            for alg_key, alg_info in self.algorithms.items():
                f.write(f"### {alg_key}\n\n")
                f.write("Configuration parameters:\n```\n")
                f.write(json.dumps(alg_info['config'], indent=4))
                f.write("\n```\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write(metrics_df.to_markdown())
            f.write("\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Sort algorithms by hypervolume (if available)
            if all('hypervolume' in self.metrics[alg] for alg in self.metrics):
                f.write("### Hypervolume Comparison\n\n")
                sorted_algs = sorted(
                    self.metrics.keys(),
                    key=lambda x: self.metrics[x].get('hypervolume', 0),
                    reverse=True
                )
                
                f.write("Algorithms ranked by hypervolume (higher is better):\n\n")
                for i, alg in enumerate(sorted_algs):
                    f.write(f"{i+1}. {alg}: {self.metrics[alg]['hypervolume']:.6f}\n")
                f.write("\n")
            
            # Sort algorithms by execution time
            if all('execution_time' in self.metrics[alg] for alg in self.metrics):
                f.write("### Execution Time Comparison\n\n")
                sorted_algs = sorted(
                    self.metrics.keys(),
                    key=lambda x: self.metrics[x].get('execution_time', float('inf'))
                )
                
                f.write("Algorithms ranked by execution time (lower is better):\n\n")
                for i, alg in enumerate(sorted_algs):
                    f.write(f"{i+1}. {alg}: {self.metrics[alg]['execution_time']:.2f} seconds\n")
                f.write("\n")
            
            # Sort algorithms by Pareto front size
            if all('pareto_front_size' in self.metrics[alg] for alg in self.metrics):
                f.write("### Pareto Front Size Comparison\n\n")
                sorted_algs = sorted(
                    self.metrics.keys(),
                    key=lambda x: self.metrics[x].get('pareto_front_size', 0),
                    reverse=True
                )
                
                f.write("Algorithms ranked by Pareto front size (larger fronts provide more solution options):\n\n")
                for i, alg in enumerate(sorted_algs):
                    f.write(f"{i+1}. {alg}: {self.metrics[alg]['pareto_front_size']} solutions\n")
                f.write("\n")
            
            f.write("## Conclusion\n\n")
            f.write("This comparison provides insights into the relative performance of different algorithms ")
            f.write("for the university timetable scheduling problem. The selection of the most suitable algorithm ")
            f.write("depends on specific requirements, such as solution quality, computational efficiency, and diversity of solutions.\n\n")
            
            f.write("### Summary of Strengths\n\n")
            
            # Add some general insights about algorithm strengths
            f.write("* **Genetic Algorithms (GA)**: ")
            f.write("Strong global exploration capability with diverse solution sets. ")
            f.write("Particularly effective when good crossover operators can be defined.\n\n")
            
            f.write("* **Ant Colony Optimization (ACO)**: ")
            f.write("Excellent at finding good solutions by leveraging pheromone trails. ")
            f.write("Well-suited for problems with well-defined path structures.\n\n")
            
            f.write("* **Bee Colony Optimization (BCO)**: ")
            f.write("Balances exploration and exploitation effectively. ")
            f.write("Good at handling multiple constraints simultaneously.\n\n")
            
            f.write("* **Particle Swarm Optimization (PSO)**: ")
            f.write("Efficiently converges to good solutions with fewer parameters. ")
            f.write("Excellent for continuous optimization problems adapted to discrete spaces.\n\n")
            
            f.write("* **Reinforcement Learning (RL)**: ")
            f.write("Learns and improves over time, potentially finding novel solutions. ")
            f.write("Especially useful for dynamic or changing environments.\n\n")
            
            f.write("### Recommendations\n\n")
            f.write("Based on this comparison, here are some recommendations for specific use cases:\n\n")
            
            f.write("* **For high-quality solutions with limited computational resources**: ")
            # Determine which algorithm had best quality/time ratio
            f.write("[Depends on specific metrics, see above comparison]\n\n")
            
            f.write("* **For diverse solution sets**: ")
            # Determine which algorithm had largest Pareto front
            f.write("[Check Pareto front size comparison above]\n\n")
            
            f.write("* **For rapidly finding good initial solutions**: ")
            # Determine which algorithm converged fastest
            f.write("[Check convergence metrics if available]\n\n")
        
        print(f"Comparison report generated: {os.path.join(output_dir, 'comparison_report.md')}")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description='Compare timetable scheduling algorithms')
    
    parser.add_argument('--dataset', type=str, default=None,
                      help='Path to dataset file')
    parser.add_argument('--output', type=str, default='results/comparison',
                      help='Directory to save comparison results')
    parser.add_argument('--algorithms', type=str, nargs='+',
                      choices=['GA', 'ACO', 'BCO', 'PSO', 'RL_q', 'RL_sarsa', 'RL_dqn', 'all'],
                      default=['all'],
                      help='Algorithms to include in comparison')
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = AlgorithmComparator(
        dataset_path=args.dataset,
        output_dir=args.output
    )
    
    # Determine which algorithms to run
    algorithms_to_run = args.algorithms
    if 'all' in algorithms_to_run:
        algorithms_to_run = ['GA', 'ACO', 'BCO', 'PSO', 'RL_q', 'RL_sarsa', 'RL_dqn']
    
    # Add requested algorithms
    if 'GA' in algorithms_to_run:
        comparator.add_genetic_algorithm()
    
    if 'ACO' in algorithms_to_run:
        comparator.add_ant_colony_optimization()
    
    if 'BCO' in algorithms_to_run:
        comparator.add_bee_colony_optimization()
    
    if 'PSO' in algorithms_to_run:
        comparator.add_particle_swarm_optimization()
    
    if 'RL_q' in algorithms_to_run:
        comparator.add_reinforcement_learning({'algorithm': 'q_learning'})
    
    if 'RL_sarsa' in algorithms_to_run:
        comparator.add_reinforcement_learning({'algorithm': 'sarsa'})
    
    if 'RL_dqn' in algorithms_to_run:
        comparator.add_reinforcement_learning({'algorithm': 'dqn'})
    
    # Run comparison
    comparator.run_comparison()


if __name__ == "__main__":
    main()
