import json
import os
import sys
import importlib
import time
import traceback
from typing import Dict, Any, Tuple, Optional

# Assuming this script will be in 'algorithms/' directory after refactoring
# Add project root to path if necessary (adjust based on final location)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Constants --- (Consider moving to a config file later)
DEFAULT_POPULATION = 50
DEFAULT_GENERATIONS = 100 # Matches run_all_experiments
DEFAULT_EPISODES = 100 # Matches run_all_experiments
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPSILON = 0.1
BASE_OUTPUT_DIR = "output" # Relative to project root

# Helper function (from old runner) - needs import update
# This conversion might be better placed within evaluation or utils
# For now, keep it here but acknowledge it needs review
from .utils.converters import timetable_to_json

def load_data(dataset_path: str):
    """Loads data dynamically based on the dataset path."""
    print(f"Loading data from: {dataset_path}")
    # Assuming loader.py has a function `load_timetable_data`
    try:
        from .data.loader import load_timetable_data
        return load_timetable_data(dataset_path)
    except ImportError:
        print("Error: Could not import load_timetable_data from .data.loader")
        # Fallback to old method temporarily if needed during transition
        # from Data_Loading import activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        # return activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        raise
    except Exception as e:
        print(f"Error loading data from {dataset_path}: {e}")
        raise

def run_optimization_algorithm(
    algorithm: str,
    dataset_path: str,
    population: int = DEFAULT_POPULATION,
    generations: int = DEFAULT_GENERATIONS,
    enable_plotting: bool = False, # Specific to SPEA2 old version?
    learning_rate: float = DEFAULT_LEARNING_RATE,
    episodes: int = DEFAULT_EPISODES,
    epsilon: float = DEFAULT_EPSILON,
    output_dir: str = None # Base output dir for the algorithm run (e.g., output/GA)
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Run the specified optimization algorithm for a given dataset.

    Args:
        algorithm: Algorithm name ('spea2', 'nsga2', 'moead', 'dqn', 'sarsa', 'implicit_q')
        dataset_path: Absolute path to the dataset JSON file.
        population: Population size for evolutionary algorithms.
        generations: Number of generations for evolutionary algorithms.
        enable_plotting: Whether to generate plots (relevant for SPEA2?).
        learning_rate: Learning rate for RL algorithms.
        episodes: Number of episodes for RL algorithms.
        epsilon: Exploration rate for RL algorithms.
        output_dir: Base directory for this algorithm's output.

    Returns:
        Tuple (best_solution, metrics dictionary) or (None, None) on error.
    """
    try:
        start_time = time.time()

        # --- Load Data --- (Now done dynamically)
        try:
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots = load_data(dataset_path)
        except Exception as e:
            print(f"Failed to load data for {algorithm} on {os.path.basename(dataset_path)}: {e}")
            return None, None

        # --- Dynamically import the module based on algorithm name (using new paths) ---
        module_path_prefix = "."
        if algorithm == 'nsga2':
            module_name = 'ga.nsga2'
            function_name = 'run_nsga2_optimizer'
        elif algorithm == 'spea2':
            module_name = 'ga.spea2'
            function_name = 'run_spea2_optimizer'
        elif algorithm == 'moead':
            # Assuming moead.py is the correct one, not moead_optimized.py
            module_name = 'ga.moead'
            function_name = 'run_moead_optimizer'
        elif algorithm == 'dqn':
            module_name = 'rl.DQN_optimizer' # Keep RL structure for now
            function_name = 'run_dqn_optimizer'
        elif algorithm == 'sarsa':
            module_name = 'rl.SARSA_optimizer'
            function_name = 'run_sarsa_optimizer'
        elif algorithm == 'implicit_q':
            module_name = 'rl.ImplicitQlearning_optimizer'
            function_name = 'run_implicit_qlearning_optimizer'
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Import the module
        try:
            module = importlib.import_module(f"{module_path_prefix}{module_name}", package='algorithms') # Adjust package if runner moves
        except ImportError as e:
             print(f"Error importing module {module_name}: {e}")
             print(f"sys.path: {sys.path}")
             raise

        # Get the optimizer function
        optimizer_func = getattr(module, function_name)

        # --- Prepare Algorithm Specific Parameters & Run --- 
        best_solution = None
        metrics_obj = None # Assuming optimizers return (solution, metrics_tracker_object)

        if algorithm in ['dqn', 'sarsa', 'implicit_q']:
            rl_params = {
                'activities_dict': activities_dict,
                'groups_dict': groups_dict,
                'spaces_dict': spaces_dict,
                'lecturers_dict': lecturers_dict,
                'slots': slots,
                'episodes': episodes,
                'epsilon': epsilon
            }
            if algorithm != 'implicit_q':
                rl_params['learning_rate'] = learning_rate

            best_solution, metrics_obj = optimizer_func(**rl_params)

        else: # Evolutionary Algorithms
            # Define the specific output directory *for this run*
            # Example: output/GA/nsga2/sliit_computing_dataset_results.json
            dataset_name_short = os.path.splitext(os.path.basename(dataset_path))[0]
            run_output_dir = os.path.join(output_dir, algorithm, dataset_name_short)
            if not os.path.exists(run_output_dir):
                 os.makedirs(run_output_dir)
            print(f"Output for this run will be in: {run_output_dir}")

            ga_params = {
                'activities_dict': activities_dict,
                'groups_dict': groups_dict,
                'spaces_dict': spaces_dict,
                'slots': slots,
                'population_size': population,
                 # Pass the specific run directory
                'output_dir': run_output_dir
            }
            # Adjust parameter names based on specific GA function signatures
            if algorithm == 'nsga2' or algorithm == 'moead':
                 ga_params['num_generations'] = generations
            elif algorithm == 'spea2':
                 ga_params['generations'] = generations
                 ga_params['enable_plotting'] = enable_plotting # Keep for now
            else:
                 # Default if signatures vary more
                 ga_params['generations'] = generations

            best_solution, metrics_obj = optimizer_func(**ga_params)

        # --- Post-processing & Evaluation --- (Needs careful review/simplification)
        if best_solution is None or metrics_obj is None:
             print(f"Algorithm {algorithm} did not return a valid solution or metrics.")
             return None, None

        # Convert timetable format if needed (This logic needs verification & simplification)
        # Consider moving conversion logic into the respective algorithm files or a util
        converted_solution = best_solution # Assume compatible format initially
        if algorithm == 'spea2': # Example: SPEA2 might return different format
            pass # Add conversion if necessary based on actual spea2 output
        elif algorithm == 'sarsa': # Example: SARSA might return different format
             pass # Add conversion if necessary

        # Evaluate final solution
        try:
            from .evaluation.evaluator import evaluate_timetable
            hard_violations, soft_score = evaluate_timetable(
                converted_solution,
                activities_dict,
                groups_dict,
                spaces_dict,
                lecturers_dict,
                slots,
                verbose=False
            )
        except ImportError:
             print("Error: Could not import evaluate_timetable from .evaluation.evaluator")
             hard_violations, soft_score = -1, -1 # Indicate evaluation failure
        except Exception as e:
             print(f"Error during final evaluation: {e}")
             hard_violations, soft_score = -1, -1

        end_time = time.time()
        execution_time = end_time - start_time

        # --- Prepare Results Dictionary --- 
        # Extract metrics from the tracker object
        final_metrics = metrics_obj.metrics if hasattr(metrics_obj, 'metrics') else {}

        # Add final evaluation and timing to metrics
        final_metrics['final_hard_violations'] = sum(hard_violations.values()) if isinstance(hard_violations, dict) else hard_violations
        final_metrics['final_soft_score'] = soft_score
        final_metrics['total_execution_time'] = execution_time
        # Add more metrics if needed for analysis (e.g., room utilization - keep minimal)

        # Convert best_solution to JSON serializable format for saving (optional)
        # json_timetable = timetable_to_json(best_solution) # If needed

        print(f"Algorithm {algorithm} finished in {execution_time:.2f} seconds.")
        print(f"Final Hard Violations: {final_metrics['final_hard_violations']}")
        print(f"Final Soft Score: {final_metrics['final_soft_score']}")

        # Return the raw best solution and the final metrics dict
        return best_solution, final_metrics

    except Exception as e:
        print(f"!!! ERROR running {algorithm} on {os.path.basename(dataset_path)}: {e} !!!")
        traceback.print_exc()
        return None, None

def run_experiments(datasets: list, algorithms: list):
    """Runs all specified algorithms on all specified datasets."""
    all_results = {}
    total_start_time = time.time()

    # Ensure base output directory exists
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)

    # Create GA and RL subdirectories
    ga_output_dir = os.path.join(BASE_OUTPUT_DIR, "GA")
    rl_output_dir = os.path.join(BASE_OUTPUT_DIR, "RL")
    if not os.path.exists(ga_output_dir):
        os.makedirs(ga_output_dir)
    if not os.path.exists(rl_output_dir):
        os.makedirs(rl_output_dir)

    for dataset_file in datasets:
        # Construct absolute path assuming datasets are in a 'data/' dir sibling to 'algorithms/'
        dataset_path = os.path.join(project_root, 'data', dataset_file)
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset file not found at {dataset_path}. Skipping.")
            continue

        print(f"\n{'='*20} RUNNING EXPERIMENTS FOR DATASET: {dataset_file} {'='*20}")
        dataset_name_short = os.path.splitext(dataset_file)[0]
        dataset_results = {}

        for algorithm in algorithms:
            print(f"\n--- Running Algorithm: {algorithm} --- ")
            algo_start_time = time.time()

            # Determine output directory and parameters based on algorithm type
            current_output_base = ""
            run_params = {}
            if algorithm in ['nsga2', 'spea2', 'moead']:
                current_output_base = ga_output_dir
                run_params = {
                    'population': DEFAULT_POPULATION,
                    'generations': DEFAULT_GENERATIONS,
                }
            elif algorithm in ['dqn', 'sarsa', 'implicit_q']:
                current_output_base = rl_output_dir
                run_params = {
                    'episodes': DEFAULT_EPISODES,
                    'epsilon': DEFAULT_EPSILON
                }
                if algorithm != 'implicit_q':
                    run_params['learning_rate'] = DEFAULT_LEARNING_RATE
            else:
                 print(f"Warning: Algorithm type unknown for {algorithm}. Using default output.")
                 current_output_base = BASE_OUTPUT_DIR

            # Run the algorithm
            best_solution, final_metrics = run_optimization_algorithm(
                algorithm=algorithm,
                dataset_path=dataset_path,
                output_dir=current_output_base,
                **run_params
            )

            algo_end_time = time.time()
            run_time = algo_end_time - algo_start_time

            if final_metrics is not None:
                print(f"--- Finished {algorithm} in {run_time:.2f} seconds --- ")
                dataset_results[algorithm] = final_metrics # Store only metrics

                # Save individual metrics result to file
                # Example path: output/GA/results_nsga2_sliit_computing_dataset.json
                output_filename = os.path.join(current_output_base, f"results_{algorithm}_{dataset_name_short}.json")
                try:
                    with open(output_filename, 'w') as f:
                        # Use default=str to handle potential non-serializable numpy types
                        json.dump(final_metrics, f, indent=4, default=str)
                    print(f"Metrics saved to {output_filename}")
                except Exception as e:
                    print(f"Error saving metrics for {algorithm} on {dataset_name_short}: {e}")
            else:
                 print(f"--- ERROR running {algorithm} after {run_time:.2f} seconds --- ")
                 dataset_results[algorithm] = {"error": f"Algorithm failed to return metrics for {dataset_name_short}"}

        all_results[dataset_name_short] = dataset_results

    total_end_time = time.time()
    print(f"\n{'='*20} ALL EXPERIMENTS COMPLETED {'='*20}")
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")

    # Save aggregated results (optional)
    agg_filename = os.path.join(BASE_OUTPUT_DIR, "all_experiment_metrics.json")
    try:
         with open(agg_filename, 'w') as f:
             json.dump(all_results, f, indent=4, default=str)
         print(f"Aggregated metrics saved to {agg_filename}")
    except Exception as e:
         print(f"Error saving aggregated metrics: {e}")

    return all_results

# Main execution block
if __name__ == "__main__":
    # Define experiments (could be loaded from config)
    # Assumes dataset files are in a 'data/' directory at the project root
    DATASETS_TO_RUN = ['sliit_computing_dataset.json', 'sliit_computing_dataset_7.json']
    ALGORITHMS_TO_RUN = ['nsga2', 'spea2', 'moead'] # Focus on GAs for now based on user goal
    # ALGORITHMS_TO_RUN = ['nsga2', 'spea2', 'moead', 'dqn', 'sarsa', 'implicit_q']

    run_experiments(datasets=DATASETS_TO_RUN, algorithms=ALGORITHMS_TO_RUN)

# --- Deprecated/Removed Code Sections from old runner.py ---
# - Old metric calculations (room utilization, satisfaction, time efficiency)
#   -> These should be handled by dedicated metrics/analysis modules if needed.
# - Old JSON saving logic within run_optimization_algorithm
#   -> Now handled by the run_experiments orchestrator.
# - Old evaluate_timetable call within run_optimization_algorithm
#   -> Simplified, final evaluation added to metrics.
# - Old specific output dir logic based on hardcoded paths.
#   -> Now relative paths based on BASE_OUTPUT_DIR.