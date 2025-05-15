"""
Run All Experiments Script for Timetable Scheduling

This script automates the execution of all algorithms (GA, RL, CO) on both datasets
and generates visualizations for analysis.
"""

import os
import argparse
import subprocess
import time
import json
from pathlib import Path

# Define constants
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GENETIC_DIR = os.path.join(PROJECT_ROOT, "1. Genetic_EAs")
RL_DIR = os.path.join(PROJECT_ROOT, "2. Reinforcement_Learning")
CO_DIR = os.path.join(PROJECT_ROOT, "3. Colony_Optimization")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "Plots")

# Ensure plot directories exist
for subdir in ["GA", "RL", "CO", "Dataset_Analysis"]:
    os.makedirs(os.path.join(PLOTS_DIR, subdir), exist_ok=True)

def run_genetic_algorithms(dataset="4room", generations=50, pop_size=100, algorithm="all"):
    """Run genetic algorithms on the specified dataset."""
    print(f"\n{'='*80}")
    print(f"Running Genetic Algorithms on {dataset} dataset")
    print(f"{'='*80}")
    
    # Construct the command
    cmd = [
        "python", os.path.join(GENETIC_DIR, "ea_algorithm.py"),
        "--dataset", dataset,
        "--algorithm", algorithm,
        "--generations", str(generations),
        "--pop-size", str(pop_size),
        "--save-plots"
    ]
    
    # Run the command
    start_time = time.time()
    process = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    end_time = time.time()
    
    # Print output
    print(process.stdout)
    if process.stderr:
        print("Errors:", process.stderr)
    
    print(f"Genetic Algorithms completed in {end_time - start_time:.2f} seconds")
    return process.returncode == 0

def run_reinforcement_learning(dataset="4room", episodes=2000, algorithm="all"):
    """Run reinforcement learning algorithms on the specified dataset."""
    print(f"\n{'='*80}")
    print(f"Running Reinforcement Learning Algorithms on {dataset} dataset")
    print(f"{'='*80}")
    
    # Check if RL directory exists
    if not os.path.exists(RL_DIR):
        print(f"Warning: Reinforcement Learning directory not found at {RL_DIR}")
        print("Skipping RL algorithms...")
        return False
        
    # Look for the main RL script
    rl_script = None
    for script_name in ["RL_Script.py", "rl_script.py", "main.py", "run.py"]:
        potential_script = os.path.join(RL_DIR, script_name)
        if os.path.exists(potential_script):
            rl_script = potential_script
            break
    
    if not rl_script:
        print("Warning: No RL script found in the RL directory")
        print("Skipping RL algorithms...")
        return False
    
    # Set environment variable for dataset
    env = os.environ.copy()
    if dataset == "4room":
        env["TIMETABLE_DATASET"] = "sliit_computing_dataset_4_balanced.json"
    else:
        env["TIMETABLE_DATASET"] = "sliit_computing_dataset_7_balanced.json"
    
    # Construct the command
    cmd = [
        "python", rl_script,
        "--episodes", str(episodes),
        "--algorithm", algorithm,
        "--save-results"
    ]
    
    # Run the command
    start_time = time.time()
    try:
        process = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        end_time = time.time()
        
        # Print output
        print(process.stdout)
        if process.stderr:
            print("Errors:", process.stderr)
        
        print(f"Reinforcement Learning completed in {end_time - start_time:.2f} seconds")
        return process.returncode == 0
    except Exception as e:
        end_time = time.time()
        print(f"Error running RL algorithms: {e}")
        print(f"Reinforcement Learning failed in {end_time - start_time:.2f} seconds")
        return False

def run_colony_optimization(dataset="4room", iterations=100, algorithm="all"):
    """Run colony optimization algorithms on the specified dataset."""
    print(f"\n{'='*80}")
    print(f"Running Colony Optimization Algorithms on {dataset} dataset")
    print(f"{'='*80}")
    
    # Check if CO directory exists
    if not os.path.exists(CO_DIR):
        print(f"Warning: Colony Optimization directory not found at {CO_DIR}")
        print("Skipping CO algorithms...")
        return False
        
    # Look for the main CO script
    co_script = None
    for script_name in ["colony_optimization.py", "ant_colony.py", "main.py", "run.py"]:
        potential_script = os.path.join(CO_DIR, script_name)
        if os.path.exists(potential_script):
            co_script = potential_script
            break
    
    if not co_script:
        print("Warning: No Colony Optimization script found in the CO directory")
        print("Skipping CO algorithms...")
        return False
    
    # Set environment variable for dataset
    env = os.environ.copy()
    if dataset == "4room":
        env["TIMETABLE_DATASET"] = "sliit_computing_dataset_4_balanced.json"
    else:
        env["TIMETABLE_DATASET"] = "sliit_computing_dataset_7_balanced.json"
    
    # Construct the command
    cmd = [
        "python", co_script,
        "--iterations", str(iterations),
        "--algorithm", algorithm,
        "--save-results"
    ]
    
    # Run the command
    start_time = time.time()
    try:
        process = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        end_time = time.time()
        
        # Print output
        print(process.stdout)
        if process.stderr:
            print("Errors:", process.stderr)
        
        print(f"Colony Optimization completed in {end_time - start_time:.2f} seconds")
        return process.returncode == 0
    except Exception as e:
        end_time = time.time()
        print(f"Error running Colony Optimization algorithms: {e}")
        print(f"Colony Optimization failed in {end_time - start_time:.2f} seconds")
        return False

def run_visualizations():
    """Run all visualization scripts."""
    print(f"\n{'='*80}")
    print(f"Generating Visualizations")
    print(f"{'='*80}")
    
    # Check and run dataset bias analysis
    dataset_bias_script = os.path.join(PROJECT_ROOT, "analyze_dataset_bias.py")
    if os.path.exists(dataset_bias_script):
        print("\nRunning Dataset Bias Analysis...")
        try:
            result = subprocess.run(["python", dataset_bias_script], 
                          cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode == 0:
                print("Dataset bias analysis completed successfully")
            else:
                print("Dataset bias analysis encountered errors:")
                print(result.stderr)
        except Exception as e:
            print(f"Error running dataset bias analysis: {e}")
    else:
        print("\nSkipping Dataset Bias Analysis (script not found)")
    
    # Check and run GA Pareto front visualization
    ga_viz_script = os.path.join(PROJECT_ROOT, "visualize_ga_pareto.py")
    if os.path.exists(ga_viz_script):
        print("\nRunning GA Pareto Front Visualization...")
        try:
            result = subprocess.run(["python", ga_viz_script], 
                          cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode == 0:
                print("GA visualization completed successfully")
            else:
                print("GA visualization encountered errors:")
                print(result.stderr)
        except Exception as e:
            print(f"Error running GA visualization: {e}")
    else:
        print("\nSkipping GA Visualization (script not found)")
    
    # Check and run RL learning curve visualization
    rl_viz_script = os.path.join(PROJECT_ROOT, "visualize_rl_learning.py")
    if os.path.exists(rl_viz_script):
        print("\nRunning RL Learning Curve Visualization...")
        try:
            result = subprocess.run(["python", rl_viz_script], 
                          cwd=PROJECT_ROOT, capture_output=True, text=True)
            if result.returncode == 0:
                print("RL visualization completed successfully")
            else:
                print("RL visualization encountered errors:")
                print(result.stderr)
        except Exception as e:
            print(f"Error running RL visualization: {e}")
    else:
        print("\nSkipping RL Visualization (script not found)")
    
    print("\nVisualization process completed!")

def run_hybrid_approach(dataset="4room"):
    """Run a hybrid approach combining GA and RL."""
    print(f"\n{'='*80}")
    print(f"Running Hybrid Approach (GA -> RL) on {dataset} dataset")
    print(f"{'='*80}")
    
    # Check if RL directory exists
    if not os.path.exists(RL_DIR):
        print(f"Warning: Reinforcement Learning directory not found at {RL_DIR}")
        print("Cannot run hybrid approach without RL component")
        return False
    
    # Look for the main RL script
    rl_script = None
    for script_name in ["RL_Script.py", "rl_script.py", "main.py", "run.py"]:
        potential_script = os.path.join(RL_DIR, script_name)
        if os.path.exists(potential_script):
            rl_script = potential_script
            break
    
    if not rl_script:
        print("Warning: No RL script found in the RL directory")
        print("Cannot run hybrid approach without RL component")
        return False
    
    # First run GA to get initial solutions
    print("Step 1: Running GA to get initial solutions...")
    ga_success = run_genetic_algorithms(dataset=dataset, generations=20, pop_size=80, algorithm="nsga2")
    
    if not ga_success:
        print("GA failed, cannot continue with hybrid approach")
        return False
    
    # Find the best solution from GA
    print("Step 2: Finding best GA solution...")
    ga_results_dir = os.path.join(PROJECT_ROOT, "output", dataset, f"ga_results_{dataset}")
    
    try:
        with open(os.path.join(ga_results_dir, "pareto_front_data.json"), "r") as f:
            ga_data = json.load(f)
        
        # Get the best solution from NSGA-II
        pareto_fronts = ga_data.get("pareto_fronts", {})
        nsga2_front = pareto_fronts.get("NSGA-II", [])
        
        if not nsga2_front:
            print("No NSGA-II solutions found")
            return False
        
        # Find solution with minimum conflicts and unassigned activities
        best_solution_idx = 0
        best_score = float('inf')
        
        for i, solution in enumerate(nsga2_front):
            # Sum of conflicts and unassigned activities
            score = sum(solution[:4])  # First 4 objectives are conflicts and unassigned
            if score < best_score:
                best_score = score
                best_solution_idx = i
        
        # Save the best solution for RL to use
        best_solution_file = os.path.join(PROJECT_ROOT, "hybrid_initial_solution.json")
        with open(best_solution_file, "w") as f:
            json.dump({
                "solution": nsga2_front[best_solution_idx],
                "dataset": dataset
            }, f)
        
        print(f"Best GA solution saved with score: {best_score}")
        
        # Now run RL with the initial solution
        print("Step 3: Running RL with GA initial solution...")
        env = os.environ.copy()
        if dataset == "4room":
            env["TIMETABLE_DATASET"] = "sliit_computing_dataset.json"
        else:
            env["TIMETABLE_DATASET"] = "sliit_computing_dataset_7.json"
        
        env["INITIAL_SOLUTION"] = best_solution_file
        
        # Run RL with initial solution
        cmd = [
            "python", rl_script,
            "--episodes", "1000",
            "--algorithm", "dqn",  # DQN works best with initial solutions
            "--save-results",
            "--use-initial-solution"
        ]
        
        try:
            process = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
            
            print(process.stdout)
            if process.stderr:
                print("Errors:", process.stderr)
            
            print("Hybrid approach completed successfully!")
            return True
        except Exception as e:
            print(f"Error running RL component of hybrid approach: {e}")
            print("GA part of hybrid approach was successful, but RL failed")
            return False
        
    except Exception as e:
        print(f"Error in hybrid approach: {e}")
        return False

def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description='Run all experiments for timetable scheduling')
    parser.add_argument('--datasets', nargs='+', choices=['4room', '7room', 'all'], default=['all'],
                       help='Datasets to run experiments on')
    parser.add_argument('--algorithms', nargs='+', choices=['ga', 'rl', 'co', 'hybrid', 'all'], default=['all'],
                       help='Algorithms to run')
    parser.add_argument('--ga-generations', type=int, default=30,
                       help='Number of generations for GA')
    parser.add_argument('--ga-pop-size', type=int, default=100,
                       help='Population size for GA')
    parser.add_argument('--rl-episodes', type=int, default=2000,
                       help='Number of episodes for RL')
    parser.add_argument('--co-iterations', type=int, default=100,
                       help='Number of iterations for CO')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only run visualizations, not algorithms')
    parser.add_argument('--skip-missing', action='store_true',
                       help='Skip missing algorithms instead of showing errors')
    
    args = parser.parse_args()
    
    # Determine which datasets to run
    datasets = []
    if 'all' in args.datasets:
        datasets = ['4room', '7room']
    else:
        datasets = args.datasets
    
    # Determine which algorithms to run
    run_ga = 'all' in args.algorithms or 'ga' in args.algorithms
    run_rl = 'all' in args.algorithms or 'rl' in args.algorithms
    run_co = 'all' in args.algorithms or 'co' in args.algorithms
    run_hybrid = 'all' in args.algorithms or 'hybrid' in args.algorithms
    
    # Track overall start time
    overall_start = time.time()
    
    if args.visualize_only:
        # Only run visualizations
        run_visualizations()
    else:
        # Run algorithms on each dataset
        for dataset in datasets:
            print(f"\n{'#'*100}")
            print(f"# Starting experiments on {dataset} dataset")
            print(f"{'#'*100}")
            
            if run_ga:
                run_genetic_algorithms(dataset, args.ga_generations, args.ga_pop_size)
            
            if run_rl:
                run_reinforcement_learning(dataset, args.rl_episodes)
            
            if run_co:
                run_colony_optimization(dataset, args.co_iterations)
            
            if run_hybrid:
                run_hybrid_approach(dataset)
            
            # Run visualizations after each dataset
            run_visualizations()
    
    # Print overall execution time
    overall_end = time.time()
    hours, remainder = divmod(overall_end - overall_start, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'#'*100}")
    print(f"# All experiments completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print(f"{'#'*100}")
    
    print("\nResults and visualizations are available in the following directories:")
    print(f"- GA results: {os.path.join(PROJECT_ROOT, 'output')}")
    print(f"- Visualizations: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
