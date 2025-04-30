"""
Comprehensive experiment runner for generating publication-quality results
for the paper "Evaluating Genetic, Reinforcement Learning, and Colony Optimization 
Algorithms for Scalable University Timetable Scheduling".

This script runs experiments with multiple algorithms on both 4-room and 7-room datasets
with parameters configured for proper convergence and meaningful comparisons.
"""

import os
import sys
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from algorithms.run_experiments import run_algorithm
from algorithms.plotting.reviewer_plots import generate_all_paper_plots

# Define experiment parameters
GA_PARAMS = {
    'nsga2': {
        'population_size': 20,  # Increased from 25
        'num_generations': 10,   # Increased from 10
        'crossover_rate': 0.8,
        'mutation_rate': 0.1
    },
    'spea2': {
        'population_size': 10,
        'num_generations': 20,       # Renamed from 'generations'
        'enable_plotting': False, # Additional parameter needed
        'output_dir': 'output/GA'  # Adding output_dir parameter which is actually accepted
    },
    'moead': {
        'population_size': 20,
        'num_generations': 10
    }
}

RL_PARAMS = {
    'dqn': {
        'episodes': 20,
        'epsilon': 0.1,
        'learning_rate': 0.001
    },
    'sarsa': {
        'episodes': 20,
        'learning_rate': 0.1,       # Renamed from 'alpha'
        'epsilon': 0.1
    },
    'qlearning': {
        'episodes': 20,
        'epsilon': 0.1
    }
}

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_4ROOM = str(PROJECT_ROOT / "Dataset/sliit_computing_dataset.json")
DATASET_7ROOM = str(PROJECT_ROOT / "Dataset/sliit_computing_dataset_7.json")
OUTPUT_4ROOM = str(PROJECT_ROOT / "output")
OUTPUT_7ROOM = str(PROJECT_ROOT / "output_7room")
FIGURES_DIR = str(PROJECT_ROOT / "paper_figures")

def run_all_ga_experiments():
    """Run all genetic algorithm experiments for both datasets."""
    print("\n" + "="*80)
    print("RUNNING GENETIC ALGORITHM EXPERIMENTS")
    print("="*80)
    
    # Create output directories
    os.makedirs(OUTPUT_4ROOM, exist_ok=True)
    os.makedirs(OUTPUT_7ROOM, exist_ok=True)
    
    # Run GA algorithms on 4-room dataset
    print("\nRunning GA algorithms on 4-room dataset:")
    for algorithm, params in GA_PARAMS.items():
        print(f"\n{algorithm.upper()} with parameters: {params}")
        result = run_algorithm(algorithm, DATASET_4ROOM, OUTPUT_4ROOM, params)
        if result:
            print(f"Experiment completed successfully. Results saved.")
        else:
            print(f"Error running {algorithm} on 4-room dataset.")
    
    # Run GA algorithms on 7-room dataset
    print("\nRunning GA algorithms on 7-room dataset:")
    for algorithm, params in GA_PARAMS.items():
        print(f"\n{algorithm.upper()} with parameters: {params}")
        result = run_algorithm(algorithm, DATASET_7ROOM, OUTPUT_7ROOM, params)
        if result:
            print(f"Experiment completed successfully. Results saved.")
        else:
            print(f"Error running {algorithm} on 7-room dataset.")

def run_all_rl_experiments():
    """Run all reinforcement learning experiments for both datasets."""
    print("\n" + "="*80)
    print("RUNNING REINFORCEMENT LEARNING EXPERIMENTS")
    print("="*80)
    
    # Run RL algorithms on 4-room dataset
    print("\nRunning RL algorithms on 4-room dataset:")
    for algorithm, params in RL_PARAMS.items():
        print(f"\n{algorithm.upper()} with parameters: {params}")
        result = run_algorithm(algorithm, DATASET_4ROOM, OUTPUT_4ROOM, params)
        if result:
            print(f"Experiment completed successfully. Results saved.")
        else:
            print(f"Error running {algorithm} on 4-room dataset.")
    
    # Run RL algorithms on 7-room dataset
    print("\nRunning RL algorithms on 7-room dataset:")
    for algorithm, params in RL_PARAMS.items():
        print(f"\n{algorithm.upper()} with parameters: {params}")
        result = run_algorithm(algorithm, DATASET_7ROOM, OUTPUT_7ROOM, params)
        if result:
            print(f"Experiment completed successfully. Results saved.")
        else:
            print(f"Error running {algorithm} on 7-room dataset.")

def generate_visualizations():
    """Generate all visualizations for the paper."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    generate_all_paper_plots(
        four_room_dir=OUTPUT_4ROOM,
        seven_room_dir=OUTPUT_7ROOM,
        output_dir=FIGURES_DIR
    )
    
    print(f"\nAll visualizations saved to {FIGURES_DIR}")

def main():
    """Main function to run all experiments and generate visualizations."""
    start_time = time.time()
    
    print("Starting comprehensive experiments for the research paper...")
    
    # Run GA experiments
    run_all_ga_experiments()
    
   # Run RL experiments (Commented out for GA debugging)
    run_all_rl_experiments()
    
    #Generate visualizations (Commented out for GA debugging)
    generate_visualizations()
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Visualizations saved to: {FIGURES_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
