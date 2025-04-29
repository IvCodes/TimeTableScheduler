"""
Run a single algorithm experiment on both datasets to verify the process works.
"""

import os
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from algorithms.run_experiments import run_algorithm
from algorithms.plotting.reviewer_plots import generate_all_paper_plots

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_4ROOM = str(PROJECT_ROOT / "Dataset/sliit_computing_dataset.json")
DATASET_7ROOM = str(PROJECT_ROOT / "Dataset/sliit_computing_dataset_7.json")
OUTPUT_4ROOM = str(PROJECT_ROOT / "output")
OUTPUT_7ROOM = str(PROJECT_ROOT / "output_7room")
FIGURES_DIR = str(PROJECT_ROOT / "paper_figures")

# NSGA-II parameters
params = {
    'population_size': 50,   # Reduced from 100 for faster testing
    'num_generations': 20,   # Reduced from 50 for faster testing
    'crossover_rate': 0.8,
    'mutation_rate': 0.1
}

def main():
    """Run NSGA-II on both datasets and generate plots."""
    start_time = time.time()
    
    print("Running NSGA-II experiment on 4-room dataset...")
    # Create output directory
    os.makedirs(OUTPUT_4ROOM, exist_ok=True)
    run_algorithm('nsga2', DATASET_4ROOM, OUTPUT_4ROOM, params)
    
    print("\nRunning NSGA-II experiment on 7-room dataset...")
    # Create output directory
    os.makedirs(OUTPUT_7ROOM, exist_ok=True)
    run_algorithm('nsga2', DATASET_7ROOM, OUTPUT_7ROOM, params)
    
    print("\nGenerating visualizations...")
    generate_all_paper_plots(
        four_room_dir=OUTPUT_4ROOM,
        seven_room_dir=OUTPUT_7ROOM,
        output_dir=FIGURES_DIR
    )
    
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\nExperiment completed in {int(minutes)}m {int(seconds)}s")
    print(f"Visualizations saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main()
