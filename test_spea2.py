"""
Test script to run just SPEA2 with the correct parameters.
"""

import os
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from algorithms.run_experiments import run_algorithm

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_4ROOM = str(PROJECT_ROOT / "Dataset/sliit_computing_dataset.json")
OUTPUT_DIR = str(PROJECT_ROOT / "output_spea2_test")

# SPEA2 correct parameters
params = {
    'population_size': 100,
    'generations': 50,
    'enable_plotting': False,
    'output_dir': OUTPUT_DIR
}

def main():
    """Run SPEA2 with correct parameters."""
    start_time = time.time()
    
    print(f"Running SPEA2 with parameters: {params}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    result = run_algorithm('spea2', DATASET_4ROOM, OUTPUT_DIR, params)
    
    if result:
        print("SPEA2 experiment completed successfully!")
    else:
        print("Error running SPEA2 experiment.")
    
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Execution time: {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()
