"""
Test script to generate a single Pareto front plot from result data.
"""

import os
import matplotlib.pyplot as plt
from algorithms.plotting.reviewer_plots import plot_pareto_front_comparison, load_result_file

# Directory paths
output_dir = 'paper_figures_test'
os.makedirs(output_dir, exist_ok=True)

# Result files to plot
result_files = {
    'NSGA-II': 'output/GA/results_nsga2_sliit_computing_dataset.json'
}

# Generate Pareto front plot
plot_pareto_front_comparison(result_files, output_dir, dataset_name='4-room')

print(f"Test plot generated in {output_dir}/pareto_front_4-room.png")

# Print actual data from result file
result_data = load_result_file('output/GA/results_nsga2_sliit_computing_dataset.json')
if result_data and 'final_pareto_front' in result_data:
    print(f"Number of Pareto points found: {len(result_data['final_pareto_front'])}")
    # Print a few example points to verify
    print("Example Pareto points:")
    for i, point in enumerate(result_data['final_pareto_front'][:5]):
        print(f"  Point {i+1}: Hard violations={point[0]}, Soft score={point[1]}")
else:
    print("No Pareto front data found in result file")
