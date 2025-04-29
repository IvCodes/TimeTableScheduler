"""
Script to generate all required plots for the research paper.
Addresses reviewer comments by creating Pareto fronts, learning curves, and more.
"""

import os
import sys
import json
from pathlib import Path

# Ensure algorithms directory is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import visualization modules
from algorithms.plotting.reviewer_plots import (
    generate_all_paper_plots,
    plot_pareto_front_comparison,
    plot_learning_curves,
    plot_convergence_comparison,
    plot_comparative_analysis
)

def main():
    """Main function to generate all required plots for the paper."""
    # Project root directory
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Output directories for results
    four_room_dir = project_root / "output"
    seven_room_dir = project_root / "output_7room"  # May not exist yet
    
    # Output directory for paper figures
    paper_figures_dir = project_root / "paper_figures"
    os.makedirs(paper_figures_dir, exist_ok=True)
    
    print("Generating plots for the research paper...")
    
    # Check if four_room_dir exists and has results
    if not four_room_dir.exists():
        print(f"Error: 4-room results directory {four_room_dir} does not exist.")
        print("Please run the experiments first or check the output path.")
        return
    
    # Check if GA results exist for 4-room dataset
    ga_results_dir = four_room_dir / "GA"
    if not ga_results_dir.exists() or not list(ga_results_dir.glob("*.json")):
        print(f"Warning: No GA results found in {ga_results_dir}")
    
    # Check if RL results exist for 4-room dataset
    rl_results_dir = four_room_dir / "RL"
    if not rl_results_dir.exists() or not list(rl_results_dir.glob("*.json")):
        print(f"Warning: No RL results found in {rl_results_dir}")
    
    # Generate plots for 4-room dataset
    print("\n--- Generating plots for 4-room dataset ---")
    generate_all_paper_plots(
        four_room_dir=str(four_room_dir),
        seven_room_dir=str(seven_room_dir) if seven_room_dir.exists() else None,
        output_dir=str(paper_figures_dir)
    )
    
    print("\nPlot generation complete!")
    print(f"All figures have been saved to: {paper_figures_dir}")
    
    # Instructions for the next steps
    print("\nNext steps:")
    print("1. Run the experiments on the 7-room dataset")
    print("2. Run this script again to include 7-room comparison plots")
    print("3. Insert the generated figures into your research paper")

if __name__ == "__main__":
    main()
