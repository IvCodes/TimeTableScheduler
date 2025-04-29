"""
Test script to verify that all import paths are correctly set up.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

# Try importing from the refactored structure
print("Testing imports...")
try:
    from algorithms.data.loader import load_data
    print("[OK] Successfully imported data.loader")
    
    from algorithms.metrics.tracker import MetricsTracker
    print("[OK] Successfully imported metrics.tracker")
    
    from algorithms.metrics.calculator import extract_pareto_front
    print("[OK] Successfully imported metrics.calculator")
    
    from algorithms.evaluation.evaluator import evaluate_timetable
    print("[OK] Successfully imported evaluation.evaluator")
    
    from algorithms.plotting.plot_utils import plot_convergence
    print("[OK] Successfully imported plotting.plot_utils")
    
    from algorithms.plotting.reviewer_plots import generate_all_paper_plots
    print("[OK] Successfully imported plotting.reviewer_plots")
    
    # Try to import at least one algorithm from each category
    from algorithms.ga.nsga2 import run_nsga2_optimizer
    print("[OK] Successfully imported ga.nsga2")
    
    print("\nAll imports successful!")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    
    # Suggest fixing the import path
    module_name = str(e).split("'")[1] if "'" in str(e) else str(e).split("No module named ")[1]
    print(f"\nSuggestion: Fix the import path for module '{module_name}'")
