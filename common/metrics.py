"""
Common metrics module for timetable scheduling algorithms.
Provides performance metrics calculations for evaluating and comparing algorithms.
"""

import numpy as np
import time
import psutil
from typing import List, Dict, Tuple, Any, Optional
from scipy.spatial import ConvexHull

# Constants for objective labels - used in visualization and reporting
OBJECTIVE_LABELS = [
    "Professor Conflicts",
    "Group Conflicts",
    "Room Conflicts",
    "Unassigned Activities",
    "Soft Constraints",
    "Student Fatigue",
    "Student Idle Time",
    "Lecturer Fatigue",
    "Workload Balance"
]

# Constants for frequently used objective strings 
PROFESSOR_CONFLICTS = "Professor Conflicts"
GROUP_CONFLICTS = "Group Conflicts"
ROOM_CONFLICTS = "Room Conflicts"
UNASSIGNED_ACTIVITIES = "Unassigned Activities"
SOFT_CONSTRAINTS = "Soft Constraints"

def calculate_hypervolume(pareto_front, reference_point):
    """
    Calculate the hypervolume indicator for a Pareto front.
    
    A higher hypervolume indicates a better Pareto front (better coverage, diversity, and convergence).
    
    Args:
        pareto_front: List of objective vectors (each a tuple of objective values)
        reference_point: Reference point (worst values for each objective)
        
    Returns:
        float: Hypervolume value
    """
    if not pareto_front or len(pareto_front) < 3:  # Need at least 3 points for ConvexHull
        # Fallback for small fronts: approximate with rectangular volumes
        return simple_hypervolume(pareto_front, reference_point)
    
    try:
        # Convert to numpy array for hull calculation
        points = np.array(pareto_front)
        # Add reference point to create closed volume
        points_with_ref = np.vstack([points, reference_point])
        hull = ConvexHull(points_with_ref)
        return hull.volume
    except Exception as e:
        print(f"Error calculating hypervolume: {e}")
        return simple_hypervolume(pareto_front, reference_point)


def simple_hypervolume(pareto_front, reference_point):
    """
    Simplified hypervolume calculation for small fronts or when ConvexHull fails.
    
    Args:
        pareto_front: List of objective vectors (each a tuple of objective values)
        reference_point: Reference point (worst values for each objective)
        
    Returns:
        float: Approximate hypervolume value
    """
    if not pareto_front:
        return 0.0
    
    # Check if solutions are dictionaries or lists
    is_dict = isinstance(pareto_front[0], dict) if pareto_front else False
    
    total_volume = 0.0
    for point in pareto_front:
        # Calculate volume based on solution type
        if is_dict:
            volume = _calculate_dict_volume(point, reference_point)
        else:
            volume = _calculate_list_volume(point, reference_point)
        
        total_volume += volume
    return total_volume


def _calculate_dict_volume(point, reference_point):
    """Calculate the volume contribution for a dictionary-based solution."""
    volume = 1.0
    objectives = ['conflicts', 'utilization', 'preferences', 'quality', 'diversity']
    
    for i, obj in enumerate(objectives):
        if i >= len(reference_point):  # Skip if beyond reference point dimensions
            continue
            
        value = point.get(obj, 0)
        
        # For minimization objectives (conflicts), use standard comparison
        if obj == 'conflicts':
            volume *= max(0, reference_point[i] - value)
        else:  # For maximization objectives, invert the comparison
            volume *= max(0, value - reference_point[i])
    
    return volume


def _calculate_list_volume(point, reference_point):
    """Calculate the volume contribution for a list-based solution."""
    volume = 1.0
    
    for i, value in enumerate(point):
        if i < len(reference_point):  # Ensure we don't go out of bounds
            volume *= max(0, reference_point[i] - value)
    
    return volume


def calculate_igd(pareto_front, reference_front):
    """
    Calculate Inverted Generational Distance (IGD).
    
    Lower IGD indicates a Pareto front that is closer to the reference front.
    
    Args:
        pareto_front: List of objective vectors from algorithm
        reference_front: True Pareto front or best-known front
        
    Returns:
        float: IGD value
    """
    if not pareto_front or not reference_front:
        return float('inf')
        
    total_distance = 0.0
    
    for ref_point in reference_front:
        # Find minimum distance from ref_point to any point in pareto_front
        min_distance = float('inf')
        for point in pareto_front:
            # Euclidean distance
            distance = np.sqrt(sum((r - p) ** 2 for r, p in zip(ref_point, point)))
            min_distance = min(min_distance, distance)
        total_distance += min_distance
        
    return total_distance / len(reference_front)


def calculate_gd(pareto_front, reference_front):
    """
    Calculate Generational Distance (GD).
    
    Lower GD indicates a Pareto front that is closer to the reference front.
    
    Args:
        pareto_front: List of objective vectors from algorithm
        reference_front: True Pareto front or best-known front
        
    Returns:
        float: GD value
    """
    if not pareto_front or not reference_front:
        return float('inf')
        
    total_distance = 0.0
    
    for point in pareto_front:
        # Find minimum distance from point to any reference point
        min_distance = float('inf')
        for ref_point in reference_front:
            # Euclidean distance
            distance = np.sqrt(sum((p - r) ** 2 for p, r in zip(point, ref_point)))
            min_distance = min(min_distance, distance)
        total_distance += min_distance ** 2
        
    return np.sqrt(total_distance / len(pareto_front))


def calculate_spread(pareto_front):
    """
    Calculate spread (diversity) of the Pareto front.
    
    Lower spread value indicates more uniform distribution of solutions.
    
    Args:
        pareto_front: List of objective vectors
        
    Returns:
        float: Spread value
    """
    if len(pareto_front) < 2:
        return 0.0
        
    # Sort by first objective
    sorted_front = sorted(pareto_front, key=lambda x: x[0])
    
    # Calculate distances between adjacent solutions
    distances = []
    for i in range(len(sorted_front) - 1):
        dist = np.sqrt(sum((sorted_front[i][j] - sorted_front[i+1][j]) ** 2 
                          for j in range(len(sorted_front[i]))))
        distances.append(dist)
    
    # Calculate mean distance
    if not distances:
        return 0.0
        
    mean_dist = np.mean(distances)
    
    # Calculate standard deviation
    std_dev = np.std(distances)
    
    # Return normalized spread
    return std_dev / mean_dist if mean_dist > 0 else 0.0


def calculate_coverage(pareto_front_a, pareto_front_b):
    """
    Calculate coverage metric (how many points in B are dominated by points in A).
    
    Higher value means A dominates more of B.
    
    Args:
        pareto_front_a: First Pareto front
        pareto_front_b: Second Pareto front
        
    Returns:
        float: Coverage metric (0 to 1)
    """
    if not pareto_front_a or not pareto_front_b:
        return 0.0
    
    dominated_count = 0
    for b in pareto_front_b:
        if any(dominates(a, b) for a in pareto_front_a):
            dominated_count += 1
    
    return dominated_count / len(pareto_front_b)


def dominates(a, b):
    """
    Check if solution a dominates solution b (for minimization problems).
    
    Args:
        a: First solution (tuple of objective values)
        b: Second solution (tuple of objective values)
        
    Returns:
        bool: True if a dominates b
    """
    better_in_one = False
    for i in range(len(a)):
        if a[i] > b[i]:  # a is worse in any objective
            return False
        if a[i] < b[i]:  # a is better in at least one objective
            better_in_one = True
    return better_in_one


def calculate_convergence_speed(history, reference_value, max_iterations):
    """
    Calculate convergence speed metric.
    
    Higher value indicates faster convergence to good solutions.
    
    Args:
        history: List of best fitness values per iteration
        reference_value: Best known or ideal fitness value
        max_iterations: Maximum iterations algorithm could run
        
    Returns:
        float: Convergence speed metric (0 to 1)
    """
    if not history:
        return 0.0
    
    # Normalize history values relative to reference
    normalized_history = [abs(h - reference_value) for h in history]
    
    # Calculate area under convergence curve (smaller is better)
    auc = np.trapz(normalized_history, dx=1.0/len(normalized_history))
    
    # Calculate iterations to reach 90% of final value
    final_value = normalized_history[-1]
    threshold = final_value * 1.1  # 110% of final (allowing some margin)
    
    for i, value in enumerate(normalized_history):
        if value <= threshold:
            iteration_ratio = i / max_iterations
            break
    else:
        iteration_ratio = 1.0
    
    # Combine metrics (lower is better, so invert)
    convergence_speed = 1.0 - (0.5 * auc + 0.5 * iteration_ratio)
    
    return max(0.0, min(1.0, convergence_speed))  # Clamp to [0,1]


def calculate_resource_efficiency(execution_time, peak_memory, solution_quality, baseline_time=None, baseline_memory=None):
    """
    Calculate resource efficiency metric.
    
    Higher value indicates better efficiency relative to quality.
    
    Args:
        execution_time: Algorithm execution time in seconds
        peak_memory: Peak memory usage in MB
        solution_quality: Quality metric of the solution (higher is better)
        baseline_time: Reference execution time (optional)
        baseline_memory: Reference memory usage (optional)
        
    Returns:
        float: Resource efficiency metric (0 to 1)
    """
    # Normalize inputs if baselines provided
    if baseline_time and baseline_memory:
        norm_time = min(1.0, baseline_time / max(0.001, execution_time))
        norm_memory = min(1.0, baseline_memory / max(0.001, peak_memory))
    else:
        # Simple scaling for standalone evaluation
        norm_time = min(1.0, 100.0 / max(0.001, execution_time))
        norm_memory = min(1.0, 1000.0 / max(0.001, peak_memory))
    
    # Combine with solution quality
    resource_efficiency = 0.4 * norm_time + 0.2 * norm_memory + 0.4 * solution_quality
    
    return max(0.0, min(1.0, resource_efficiency))  # Clamp to [0,1]
