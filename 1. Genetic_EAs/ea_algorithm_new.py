"""Evolutionary Algorithm implementation for timetable scheduling optimization.

This module implements multiple evolutionary algorithms (NSGA-II, SPEA2, MOEA/D) 
for solving the University Timetabling Problem with multi-objective optimization.
"""

import os
import argparse
import copy
import json
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import itertools
from scipy import stats

# Import from common modules
from common.data_loader import Activity, Group, Space, Lecturer, load_data, initialize_empty_schedule, is_space_suitable
from common.metrics import calculate_hypervolume, simple_hypervolume, calculate_igd, calculate_gd, calculate_spread
from common.metrics import dominates, calculate_convergence_speed, calculate_coverage, OBJECTIVE_LABELS
from common.evaluator import multi_objective_evaluator, evaluate_schedule, hard_constraint_lecturer_clash, hard_constraint_group_clash

# === Constants ===
NUM_OBJECTIVES_GA = 5  # Define number of objectives for GA variants

# Global dictionary to store resource metrics for each algorithm run
ALGORITHM_RESOURCE_METRICS = {}

# Function to track computational resources
def track_computational_resources(algorithm_func):
    """
    Decorator to track computational resources used by algorithms.
    
    Args:
        algorithm_func: Function implementing an evolutionary algorithm
        
    Returns:
        Wrapped function that tracks execution time and memory usage
    """
    def wrapper(*args, **kwargs):
        # Get algorithm name
        func_name = algorithm_func.__name__.upper()
        
        # Start tracking memory usage (in MB)
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024
        
        # Start tracking execution time
        start_time = time.time()
        
        # Run the algorithm
        result = algorithm_func(*args, **kwargs)
        
        # Measure execution time
        execution_time = time.time() - start_time
        
        # Measure memory usage (in MB)
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        # Store metrics
        ALGORITHM_RESOURCE_METRICS[func_name] = {
            'execution_time': execution_time,
            'memory_usage': memory_usage
        }
        
        return result
    return wrapper


# === Generate Initial Population ===

def generate_initial_population(pop_size, slots, activities_dict, spaces_dict, groups_dict_local):
    """
    Generate an initial population using a more aggressive approach to maximize scheduling.
    
    Args:
        pop_size: Population size
        slots: List of available time slots
        activities_dict: Dictionary of activities
        spaces_dict: Dictionary of spaces
        groups_dict_local: Dictionary of groups
        
    Returns:
        list: Initial population of timetables
    """
    population = []
    
    for _ in range(pop_size):
        # Initialize empty timetable
        timetable = {slot: {space_id: None for space_id in spaces_dict} for slot in slots}
        
        # Create a list of activities to schedule (shuffled)
        activities_to_schedule = list(activities_dict.keys())
        random.shuffle(activities_to_schedule)
        
        # Create a list of slots (shuffled for each activity)
        slot_list = list(slots)
        
        # Track scheduled activities to avoid duplicates
        scheduled_activities = set()
        
        # Track assignment conflicts
        lecturer_assignments = {slot: set() for slot in slots}
        group_assignments = {slot: set() for slot in slots}
        
        # Process activities greedily with heuristic guidance
        for activity_id in activities_to_schedule:
            if activity_id in scheduled_activities:
                continue
                
            activity = activities_dict[activity_id]
            
            # Create a shuffled list of slots
            random.shuffle(slot_list)
            
            # Try to find a suitable slot and space for this activity
            scheduled = False
            
            for slot in slot_list:
                # Skip if lecturer already has an activity in this slot
                if activity.lecturer_id and activity.lecturer_id in lecturer_assignments[slot]:
                    continue
                    
                # Skip if any group has an activity in this slot
                if any(group_id in group_assignments[slot] for group_id in activity.group_ids):
                    continue
                
                # Get class size
                class_size = sum(groups_dict_local.get(group_id, Group(id=group_id, size=0)).size 
                                for group_id in activity.group_ids)
                
                # Try to find a suitable space
                space_ids = list(spaces_dict.keys())
                random.shuffle(space_ids)  # Randomize space selection
                
                for space_id in space_ids:
                    space = spaces_dict[space_id]
                    
                    # Check if space is available in this slot
                    if timetable[slot][space_id] is not None:
                        continue
                        
                    # Check if space is large enough
                    if space.capacity < class_size:
                        continue
                    
                    # Schedule the activity
                    timetable[slot][space_id] = activity_id
                    scheduled_activities.add(activity_id)
                    
                    # Update conflict trackers
                    if activity.lecturer_id:
                        lecturer_assignments[slot].add(activity.lecturer_id)
                    for group_id in activity.group_ids:
                        group_assignments[slot].add(group_id)
                        
                    scheduled = True
                    break
                    
                if scheduled:
                    break
        
        population.append(timetable)
    
    return population


# === Crossover ===

def crossover(parent1, parent2):
    """
    Perform crossover by swapping time slots between two parents.
    
    Args:
        parent1: First parent timetable
        parent2: Second parent timetable
        
    Returns:
        tuple: Two offspring timetables
    """
    # Create deep copies to avoid modifying parents
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)
    
    # Get list of slots
    slots = list(parent1.keys())
    
    # Select a random crossover point
    crossover_point = random.randint(1, len(slots) - 1)
    
    # Swap slots after crossover point
    for i in range(crossover_point, len(slots)):
        slot = slots[i]
        offspring1[slot], offspring2[slot] = offspring2[slot], offspring1[slot]
    
    return offspring1, offspring2


# === Mutation ===

def mutate(individual, activities_dict_local, slots_local, spaces_dict_local):
    """
    Perform enhanced mutation strategies to improve scheduling rate.
    
    Args:
        individual: Timetable to mutate
        activities_dict_local: Dictionary of activities
        slots_local: List of available time slots
        spaces_dict_local: Dictionary of spaces
        
    Returns:
        dict: Mutated timetable
    """
    # Create a deep copy to avoid modifying the original
    mutated = copy.deepcopy(individual)
    
    # Get all scheduled activities
    scheduled_activities = set()
    activity_locations = {}  # Maps activity_id to (slot, space_id)
    
    for slot, spaces in mutated.items():
        for space_id, activity_id in spaces.items():
            if activity_id is not None:
                scheduled_activities.add(activity_id)
                activity_locations[activity_id] = (slot, space_id)
    
    # Mutation strategy 1: Move a randomly selected activity to a new time/space
    if scheduled_activities and random.random() < 0.5:
        # Select a random activity to move
        activity_id = random.choice(list(scheduled_activities))
        old_slot, old_space_id = activity_locations[activity_id]
        
        # Remove from current location
        mutated[old_slot][old_space_id] = None
        
        # Try to find a new location
        new_slot = random.choice(slots_local)
        new_space_id = random.choice(list(spaces_dict_local.keys()))
        
        # Check if new location is free
        if mutated[new_slot][new_space_id] is None:
            mutated[new_slot][new_space_id] = activity_id
        else:
            # If not free, swap with the activity at the new location
            other_activity_id = mutated[new_slot][new_space_id]
            mutated[new_slot][new_space_id] = activity_id
            mutated[old_slot][old_space_id] = other_activity_id
    
    # Mutation strategy 2: Try to schedule an unscheduled activity
    unscheduled = set(activities_dict_local.keys()) - scheduled_activities
    if unscheduled and random.random() < 0.3:
        activity_id = random.choice(list(unscheduled))
        
        # Try to find a free slot and space
        random_slots = list(slots_local)
        random.shuffle(random_slots)
        
        for slot in random_slots:
            random_spaces = list(spaces_dict_local.keys())
            random.shuffle(random_spaces)
            
            for space_id in random_spaces:
                if mutated[slot][space_id] is None:
                    # Found a free slot and space
                    mutated[slot][space_id] = activity_id
                    return mutated
    
    # Mutation strategy 3: Swap two activities
    if len(scheduled_activities) >= 2 and random.random() < 0.2:
        # Select two random activities
        activities = list(scheduled_activities)
        act1, act2 = random.sample(activities, 2)
        
        # Get their locations
        slot1, space1 = activity_locations[act1]
        slot2, space2 = activity_locations[act2]
        
        # Swap them
        mutated[slot1][space1] = act2
        mutated[slot2][space2] = act1
    
    return mutated


# === Non-dominated Sorting ===

def non_dominated_sort(population_indices, fitness_values):
    """
    Sort population indices into fronts based on dominance.
    
    Args:
        population_indices: List of indices in the population
        fitness_values: List of fitness tuples
        
    Returns:
        list: List of fronts, where each front is a list of indices
    """
    fronts = []
    remaining = set(population_indices)
    
    while remaining:
        # Find non-dominated individuals in the remaining set
        front = []
        
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and dominates(fitness_values[j], fitness_values[i]):
                    dominated = True
                    break
            
            if not dominated:
                front.append(i)
        
        # Add front to fronts and remove from remaining
        fronts.append(front)
        remaining -= set(front)
    
    return fronts


# === Crowding Distance ===

def crowding_distance(fitness_values, front_indices):
    """
    Calculate crowding distance for solutions in a front (list of indices).
    
    Args:
        fitness_values: List of fitness tuples
        front_indices: List of indices in the front
        
    Returns:
        dict: Dictionary mapping indices to crowding distances
    """
    n = len(front_indices)
    distances = {idx: 0.0 for idx in front_indices}
    
    if n <= 2:
        # If front has two or fewer points, set maximum distance
        for idx in front_indices:
            distances[idx] = float('inf')
        return distances
        
    # Calculate crowding distance for each objective
    num_objectives = len(fitness_values[0])
    
    for obj in range(num_objectives):
        # Sort front by this objective
        sorted_front = sorted(front_indices, key=lambda idx: fitness_values[idx][obj])
        
        # Set infinite distance for extreme points
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')
        
        # Calculate distances for non-extreme points
        if n > 2:
            # Get objective value range
            obj_range = (
                fitness_values[sorted_front[-1]][obj] - 
                fitness_values[sorted_front[0]][obj]
            )
            
            if obj_range > 0:
                for i in range(1, n-1):
                    distances[sorted_front[i]] += (
                        (fitness_values[sorted_front[i+1]][obj] - 
                         fitness_values[sorted_front[i-1]][obj]) / 
                        obj_range
                    )
    
    return distances


# === Selection (NSGA-II) ===

def selection(population, fitness_values, pop_size):
    """
    Select parents for the next generation using NSGA-II logic.
    
    Args:
        population: Current population
        fitness_values: List of fitness tuples
        pop_size: Population size to select
        
    Returns:
        list: Selected population
    """
    indices = list(range(len(population)))
    
    # Non-dominated sorting
    fronts = non_dominated_sort(indices, fitness_values)
    
    # Select individuals from fronts
    selected_indices = []
    for front in fronts:
        # If adding this front exceeds the population size, use crowding distance
        if len(selected_indices) + len(front) > pop_size:
            # Calculate crowding distance
            distances = crowding_distance(fitness_values, front)
            
            # Sort front by crowding distance (descending)
            sorted_front = sorted(front, key=lambda idx: -distances[idx])
            
            # Add individuals until population size is reached
            needed = pop_size - len(selected_indices)
            selected_indices.extend(sorted_front[:needed])
            break
        else:
            # Add the entire front
            selected_indices.extend(front)
    
    # Select the individuals from the population
    selected_population = [population[idx] for idx in selected_indices]
    
    return selected_population


# === NSGA-II Main Loop ===

@track_computational_resources
def nsga2(pop_size, generations, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots):
    """
    Main NSGA-II algorithm with improved scheduling prioritization.
    
    Args:
        pop_size: Population size
        generations: Number of generations
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of groups
        lecturers_dict: Dictionary of lecturers
        spaces_dict: Dictionary of spaces
        slots: List of available time slots
        
    Returns:
        tuple: (final_population, fitness_values, pareto_indices, best_fitness_history)
    """
    # Initialize population with aggressive scheduling
    population = generate_initial_population(pop_size, slots, activities_dict, spaces_dict, groups_dict)
    
    # Track best solutions and convergence
    best_fitness_history = []
    
    # Main evolutionary loop
    for generation in range(generations):
        # Evaluate population
        fitness_values = [
            multi_objective_evaluator(individual, activities_dict, groups_dict, 
                                    lecturers_dict, spaces_dict, slots)
            for individual in population
        ]
        
        # Track best solution (minimum sum of objectives)
        fitness_sums = [sum(fitness) for fitness in fitness_values]
        best_idx = fitness_sums.index(min(fitness_sums))
        best_fitness_history.append(fitness_values[best_idx])
        
        # Print progress (every 10 generations)
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {fitness_values[best_idx]}")
        
        # Stop if this is the last generation
        if generation == generations - 1:
            break
            
        # Create offspring population
        offspring = []
        
        # Tournament selection, crossover, and mutation
        while len(offspring) < pop_size:
            # Tournament selection
            tournament_size = 3
            parents = []
            
            for _ in range(2):
                # Select tournament candidates
                candidates = random.sample(range(pop_size), tournament_size)
                
                # Find best candidate
                best_candidate = candidates[0]
                for candidate in candidates[1:]:
                    if sum(fitness_values[candidate]) < sum(fitness_values[best_candidate]):
                        best_candidate = candidate
                
                parents.append(population[best_candidate])
            
            # Crossover
            child1, child2 = crossover(parents[0], parents[1])
            
            # Mutation
            child1 = mutate(child1, activities_dict, slots, spaces_dict)
            child2 = mutate(child2, activities_dict, slots, spaces_dict)
            
            # Add to offspring
            offspring.append(child1)
            offspring.append(child2)
        
        # Ensure offspring population size is exactly pop_size
        offspring = offspring[:pop_size]
        
        # Combine parent and offspring populations
        combined_population = population + offspring
        
        # Evaluate combined population
        combined_fitness = [
            multi_objective_evaluator(individual, activities_dict, groups_dict, 
                                    lecturers_dict, spaces_dict, slots)
            for individual in combined_population
        ]
        
        # Select next generation
        population = selection(combined_population, combined_fitness, pop_size)
    
    # Final evaluation
    final_fitness = [
        multi_objective_evaluator(individual, activities_dict, groups_dict, 
                                lecturers_dict, spaces_dict, slots)
        for individual in population
    ]
    
    # Get indices of non-dominated solutions (Pareto front)
    indices = list(range(len(population)))
    pareto_indices = []
    
    for i in indices:
        dominated = False
        for j in indices:
            if i != j and dominates(final_fitness[j], final_fitness[i]):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    
    return population, final_fitness, pareto_indices, best_fitness_history


# === Main Function ===

def main(args=None):
    """
    Execute the EA algorithms when the script is run directly.
    
    Args:
        args: Command-line arguments (parsed)
    """
    # Parse arguments if not provided
    if args is None:
        parser = argparse.ArgumentParser(description='Genetic Algorithm based Timetable Scheduler')
        parser.add_argument('--dataset', type=str, choices=['4room', '7room'], default='4room',
                            help='Dataset to use: 4room or 7room')
        parser.add_argument('--algorithm', type=str, choices=['nsga2', 'spea2', 'moead', 'all'], 
                            default='nsga2', help='Algorithm to run')
        parser.add_argument('--generations', type=int, default=50, 
                            help='Number of generations')
        parser.add_argument('--pop-size', type=int, default=100, 
                            help='Population size')
        parser.add_argument('--archive-size', type=int, default=50, 
                            help='Archive size (for SPEA2)')
        parser.add_argument('--output-dir', type=str, default='output', 
                            help='Output directory for results')
        args = parser.parse_args()
    
    # Set up paths and parameters based on arguments
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    
    if args.dataset == '4room':
        DATASET_FILENAME = 'sliit_computing_dataset_4.json'
        DATASET_NAME = "4-Room Dataset"
    else:  # 7room
        DATASET_FILENAME = 'sliit_computing_dataset_7.json'
        DATASET_NAME = "7-Room Dataset"
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR_BASE = os.path.join(PROJECT_ROOT, args.output_dir, args.dataset)
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    
    # Define algorithm parameters
    POP_SIZE_MAIN = args.pop_size
    GENERATIONS_MAIN = args.generations
    ARCHIVE_SIZE_MAIN = args.archive_size
    
    # Load data
    try:
        print(f"Loading data from {DATASET_FILENAME}...")
        data_result = load_data()
        activities_dict, groups_dict, spaces_dict, lecturers_dict, _, _, _, _, _, _, _, _, slots = data_result
        print(f"Data loaded successfully: {len(activities_dict)} activities, {len(spaces_dict)} spaces")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run the specified algorithm(s)
    if args.algorithm in ['nsga2', 'all']:
        print("\nRunning NSGA-II algorithm...")
        output_dir_nsga2 = os.path.join(OUTPUT_DIR_BASE, 'nsga2')
        os.makedirs(output_dir_nsga2, exist_ok=True)
        
        nsga2_population, nsga2_final_fitness, nsga2_pareto_indices, nsga2_history = nsga2(
            pop_size=POP_SIZE_MAIN, generations=GENERATIONS_MAIN,
            activities_dict=activities_dict, groups_dict=groups_dict,
            lecturers_dict=lecturers_dict, spaces_dict=spaces_dict, slots=slots
        )
        
        nsga2_pareto_fitness = [nsga2_final_fitness[i] for i in nsga2_pareto_indices]
        nsga2_execution_time = ALGORITHM_RESOURCE_METRICS.get('NSGA2', {}).get('execution_time', 0.0)
        
        print(f"NSGA-II completed in {nsga2_execution_time:.2f} seconds")
        print(f"Pareto front size: {len(nsga2_pareto_indices)}")
    
    # Print results summary
    print("\n--- Results Summary ---")
    if args.algorithm in ['nsga2', 'all']:
        print(f"NSGA-II: {len(nsga2_pareto_indices)} Pareto solutions, execution time: {nsga2_execution_time:.2f} seconds")
    
    print(f"\nResults saved to '{OUTPUT_DIR_BASE}' directory")

if __name__ == "__main__":
    main()
