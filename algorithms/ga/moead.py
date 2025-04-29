import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Set
import copy
import time
import os
from algorithms.data.loader import load_data, Activity, Group, Space # Assuming these are defined here or loaded
from algorithms.evaluation.evaluator import evaluate_timetable # General evaluator
from algorithms.ga.nsga2 import calculate_room_utilization # Import the function

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Algorithm parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
T = 5  # Size of the neighborhood
NUM_OBJECTIVES = 2  # Number of objectives

# Function to generate weight vectors for decomposition
def generate_weight_vectors(num_weights, num_objectives):
    """
    Generates weight vectors using the Dirichlet distribution, ensuring they sum to 1.
    
    Args:
        num_weights: Number of weight vectors to generate
        num_objectives: Number of objectives (dimensions of each weight vector)
        
    Returns:
        np.ndarray: Array of weight vectors
    """
    weight_vectors = []
    # Use numpy's newer random Generator for better randomness
    rng = np.random.default_rng(42)
    for _ in range(num_weights):
        vec = rng.dirichlet(np.ones(num_objectives), size=1)[0]
        weight_vectors.append(vec)
    return np.array(weight_vectors)

# Tchebycheff scalarizing function
def scalarizing_function(fitness, weight_vector, ideal_point):
    """
    Apply the Tchebycheff approach for decomposition.
    
    Args:
        fitness: Fitness values for an individual
        weight_vector: Weight vector for the subproblem
        ideal_point: Current ideal point
    
    Returns:
        float: Scalarized fitness value
    """
    return np.max(weight_vector * np.abs(np.array(fitness) - np.array(ideal_point)))

# Function to update the ideal point
def update_ideal_point(fitness_values, ideal_point):
    """
    Update the ideal point based on the best objective values found.
    
    Args:
        fitness_values: List of fitness tuples (hard_violations, soft_score)
        ideal_point: Current ideal point
        
    Returns:
        np.ndarray: Updated ideal point
    """
    # Ensure ideal_point is the correct length (2 elements - hard, soft)
    if len(ideal_point) != 2:
        # Initialize with worst possible values for 2 objectives
        ideal_point = np.array([float('inf'), float('inf')])
        
    for fitness in fitness_values:
        # Ensure fitness is a 2-element tuple (standardized format)
        if isinstance(fitness, tuple) and len(fitness) == 2:
            ideal_point = np.minimum(ideal_point, np.array(fitness))
    
    return ideal_point

# Function to select parents from a neighborhood
def select_parents_from_neighborhood(population, neighborhood):
    """
    Select two parents from a given neighborhood for crossover.
    
    Args:
        population: Current population
        neighborhood: Indices of neighbors
        
    Returns:
        tuple: Two selected parents
    """
    parent_indices = random.sample(neighborhood, 2)
    return population[parent_indices[0]], population[parent_indices[1]]

# Crossover function
def crossover(parent1, parent2):
    """
    Perform crossover by swapping time slots between two parents.
    
    Args:
        parent1: First parent timetable
        parent2: Second parent timetable
        
    Returns:
        tuple: Two new offspring timetables
    """
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    slots = list(parent1.keys())
    split = random.randint(0, len(slots) - 1)
    for i in range(split, len(slots)):
        child1[slots[i]], child2[slots[i]] = parent2[slots[i]], parent1[slots[i]]
    return child1, child2

# Mutation function: Apply mutation to a solution
def mutate(timetable: Dict[int, Dict[str, Any]], activities_dict: Dict[str, Any], slots: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Apply mutation to a timetable solution.
    
    Args:
        timetable: Timetable to mutate
        activities_dict: Dictionary of activities
        slots: List of time slot IDs
        
    Returns:
        Dict[int, Dict[str, Any]]: Mutated timetable
    """
    mutated_timetable = copy.deepcopy(timetable)
    
    # Use globally loaded data (except slots, which is now passed)
    global groups_dict, spaces_dict
    
    # Get all activity IDs from the global activities_dict
    all_activities = set(a.id for a in activities_dict.values()) # Use .id instead of .code
    assigned_activities = set()
    for slot in mutated_timetable:
        for room in mutated_timetable[slot]:
            activity = mutated_timetable[slot][room]
            if activity:
                assigned_activities.add(activity.id)  # Use .id instead of .code
    
    # Find unassigned activities
    unassigned_activities = list(all_activities - assigned_activities)
    
    if len(slots) < 2:
        return mutated_timetable
    
    # Choose mutation type based on probability
    mutation_type = random.choices(
        ['swap', 'rebalance', 'fill_unused', 'optimize_room_fit'],
        weights=[0.4, 0.3, 0.15, 0.15],
        k=1
    )[0]
    
    if mutation_type == 'swap':
        # Standard swap mutation (more explorative)
        slot1, slot2 = random.sample(slots, 2)
        
        # Ensure there are rooms in both slots
        room_keys1 = list(mutated_timetable[slot1].keys())
        room_keys2 = list(mutated_timetable[slot2].keys())
        
        if not room_keys1 or not room_keys2:
            return mutated_timetable
        
        room1 = random.choice(room_keys1)
        room2 = random.choice(room_keys2)
        
        # Swap activities
        mutated_timetable[slot1][room1], mutated_timetable[slot2][room2] = mutated_timetable[slot2][room2], mutated_timetable[slot1][room1]
    
    elif mutation_type == 'rebalance':
        # Calculate room utilization
        utilization = calculate_room_utilization(mutated_timetable, spaces_dict)
        
        # Find an overused and underused room to rebalance
        overused_rooms = sorted(utilization['room_usage'].items(), key=lambda x: x[1], reverse=True)
        underused_rooms = sorted(utilization['room_usage'].items(), key=lambda x: x[1])
        
        if overused_rooms and underused_rooms and overused_rooms[0][1] > underused_rooms[0][1]:
            over_room_id = overused_rooms[0][0]
            under_room_id = underused_rooms[0][0]
            
            # Find a slot where the overused room has an activity
            valid_slots = [s for s in slots if s not in activity_slots[activity_id]]
            
            if valid_slots:
                target_slot = random.choice(valid_slots)
                
                # Move activity from overused to underused room
                if under_room_id in mutated_timetable[target_slot] and mutated_timetable[target_slot][under_room_id] is None:
                    activity = mutated_timetable[target_slot][over_room_id]
                    # Check if the room is suitable for the activity
                    activity_size = get_classsize(activity, groups_dict) if activity else 0
                    
                    if activity and spaces_dict[under_room_id].size >= activity_size:
                        mutated_timetable[target_slot][under_room_id] = activity
                        mutated_timetable[target_slot][over_room_id] = None
    
    elif mutation_type == 'fill_unused':
        # Try to find an unassigned activity and place it in an unused room
        if unassigned_activities: # Check if there are any unassigned activities first
            # Choose a random unassigned activity
            activity_id = random.choice(unassigned_activities) # Now safe to define
            activity = activities_dict[activity_id]
            activity_size = get_classsize(activity, groups_dict)
            
            # Find empty slots/rooms that can accommodate this activity
            potential_placements = []
            for slot in slots:
                for room_id, room in spaces_dict.items():
                    if room_id in mutated_timetable[slot] and mutated_timetable[slot][room_id] is None \
                            and room.size >= activity_size:
                        potential_placements.append((slot, room_id))
            
            if potential_placements:
                chosen_slot, chosen_room = random.choice(potential_placements)
                mutated_timetable[chosen_slot][chosen_room] = activity
    
    elif mutation_type == 'optimize_room_fit':
        # Find mismatches between room capacity and activity needs
        for slot in random.sample(slots, min(3, len(slots))):  # Limit to 3 random slots for efficiency
            room_activities = []
            for room_id in mutated_timetable[slot]:
                if mutated_timetable[slot][room_id] is not None:
                    activity = mutated_timetable[slot][room_id]
                    activity_size = get_classsize(activity, groups_dict)
                    room_capacity = spaces_dict[room_id].size
                    # Calculate how well the room fits the activity
                    fit_score = room_capacity - activity_size
                    room_activities.append((room_id, activity, fit_score))
            
            # If we have at least 2 activities, try to optimize fit by swapping
            if len(room_activities) >= 2:
                # Sort by fit score (higher means more wasted space)
                room_activities.sort(key=lambda x: x[2], reverse=True)
                
                # Swap the worst and best fits if it improves overall fit
                if room_activities[0][2] > 0 and room_activities[-1][2] < room_activities[0][2]:
                    # Check if the swap is valid (capacities are sufficient)
                    worst_room_id, worst_activity, _ = room_activities[0]
                    best_room_id, best_activity, _ = room_activities[-1]
                    
                    worst_activity_size = get_classsize(worst_activity, groups_dict)
                    best_activity_size = get_classsize(best_activity, groups_dict)
                    
                    if (spaces_dict[best_room_id].size >= worst_activity_size and 
                            spaces_dict[worst_room_id].size >= best_activity_size):
                        # Swap activities for better room fit
                        mutated_timetable[slot][worst_room_id], mutated_timetable[slot][best_room_id] = \
                            mutated_timetable[slot][best_room_id], mutated_timetable[slot][worst_room_id]
    
    return mutated_timetable

# Generate initial population
def generate_initial_population(slots, spaces_dict, activities_dict, groups_dict, population_size):
    """
    Generate an initial population with timetables using multiple strategies.
    Enhanced to better utilize all available rooms and resource constraints.
    
    Args:
        slots: List of time slots
        spaces_dict: Dictionary of spaces
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of groups
        population_size: Size of the population
        
    Returns:
        list: Initial population of timetables
    """
    population = []
    
    # Define multiple initialization strategies to create diverse solutions
    strategies = [
        'prioritize_fitting', # Prioritize room fit (capacity close to needed)
        'prioritize_spread',  # Prioritize balanced room usage
        'random',            # Standard random assignment
        'greedy'             # Pack activities tightly
    ]
    
    # Distribute population across strategies
    strategy_assignments = {}
    base_count = population_size // len(strategies)
    remainder = population_size % len(strategies)
    
    for i, strategy in enumerate(strategies):
        count = base_count + (1 if i < remainder else 0)
        strategy_assignments[strategy] = count
    
    # Generate individuals for each strategy
    for strategy, count in strategy_assignments.items():
        for _ in range(count):
            # Initialize empty timetable
            timetable = {}
            for slot in slots:
                timetable[slot] = {}
                for space_id in spaces_dict.keys():
                    timetable[slot][space_id] = None  # Start with all empty
            
            activity_slots = {activity_id: [] for activity_id in activities_dict.keys()}
            
            # Track room usage for balanced strategy
            room_usage = {room_id: 0 for room_id in spaces_dict.keys()}
            
            # Make a copy of activities that need to be scheduled
            activities_list = list(activities_dict.values())
            activity_copies = []
            for activity in activities_list:
                for _ in range(activity.duration):
                    activity_copies.append(activity.id)
            
            # Sort activities based on the strategy
            if strategy == 'prioritize_fitting' or strategy == 'greedy':
                # Sort by size to prioritize fitting activities efficiently
                activities_to_schedule = sorted(activity_copies, 
                                             key=lambda a_id: get_classsize(activities_dict[a_id], groups_dict),
                                             reverse=(strategy == 'greedy'))  # Largest first for greedy
            else:
                # Random order for other strategies
                activities_to_schedule = activity_copies.copy()
                random.shuffle(activities_to_schedule)
            
            # Try to schedule activities
            for activity_id in activities_to_schedule:
                activity = activities_dict[activity_id]
                activity_size = get_classsize(activity, groups_dict)
                
                # Find suitable slots and rooms
                available_slots = [s for s in slots if s not in activity_slots[activity_id]]
                if not available_slots:
                    continue  # Skip if no slots available
                
                # Choose slot based on strategy
                if strategy == 'prioritize_spread':
                    # For the spread strategy, try to distribute across time slots
                    # Pick slots with fewer activities scheduled
                    slot_usage = {}
                    for s in available_slots:
                        slot_usage[s] = sum(1 for r in timetable[s] if timetable[s][r] is not None)
                    
                    # Choose slots with minimal usage
                    min_usage = min(slot_usage.values()) if slot_usage else 0
                    least_used_slots = [s for s in available_slots if slot_usage[s] == min_usage]
                    chosen_slot = random.choice(least_used_slots) if least_used_slots else random.choice(available_slots)
                else:
                    chosen_slot = random.choice(available_slots)
                
                # Find suitable rooms in that slot
                suitable_rooms = [
                    room_id for room_id, room in spaces_dict.items()
                    if room_id in timetable[chosen_slot] and timetable[chosen_slot][room_id] is None
                    and room.size >= activity_size
                ]
                
                if not suitable_rooms:
                    # If no suitable empty room, consider all rooms and potentially create conflicts
                    # This helps avoid unassigned activities
                    if strategy != 'random':
                        suitable_rooms = [
                            room_id for room_id, room in spaces_dict.items()
                            if room.size >= activity_size
                        ]
                    
                    if not suitable_rooms:
                        continue  # No suitable rooms at all, skip
                
                # Choose room based on strategy
                if strategy == 'prioritize_fitting':
                    # Choose room that best fits the class size (minimize wasted space)
                    suitable_rooms.sort(key=lambda r: spaces_dict[r].size - activity_size)
                    chosen_room = suitable_rooms[0]  # Best fit
                elif strategy == 'prioritize_spread':
                    # Choose least used room
                    suitable_rooms.sort(key=lambda r: room_usage[r])
                    chosen_room = suitable_rooms[0]  # Least used
                elif strategy == 'greedy':
                    # Choose smallest suitable room to make larger rooms available for bigger classes
                    suitable_rooms.sort(key=lambda r: spaces_dict[r].size)
                    chosen_room = suitable_rooms[0]  # Smallest suitable
                else:  # 'random' strategy
                    chosen_room = random.choice(suitable_rooms)
                
                # Assign the activity
                # Check if we need to force-assign (displacing another activity) - not for random
                if timetable[chosen_slot][chosen_room] is not None and strategy != 'random':
                    # Existing activity needs to be displaced - skip for random strategy
                    # In the other strategies, we'll allow displacement to improve assignments
                    timetable[chosen_slot][chosen_room] = activity
                else:
                    timetable[chosen_slot][chosen_room] = activity
                
                # Update tracking
                activity_slots[activity_id].append(chosen_slot)
                room_usage[chosen_room] += 1
            
            population.append(timetable)
    
    return population

# Helper function to get class size
def get_classsize(activity, groups_dict):
    """
    Calculate the total size of all groups in an activity.
    
    Args:
        activity: Activity object
        groups_dict: Dictionary of groups
        
    Returns:
        int: Total class size
    """
    classsize = 0
    for group_id in activity.group_ids:
        classsize += groups_dict[group_id].size
    return classsize

# Main MOEA/D function
def run_moead_optimizer(activities_dict, groups_dict, spaces_dict, slots, 
                        population_size=None, num_generations=None, neighborhood_size=20,
                        output_dir='moead_output'):
    """
    Main MOEA/D algorithm for timetable optimization.
    
    Args:
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of groups
        spaces_dict: Dictionary of spaces
        slots: List of time slots
        population_size: Size of the population (optional)
        num_generations: Number of generations (optional)
        neighborhood_size: Size of the neighborhood for updates (optional)
        output_dir: Directory to save outputs (optional)
        
    Returns:
        tuple: Best solution and metrics dictionary
    """
    # Use provided parameters or defaults
    pop_size = population_size if population_size is not None else POPULATION_SIZE
    num_generations = num_generations if num_generations is not None else NUM_GENERATIONS
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics tracking to match the format expected by plots.py
    metrics = {
        'generations': list(range(num_generations)),
        'best_hard_violations': [],        # Best hard constraint violations
        'average_hard_violations': [],     # Average hard constraint violations
        'best_soft_score': [],             # Best soft constraint score
        'average_soft_score': [],          # Average soft constraint score
        'best_fitness': [],                # Best overall fitness
        'avg_fitness': [],                 # Average overall fitness
        'hypervolume': [],                 # Hypervolume indicator
        'constraint_violations': [],       # Detailed constraint violations
        'pareto_front_size': [],           # Size of Pareto front
        'execution_time': [],              # Execution time per generation
        'spacing': [],                     # Spacing metric
        'igd': []                          # Inverse Generational Distance
    }
    
    start_time_total = time.time()
    
    # Generate initial population
    print("Generating initial population...")
    population = generate_initial_population(slots, spaces_dict, activities_dict, groups_dict, pop_size)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    fitness_values = [evaluator(timetable, activities_dict, groups_dict, spaces_dict) for timetable in population]
    
    # Initialize ideal point with worst possible values for 2 objectives (hard, soft)
    ideal_point = np.array([float('inf'), float('inf')])
    ideal_point = update_ideal_point(fitness_values, ideal_point)
    
    # Generate weight vectors
    print("Generating weight vectors...")
    weight_vectors = generate_weight_vectors(pop_size, NUM_OBJECTIVES)
    
    # Use the provided neighborhood_size or default from global/constants if applicable
    T = neighborhood_size # T is commonly used for neighborhood size in MOEA/D literature

    # Create neighborhoods based on weight vector distances
    print("Creating neighborhoods...")
    distances = np.zeros((pop_size, pop_size))
    for i in range(pop_size):
        for j in range(pop_size):
            distances[i, j] = np.linalg.norm(weight_vectors[i] - weight_vectors[j])
    
    neighborhoods = [list(np.argsort(distances[i])[:T]) for i in range(pop_size)]
    
    # Main algorithm loop
    for generation in range(num_generations):
        start_time_gen = time.time()
        print(f"Generation {generation}: Population size {len(population)}")
        
        new_population = copy.deepcopy(population)
        
        for i in range(pop_size):
            # Select parents from neighborhood
            parent1, parent2 = select_parents_from_neighborhood(population, neighborhoods[i])
            
            # Apply crossover
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Apply mutation
            child1 = mutate(child1, activities_dict, slots) # Pass activities_dict and slots
            child2 = mutate(child2, activities_dict, slots) # Pass activities_dict and slots
            
            # Evaluate children
            child1_fitness = evaluator(child1, activities_dict, groups_dict, spaces_dict)
            child2_fitness = evaluator(child2, activities_dict, groups_dict, spaces_dict)
            
            # Update ideal point
            ideal_point = update_ideal_point([child1_fitness, child2_fitness], ideal_point)
            
            # Update subproblems in the neighborhood
            for j in neighborhoods[i]:
                # Calculate scalarized fitness based on the ORIGINAL fitness_values[j]
                current_scalarized = scalarizing_function(fitness_values[j], weight_vectors[j], ideal_point)
                
                child1_scalarized = scalarizing_function(child1_fitness, weight_vectors[j], ideal_point)
                if child1_scalarized < current_scalarized:
                    new_population[j] = child1
                    # fitness_values[j] = child1_fitness  # REMOVED: Do not update fitness_values here
                
                child2_scalarized = scalarizing_function(child2_fitness, weight_vectors[j], ideal_point)
                if child2_scalarized < current_scalarized:
                    # If child1 already replaced j, this overwrites it. 
                    # Consider if this is the desired behavior or if only the best child should replace.
                    new_population[j] = child2
                    # fitness_values[j] = child2_fitness # REMOVED: Do not update fitness_values here
        
        # --- ADDED: Update population and recalculate fitness values for the next generation --- 
        population = new_population
        fitness_values = [evaluator(timetable, activities_dict, groups_dict, spaces_dict) for timetable in population]
        # --- End of ADDED lines ---

        # Calculate metrics based on the NEW population and fitness_values
        hard_violations = [sum(fitness[:3]) for fitness in fitness_values]  # First 3 objectives are hard constraints
        soft_scores = [sum(fitness[3:]) for fitness in fitness_values]      # Last 2 objectives are soft constraints
        
        best_hard_idx = np.argmin(hard_violations)
        best_soft_idx = np.argmin(soft_scores)
        best_overall_idx = np.argmin([sum(fitness) for fitness in fitness_values])
        
        # Store metrics
        metrics['best_hard_violations'].append(hard_violations[best_hard_idx])
        metrics['average_hard_violations'].append(np.mean(hard_violations))
        metrics['best_soft_score'].append(soft_scores[best_soft_idx])
        metrics['average_soft_score'].append(np.mean(soft_scores))
        metrics['best_fitness'].append(sum(fitness_values[best_overall_idx]))
        metrics['avg_fitness'].append(np.mean([sum(fit) for fit in fitness_values]))
        
        # Calculate constraint violations by type
        best_solution = population[best_overall_idx]
        violation_details = detailed_constraint_violations(best_solution, activities_dict, groups_dict, spaces_dict)
        metrics['constraint_violations'].append(violation_details)
        
        # Calculate Pareto front
        non_dominated = find_non_dominated_solutions(fitness_values)
        metrics['pareto_front_size'].append(len(non_dominated))
        
        # Calculate hypervolume
        try:
            hypervolume = calculate_hypervolume([fitness_values[i] for i in non_dominated], 
                                               reference_point=[1000, 1000, 1000, 1000, 1000])
            metrics['hypervolume'].append(hypervolume)
        except Exception as e:
            print(f"Warning: Could not calculate hypervolume: {str(e)}")
            metrics['hypervolume'].append(0)
        
        # Calculate spacing metric (optional)
        try:
            spacing = calculate_spacing([fitness_values[i] for i in non_dominated])
            metrics['spacing'].append(spacing)
        except ValueError as e:
            print(f"Warning: Could not calculate spacing: {str(e)}")
            metrics['spacing'].append(0)
        
        # Calculate IGD (optional - would need reference set)
        metrics['igd'].append(0)  # Placeholder
        
        # Execution time
        generation_time = time.time() - start_time_gen
        metrics['execution_time'].append(generation_time)
    
    # Calculate total execution time
    total_time = time.time() - start_time_total
    
    # Find the best solution
    best_solution = None
    best_fitness_sum = float('inf')
    for i, fitness in enumerate(fitness_values):
        fitness_sum = sum(fitness)
        if fitness_sum < best_fitness_sum:
            best_fitness_sum = fitness_sum
            best_solution = population[i]
    
    # === Add Final Pareto Front to Metrics ===
    try:
        non_dominated_indices = find_non_dominated_solutions(fitness_values)
        final_pareto_fitness = [fitness_values[i] for i in non_dominated_indices]
        metrics["final_pareto_front"] = final_pareto_fitness
        print(f"Extracted final Pareto front with {len(final_pareto_fitness)} solutions.")
    except Exception as e:
        print(f"Error extracting final Pareto front: {e}")
        metrics["final_pareto_front"] = [] # Store empty list on error
    # =========================================
            
    print(f"Optimization completed in {total_time:.2f} seconds")
    
    return best_solution, metrics

# Function to find non-dominated solutions (for Pareto front)
def find_non_dominated_solutions(fitness_values):
    """
    Find indices of non-dominated solutions in the population.
    
    Args:
        fitness_values: List of fitness tuples
        
    Returns:
        list: Indices of non-dominated solutions
    """
    non_dominated = []
    for i, fitness_i in enumerate(fitness_values):
        dominated = False
        for j, fitness_j in enumerate(fitness_values):
            if i != j and dominates(fitness_j, fitness_i):
                dominated = True
                break
        if not dominated:
            non_dominated.append(i)
    return non_dominated

# Function to check if one solution dominates another
def dominates(fitness1, fitness2):
    """
    Check if fitness1 dominates fitness2 (Pareto dominance).
    
    Args:
        fitness1: First fitness tuple
        fitness2: Second fitness tuple
        
    Returns:
        bool: True if fitness1 dominates fitness2
    """
    return all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))

# Function for detailed constraint violations
def detailed_constraint_violations(timetable, activities_dict, groups_dict, spaces_dict):
    """
    Get detailed breakdown of constraint violations.
    
    Args:
        timetable: Timetable to evaluate
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of groups
        spaces_dict: Dictionary of spaces
        
    Returns:
        dict: Detailed constraint violations
    """
    # Calculate individual constraint violations with a separate helper function
    # since our main evaluator now returns a standardized 2-element tuple
    vacant_room = 0
    prof_conflicts = 0
    room_size_conflicts = 0
    sub_group_conflicts = 0
    unassigned_activities = len(activities_dict)
    activities_set = set()

    for slot in timetable:
        prof_set = set()
        sub_group_set = set()
        for room in timetable[slot]:
            activity = timetable[slot][room]

            if not activity:  # If no activity assigned
                vacant_room += 1
            
            elif hasattr(activity, 'id'):  # If it's an activity object
                activities_set.add(activity.id)
                
                if activity.teacher_id in prof_set:
                    prof_conflicts += 1

                # Check for subgroup conflicts
                current_groups = set(activity.group_ids)
                conflicts = len(current_groups.intersection(sub_group_set))
                sub_group_conflicts += conflicts

                # Check room size conflicts
                group_size = sum(groups_dict[group_id].size for group_id in activity.group_ids)
                if group_size > spaces_dict[room].size:
                    room_size_conflicts += 1
                
                # Update sets
                prof_set.add(activity.teacher_id)
                sub_group_set.update(activity.group_ids)

    unassigned_activities -= len(activities_set)
    
    # Get the total weighted violations from the standardized evaluator
    hard_violations, _ = evaluator(timetable, activities_dict, groups_dict, spaces_dict)
    
    return {
        'vacant_rooms': vacant_room,
        'professor_conflicts': prof_conflicts,
        'room_size_conflicts': room_size_conflicts,
        'subgroup_conflicts': sub_group_conflicts,
        'unassigned_activities': unassigned_activities,
        'total': hard_violations  # Use the total weighted value from the standardized evaluator
    }

# Function to calculate spacing metric
def calculate_spacing(front):
    """
    Calculate spacing metric for the Pareto front.
    
    Args:
        front: List of points in the Pareto front
        
    Returns:
        float: Spacing metric value
    """
    if len(front) < 2:
        return 0
    
    # Calculate distances between consecutive points
    distances = []
    for i in range(len(front) - 1):
        dist = np.linalg.norm(np.array(front[i]) - np.array(front[i+1]))
        distances.append(dist)
    
    # Calculate standard deviation of distances
    return np.std(distances)

# Evaluator function
def evaluator(timetable, activities_dict, groups_dict, spaces_dict):
    """
    Evaluates the timetable based on various constraints.
    
    Args:
        timetable: Timetable to evaluate
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of groups
        spaces_dict: Dictionary of spaces
        
    Returns:
        tuple: A tuple containing (weighted_hard_constraints_score, soft_constraints_score)
    """
    vacant_room = 0
    prof_conflicts = 0
    room_size_conflicts = 0
    sub_group_conflicts = 0
    unassigned_activities = len(activities_dict)
    activities_set = set()

    for slot in timetable:
        prof_set = set()
        sub_group_set = set()
        for room in timetable[slot]:
            activity = timetable[slot][room]

            if not activity:  # If no activity assigned
                vacant_room += 1
            
            elif hasattr(activity, 'id'):  # If it's an activity object
                activities_set.add(activity.id)
                
                if activity.teacher_id in prof_set:
                    prof_conflicts += 1

                # Check for subgroup conflicts
                current_groups = set(activity.group_ids)
                conflicts = len(current_groups.intersection(sub_group_set))
                sub_group_conflicts += conflicts

                # Check room size conflicts
                group_size = sum(groups_dict[group_id].size for group_id in activity.group_ids)
                if group_size > spaces_dict[room].size:
                    room_size_conflicts += 1
                
                # Update sets
                prof_set.add(activity.teacher_id)
                sub_group_set.update(activity.group_ids)

    unassigned_activities -= len(activities_set)
    
    # Calculate hard constraints total with similar weights to NSGA-II for consistent comparison
    hard_violations = (
        vacant_room * 1 +            # Vacant rooms (lower priority)
        prof_conflicts * 100 +        # Lecturer conflicts (high priority)
        sub_group_conflicts * 100 +   # Student group conflicts (high priority)
        room_size_conflicts * 50 +    # Room capacity (medium priority)
        unassigned_activities * 200   # Unassigned activities (highest priority)
    )
    
    # For soft constraints, since MOEAD doesn't evaluate them directly,
    # we'll use a placeholder value that's consistent with other algorithms
    # This will be refined in a future update to include actual soft constraint evaluation
    soft_score = 0.5
    
    # Return as a tuple for multi-objective optimization
    return (hard_violations, 1.0 - soft_score)

# Function to calculate hypervolume
def calculate_hypervolume(front, reference_point=None):
    """
    Calculate the hypervolume indicator for a Pareto front.
    
    Args:
        front: List of points in the Pareto front
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        float: Hypervolume value
    """
    if reference_point is None:
        # Set default reference point if none provided
        reference_point = [1000] * len(front[0])
    
    # Sort points by first objective
    sorted_front = sorted(front, key=lambda x: x[0])
    
    # Simple hypervolume calculation for 2D
    if len(front[0]) == 2:
        hypervolume = 0
        prev_x = reference_point[0]
        for point in sorted_front:
            hypervolume += (prev_x - point[0]) * (reference_point[1] - point[1])
            prev_x = point[0]
        return hypervolume
    
    # For higher dimensions, we would need a more complex algorithm
    # This is a simplified placeholder
    return sum(sum(abs(r - p) for r, p in zip(reference_point, point)) for point in front)

# If this script is run directly
if __name__ == "__main__":
    print("MOEA/D algorithm implementation for timetable scheduling")
    print("This file should be imported and used by run_optimization.py")
