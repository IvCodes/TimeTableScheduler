"""
TimeTable Scheduler - Azure Fabric Notebook Script (Part 3: Genetic Algorithm)
This script is formatted for easy conversion to a Jupyter notebook.
Each cell is marked with '# CELL: {description}' comments.
"""

# CELL: Import libraries
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Callable
from dataclasses import dataclass

# CELL: Genetic Algorithm Operators

def initialize_population(activities_dict, spaces_dict, slots, pop_size):
    """
    Initialize a population of random schedules.
    
    Args:
        activities_dict: Dictionary of activities
        spaces_dict: Dictionary of spaces
        slots: List of time slots
        pop_size: Size of the population
        
    Returns:
        List[List[Tuple]]: Population of individuals
    """
    population = []
    activities = list(activities_dict.keys())
    spaces = list(spaces_dict.keys())
    
    for _ in range(pop_size):
        # Create a random schedule
        individual = []
        
        for activity in activities:
            # Assign random slot and space
            slot_idx = random.randint(0, len(slots) - 1)
            space_id = random.choice(spaces)
            
            individual.append((slot_idx, space_id))
        
        population.append(individual)
    
    return population

def tournament_selection(population, fitness, tournament_size=3):
    """
    Select an individual using tournament selection.
    
    Args:
        population: List of individuals
        fitness: List of fitness values corresponding to population
        tournament_size: Size of the tournament
        
    Returns:
        Any: Selected individual
    """
    # Select tournament_size individuals randomly
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness[i] for i in tournament_indices]
    
    # Find the index of the best individual in the tournament
    best_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness, key=lambda x: x[-1]))]
    
    return population[best_idx]

def crossover(parent1, parent2, crossover_rate=0.9):
    """
    Perform crossover between two parents.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        crossover_rate: Probability of crossover
        
    Returns:
        Tuple[List, List]: Two offspring
    """
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    # Perform two-point crossover
    point1 = random.randint(0, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    
    offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    return offspring1, offspring2

def mutation(individual, activities_dict, spaces_dict, slots, mutation_rate=0.1):
    """
    Apply mutation to an individual.
    
    Args:
        individual: Individual to mutate
        activities_dict: Dictionary of activities
        spaces_dict: Dictionary of spaces
        slots: List of time slots
        mutation_rate: Probability of mutation per gene
        
    Returns:
        List: Mutated individual
    """
    mutated = individual.copy()
    spaces = list(spaces_dict.keys())
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # 50% chance to change the slot, 50% chance to change the space
            if random.random() < 0.5:
                slot_idx = random.randint(0, len(slots) - 1)
                mutated[i] = (slot_idx, mutated[i][1])
            else:
                space_id = random.choice(spaces)
                mutated[i] = (mutated[i][0], space_id)
    
    return mutated

def non_dominated_sort(population, fitness):
    """
    Perform non-dominated sorting for NSGA-II algorithm.
    
    Args:
        population: List of individuals
        fitness: List of fitness values corresponding to population
        
    Returns:
        List[List]: Fronts of non-dominated solutions
    """
    fronts = [[]]
    
    # Calculate domination for each individual
    domination_count = [0] * len(population)
    dominated_solutions = [[] for _ in range(len(population))]
    
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                # Check if i dominates j
                i_dominates_j = all(fitness[i][k] <= fitness[j][k] for k in range(len(fitness[i]))) and \
                               any(fitness[i][k] < fitness[j][k] for k in range(len(fitness[i])))
                
                # Check if j dominates i
                j_dominates_i = all(fitness[j][k] <= fitness[i][k] for k in range(len(fitness[i]))) and \
                               any(fitness[j][k] < fitness[i][k] for k in range(len(fitness[i])))
                
                if i_dominates_j:
                    dominated_solutions[i].append(j)
                elif j_dominates_i:
                    domination_count[i] += 1
        
        # If i is not dominated by anyone, add to first front
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Generate subsequent fronts
    front_idx = 0
    while fronts[front_idx]:
        next_front = []
        
        for i in fronts[front_idx]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                
                if domination_count[j] == 0:
                    next_front.append(j)
        
        front_idx += 1
        fronts.append(next_front)
    
    # Remove empty front
    fronts.pop()
    
    return fronts

def crowding_distance(fitness, front):
    """
    Calculate crowding distance for individuals in a front.
    
    Args:
        fitness: List of fitness values
        front: List of indices in the front
        
    Returns:
        List[float]: Crowding distances for individuals in the front
    """
    if len(front) <= 2:
        return [float('inf')] * len(front)
    
    # Initialize distances
    distances = [0] * len(front)
    
    # For each objective
    for obj_idx in range(len(fitness[0])):
        # Sort front by this objective
        sorted_front = sorted(front, key=lambda x: fitness[x][obj_idx])
        
        # Set boundary points to infinity
        distances[front.index(sorted_front[0])] = float('inf')
        distances[front.index(sorted_front[-1])] = float('inf')
        
        # Calculate distances for other points
        obj_range = fitness[sorted_front[-1]][obj_idx] - fitness[sorted_front[0]][obj_idx]
        
        if obj_range == 0:
            continue
            
        for i in range(1, len(sorted_front) - 1):
            distances[front.index(sorted_front[i])] += (
                fitness[sorted_front[i+1]][obj_idx] - fitness[sorted_front[i-1]][obj_idx]
            ) / obj_range
    
    return distances

# CELL: NSGA-II Implementation
def nsga2(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
         pop_size=100, generations=200, crossover_rate=0.9, mutation_rate=0.1):
    """
    Implementation of NSGA-II algorithm for timetable scheduling.
    
    Args:
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of student groups
        spaces_dict: Dictionary of spaces
        lecturers_dict: Dictionary of lecturers
        slots: List of time slots
        pop_size: Population size
        generations: Number of generations
        crossover_rate: Crossover rate
        mutation_rate: Mutation rate
        
    Returns:
        Tuple[List, List]: Final population and their fitness values
    """
    print(f"\n--- Running NSGA-II (Pop: {pop_size}, Gen: {generations}) ---")
    
    # Initialize population
    population = initialize_population(activities_dict, spaces_dict, slots, pop_size)
    
    # Evaluate initial population
    fitness = []
    for individual in population:
        schedule = parse_ga_schedule(individual, activities_dict, spaces_dict, slots)
        fitness.append(multi_objective_evaluator(
            schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        ))
    
    # Store history for analysis
    history = {
        'fitness': [fitness],
        'best_individuals': [],
        'hard_violations': [],
        'soft_violations': []
    }
    
    # Main loop
    for generation in range(generations):
        print(f"NSGA-II Generation {generation+1}/{generations}", end='\r')
        
        # Create offspring
        offspring = []
        offspring_fitness = []
        
        while len(offspring) < pop_size:
            # Select parents
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            
            # Apply crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            
            # Apply mutation
            child1 = mutation(child1, activities_dict, spaces_dict, slots, mutation_rate)
            child2 = mutation(child2, activities_dict, spaces_dict, slots, mutation_rate)
            
            # Evaluate children
            for child in [child1, child2]:
                if len(offspring) < pop_size:
                    schedule = parse_ga_schedule(child, activities_dict, spaces_dict, slots)
                    child_fitness = multi_objective_evaluator(
                        schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
                    )
                    
                    offspring.append(child)
                    offspring_fitness.append(child_fitness)
        
        # Combine parent and offspring populations
        combined_pop = population + offspring
        combined_fitness = fitness + offspring_fitness
        
        # Perform non-dominated sorting
        fronts = non_dominated_sort(combined_pop, combined_fitness)
        
        # Select next generation
        new_population = []
        new_fitness = []
        
        front_idx = 0
        while len(new_population) + len(fronts[front_idx]) <= pop_size:
            # Add whole front
            for idx in fronts[front_idx]:
                new_population.append(combined_pop[idx])
                new_fitness.append(combined_fitness[idx])
            
            front_idx += 1
            if front_idx == len(fronts):
                break
        
        # If we need more individuals, add from next front based on crowding distance
        if len(new_population) < pop_size and front_idx < len(fronts):
            # Calculate crowding distances
            distances = crowding_distance(combined_fitness, fronts[front_idx])
            
            # Sort front by crowding distance
            sorted_front = sorted(
                [(idx, distances[i]) for i, idx in enumerate(fronts[front_idx])],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Add individuals from front until population is filled
            for idx, _ in sorted_front:
                if len(new_population) < pop_size:
                    new_population.append(combined_pop[idx])
                    new_fitness.append(combined_fitness[idx])
        
        # Update population and fitness
        population = new_population
        fitness = new_fitness
        
        # Store generation history
        history['fitness'].append(fitness)
        
        # Find best individual in this generation
        best_idx = min(range(len(fitness)), key=lambda i: fitness[i][-1])
        best_individual = population[best_idx]
        best_schedule = parse_ga_schedule(best_individual, activities_dict, spaces_dict, slots)
        
        # Evaluate best schedule
        _, hard_violations, soft_violations = evaluate_schedule(
            best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        
        history['best_individuals'].append(best_individual)
        history['hard_violations'].append(hard_violations)
        history['soft_violations'].append(soft_violations)
    
    print("\nNSGA-II Complete!")
    return population, fitness

# CELL: SPEA2 Implementation
def spea2(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
         pop_size=100, generations=200, archive_size=100, crossover_rate=0.9, mutation_rate=0.1):
    """
    Implementation of SPEA2 algorithm for timetable scheduling.
    
    Args:
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of student groups
        spaces_dict: Dictionary of spaces
        lecturers_dict: Dictionary of lecturers
        slots: List of time slots
        pop_size: Population size
        generations: Number of generations
        archive_size: Size of the archive
        crossover_rate: Crossover rate
        mutation_rate: Mutation rate
        
    Returns:
        Tuple[List, List]: Final archive and their fitness values
    """
    print(f"\n--- Running SPEA2 (Pop: {pop_size}, Gen: {generations}, Archive: {archive_size}) ---")
    
    # Initialize population
    population = initialize_population(activities_dict, spaces_dict, slots, pop_size)
    archive = []
    
    # Evaluate initial population
    population_fitness = []
    for individual in population:
        schedule = parse_ga_schedule(individual, activities_dict, spaces_dict, slots)
        population_fitness.append(multi_objective_evaluator(
            schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        ))
    
    # Store history for analysis
    history = {
        'fitness': [population_fitness],
        'best_individuals': [],
        'hard_violations': [],
        'soft_violations': []
    }
    
    # Calculate fitness for archive (initially empty)
    archive_fitness = []
    
    # Main loop
    for generation in range(generations):
        print(f"SPEA2 Generation {generation+1}/{generations}", end='\r')
        
        # Combine population and archive
        combined_pop = population + archive
        combined_fitness = population_fitness + archive_fitness
        
        # Calculate strength of each individual
        strength = [0] * len(combined_pop)
        
        for i in range(len(combined_pop)):
            for j in range(len(combined_pop)):
                if i != j:
                    # Check if i dominates j
                    i_dominates_j = all(combined_fitness[i][k] <= combined_fitness[j][k] for k in range(len(combined_fitness[i]))) and \
                                   any(combined_fitness[i][k] < combined_fitness[j][k] for k in range(len(combined_fitness[i])))
                    
                    if i_dominates_j:
                        strength[i] += 1
        
        # Calculate raw fitness (sum of strengths of dominators)
        raw_fitness = [0] * len(combined_pop)
        
        for i in range(len(combined_pop)):
            for j in range(len(combined_pop)):
                if i != j:
                    # Check if j dominates i
                    j_dominates_i = all(combined_fitness[j][k] <= combined_fitness[i][k] for k in range(len(combined_fitness[i]))) and \
                                   any(combined_fitness[j][k] < combined_fitness[i][k] for k in range(len(combined_fitness[i])))
                    
                    if j_dominates_i:
                        raw_fitness[i] += strength[j]
        
        # Calculate density (kth nearest neighbor)
        k = int(np.sqrt(len(combined_pop)))
        density = [0] * len(combined_pop)
        
        for i in range(len(combined_pop)):
            # Calculate distances to all other individuals
            distances = []
            
            for j in range(len(combined_pop)):
                if i != j:
                    # Euclidean distance in objective space
                    dist = np.sqrt(sum((combined_fitness[i][k] - combined_fitness[j][k])**2 
                                    for k in range(len(combined_fitness[i]))))
                    distances.append(dist)
            
            # Sort distances
            distances.sort()
            
            # kth distance (add small value to avoid division by zero)
            density[i] = 1.0 / (distances[k-1] + 0.00001)
        
        # Calculate final fitness (raw fitness + density)
        final_fitness = [raw_fitness[i] + density[i] for i in range(len(combined_pop))]
        
        # Select archive for next generation
        next_archive = []
        next_archive_fitness = []
        
        # Add non-dominated individuals to archive
        for i in range(len(combined_pop)):
            if raw_fitness[i] == 0:  # Non-dominated
                next_archive.append(combined_pop[i])
                next_archive_fitness.append(combined_fitness[i])
        
        # If archive is too small, add dominated individuals based on fitness
        if len(next_archive) < archive_size:
            # Sort remaining individuals by fitness
            remaining = [(i, final_fitness[i]) for i in range(len(combined_pop)) if raw_fitness[i] > 0]
            remaining.sort(key=lambda x: x[1])
            
            # Add individuals until archive is filled
            for idx, _ in remaining:
                if len(next_archive) < archive_size:
                    next_archive.append(combined_pop[idx])
                    next_archive_fitness.append(combined_fitness[idx])
        
        # If archive is too large, truncate based on density
        elif len(next_archive) > archive_size:
            while len(next_archive) > archive_size:
                # Calculate distances between all archive members
                min_distance = float('inf')
                min_idx = -1
                
                for i in range(len(next_archive)):
                    distances = []
                    
                    for j in range(len(next_archive)):
                        if i != j:
                            # Euclidean distance in objective space
                            dist = np.sqrt(sum((next_archive_fitness[i][k] - next_archive_fitness[j][k])**2 
                                            for k in range(len(next_archive_fitness[i]))))
                            distances.append(dist)
                    
                    # Sort distances
                    distances.sort()
                    
                    # Find individual with smallest distance to its nearest neighbor
                    if distances[0] < min_distance:
                        min_distance = distances[0]
                        min_idx = i
                
                # Remove individual with smallest distance
                next_archive.pop(min_idx)
                next_archive_fitness.pop(min_idx)
        
        # Update archive
        archive = next_archive
        archive_fitness = next_archive_fitness
        
        # Create next population
        new_population = []
        new_fitness = []
        
        while len(new_population) < pop_size:
            # Select parents from archive
            parent1 = tournament_selection(archive, archive_fitness)
            parent2 = tournament_selection(archive, archive_fitness)
            
            # Apply crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            
            # Apply mutation
            child1 = mutation(child1, activities_dict, spaces_dict, slots, mutation_rate)
            child2 = mutation(child2, activities_dict, spaces_dict, slots, mutation_rate)
            
            # Evaluate children
            for child in [child1, child2]:
                if len(new_population) < pop_size:
                    schedule = parse_ga_schedule(child, activities_dict, spaces_dict, slots)
                    child_fitness = multi_objective_evaluator(
                        schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
                    )
                    
                    new_population.append(child)
                    new_fitness.append(child_fitness)
        
        # Update population
        population = new_population
        population_fitness = new_fitness
        
        # Store generation history
        combined_fitness_history = population_fitness + archive_fitness
        history['fitness'].append(combined_fitness_history)
        
        # Find best individual in this generation (from archive)
        if archive:
            best_idx = min(range(len(archive_fitness)), key=lambda i: archive_fitness[i][-1])
            best_individual = archive[best_idx]
            best_schedule = parse_ga_schedule(best_individual, activities_dict, spaces_dict, slots)
            
            # Evaluate best schedule
            _, hard_violations, soft_violations = evaluate_schedule(
                best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
            )
            
            history['best_individuals'].append(best_individual)
            history['hard_violations'].append(hard_violations)
            history['soft_violations'].append(soft_violations)
    
    print("\nSPEA2 Complete!")
    return archive, archive_fitness

# CELL: MOEA/D Implementation
def moead(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
         pop_size=100, generations=200, neighborhood_size=20, crossover_rate=0.9, mutation_rate=0.1):
    """
    Implementation of MOEA/D algorithm for timetable scheduling.
    
    Args:
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of student groups
        spaces_dict: Dictionary of spaces
        lecturers_dict: Dictionary of lecturers
        slots: List of time slots
        pop_size: Population size
        generations: Number of generations
        neighborhood_size: Size of the neighborhood
        crossover_rate: Crossover rate
        mutation_rate: Mutation rate
        
    Returns:
        Tuple[List, List]: Final population and their fitness values
    """
    print(f"\n--- Running MOEA/D (Pop: {pop_size}, Gen: {generations}, Neighborhood: {neighborhood_size}) ---")
    
    # Number of objectives
    num_objectives = 5  # 4 objectives + total
    
    # Generate weight vectors
    def generate_weights(n, m):
        """Generate n weight vectors for m objectives."""
        weights = []
        
        # For simplicity, use random weights that sum to 1
        for _ in range(n):
            w = np.random.random(m)
            w = w / np.sum(w)
            weights.append(w)
        
        return weights
    
    # Calculate Euclidean distance between weight vectors
    def euclidean_distance(w1, w2):
        return np.sqrt(np.sum((w1 - w2)**2))
    
    # Initialize weight vectors
    weights = generate_weights(pop_size, num_objectives - 1)  # Exclude total
    
    # Calculate neighborhood
    neighborhoods = []
    
    for i in range(pop_size):
        # Calculate distances to all other weight vectors
        distances = [(j, euclidean_distance(weights[i], weights[j])) for j in range(pop_size)]
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Select neighborhood_size closest neighbors
        neighborhoods.append([idx for idx, _ in distances[:neighborhood_size]])
    
    # Initialize population
    population = initialize_population(activities_dict, spaces_dict, slots, pop_size)
    
    # Evaluate initial population
    fitness = []
    for individual in population:
        schedule = parse_ga_schedule(individual, activities_dict, spaces_dict, slots)
        fitness.append(multi_objective_evaluator(
            schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        ))
    
    # Calculate ideal point (minimum value for each objective)
    ideal_point = [min(fitness[i][j] for i in range(pop_size)) for j in range(num_objectives)]
    
    # Initialize external population (non-dominated solutions)
    external_pop = []
    external_fitness = []
    
    # Store history for analysis
    history = {
        'fitness': [fitness],
        'best_individuals': [],
        'hard_violations': [],
        'soft_violations': []
    }
    
    # Main loop
    for generation in range(generations):
        print(f"MOEA/D Generation {generation+1}/{generations}", end='\r')
        
        # For each subproblem
        for i in range(pop_size):
            # Select parents from neighborhood
            parent_indices = random.sample(neighborhoods[i], 2)
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            
            # Apply crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            
            # Apply mutation
            child1 = mutation(child1, activities_dict, spaces_dict, slots, mutation_rate)
            child2 = mutation(child2, activities_dict, spaces_dict, slots, mutation_rate)
            
            # Evaluate children
            for child in [child1, child2]:
                schedule = parse_ga_schedule(child, activities_dict, spaces_dict, slots)
                child_fitness = multi_objective_evaluator(
                    schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
                )
                
                # Update ideal point
                for j in range(num_objectives):
                    ideal_point[j] = min(ideal_point[j], child_fitness[j])
                
                # Check if child is non-dominated
                is_dominated = False
                
                for j in range(len(external_pop)):
                    # Check if external solution dominates child
                    ext_dominates_child = all(external_fitness[j][k] <= child_fitness[k] for k in range(num_objectives)) and \
                                          any(external_fitness[j][k] < child_fitness[k] for k in range(num_objectives))
                    
                    # Check if child dominates external solution
                    child_dominates_ext = all(child_fitness[k] <= external_fitness[j][k] for k in range(num_objectives)) and \
                                          any(child_fitness[k] < external_fitness[j][k] for k in range(num_objectives))
                    
                    if ext_dominates_child:
                        is_dominated = True
                        break
                    elif child_dominates_ext:
                        # Remove dominated external solution
                        external_pop.pop(j)
                        external_fitness.pop(j)
                
                # Add non-dominated child to external population
                if not is_dominated:
                    external_pop.append(child)
                    external_fitness.append(child_fitness)
                
                # Update neighboring solutions
                for j in neighborhoods[i]:
                    # Calculate Tchebycheff aggregation function
                    current_value = max(weights[j][k] * abs(fitness[j][k] - ideal_point[k]) 
                                        for k in range(num_objectives - 1))
                    child_value = max(weights[j][k] * abs(child_fitness[k] - ideal_point[k]) 
                                      for k in range(num_objectives - 1))
                    
                    # If child is better, replace
                    if child_value < current_value:
                        population[j] = child
                        fitness[j] = child_fitness
        
        # Store generation history
        history['fitness'].append(fitness)
        
        # Find best individual in this generation (from external population)
        if external_pop:
            best_idx = min(range(len(external_fitness)), key=lambda i: external_fitness[i][-1])
            best_individual = external_pop[best_idx]
            best_schedule = parse_ga_schedule(best_individual, activities_dict, spaces_dict, slots)
            
            # Evaluate best schedule
            _, hard_violations, soft_violations = evaluate_schedule(
                best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
            )
            
            history['best_individuals'].append(best_individual)
            history['hard_violations'].append(hard_violations)
            history['soft_violations'].append(soft_violations)
    
    print("\nMOEA/D Complete!")
    return external_pop, external_fitness

# CELL: Run GA Example
# Uncomment to run different GA variants
# 
# # Load data
# data_tuple = load_data_from_local()
# (
#     activities_dict, groups_dict, spaces_dict, lecturers_dict,
#     activities_list, groups_list, spaces_list, lecturers_list,
#     activity_types, timeslots_list, days_list, periods_list, slots
# ) = data_tuple
# 
# # Run NSGA-II with 4-room dataset
# pop_size = 150
# generations = 200
# population, fitness = nsga2(
#     activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
#     pop_size=pop_size, generations=generations
# )
# 
# # Visualize Pareto front
# fig = visualize_pareto_front(
#     fitness, 
#     metrics_names=[
#         'Space Capacity Violations', 
#         'Lecturer/Group Clashes', 
#         'Space Clashes', 
#         'Soft Constraint Violations', 
#         'Total Fitness'
#     ],
#     title="NSGA-II Pareto Front (4-Room Dataset)"
# )
# plt.show()
# 
# # Get best solution
# best_idx = min(range(len(fitness)), key=lambda i: fitness[i][-1])
# best_schedule = parse_ga_schedule(population[best_idx], activities_dict, spaces_dict, slots)
# 
# # Visualize best schedule
# fig = visualize_schedule(
#     best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
#     title="Best Schedule - NSGA-II (4-Room Dataset)"
# )
# plt.show()
