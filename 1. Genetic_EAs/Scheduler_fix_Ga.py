# scheduler_ga.py
# Consolidated script for GA-based timetable scheduling (NSGA-II, SPEA2, MOEA/D)
# Aligned with multi-objective evaluation from notebooks.

import json
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from pprint import pprint
import time
import os
import math  # For MOEA/D helper

# === Data Classes ===


class Space:
    def __init__(self, code=None, capacity=None, id=None, size=None, **kwargs):
        self.code = code or id
        self.size = size or capacity  # Use size primarily

    def __repr__(self):
        return f"Space(code={self.code}, size={self.size})"


class Group:
    def __init__(self, id, size, **kwargs):
        self.id = id
        self.size = size

    def __repr__(self):
        return f"Group(id={self.id}, size={self.size})"


class Activity:
    def __init__(self, id=None, code=None, subject=None, name=None,
                 teacher_id=None, lecturer=None, teacher_ids=None,  # Added teacher_ids
                 group_ids=None, group=None, subgroup_ids=None,  # Added subgroup_ids
                 duration=1, type=None,  # Added type
                 **kwargs):
        self.id = id or code
        self.subject = subject or name
        # Handle potential list in teacher_ids from JSON
        if teacher_ids and isinstance(teacher_ids, list):
            self.teacher_id = teacher_ids[0]
        else:
            self.teacher_id = teacher_id or (lecturer.id if lecturer else None)
        # Handle potential subgroup_ids from JSON
        self.group_ids = group_ids or subgroup_ids or (
            [group.id] if group else [])
        self.duration = duration
        self.type = type  # Store activity type (e.g., Lecture, Lab)
        self.lecturer = lecturer  # Keep reference if available
        self.group = group  # Keep reference if available

    def __repr__(self):
        return f"Activity(id={self.id}, subject={self.subject}, teacher_id={self.teacher_id}, group_ids={self.group_ids}, duration={self.duration})"


class Lecturer:
    def __init__(self, id, first_name=None, last_name=None, username=None, department=None, **kwargs):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.username = username
        self.department = department

    def __repr__(self):
        return f"Lecturer(id={self.id}, name={self.first_name} {self.last_name}, dept={self.department})"

# === Data Loading ===


def load_data(path):
    """Load scheduling data from a JSON file and return dictionaries for spaces, groups, activities, lecturers, and slots."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Load spaces (classrooms)
    spaces_dict = {}
    for space_data in data.get('spaces', []):
        # Ensure 'size' is prioritized if 'capacity' exists
        if 'capacity' in space_data and 'size' not in space_data:
            space_data['size'] = space_data['capacity']
        spaces_dict[space_data.get(
            'code', space_data.get('id'))] = Space(**space_data)

    # Load student groups
    groups_dict = {}
    for group_data in data.get('years', data.get('groups', [])):
        groups_dict[group_data['id']] = Group(**group_data)

    # Load lecturers/teachers
    lecturers_dict = {}
    for user_data in data.get('users', []):
        if user_data.get('role') == 'lecturer':
            lecturers_dict[user_data['id']] = Lecturer(**user_data)

    # Load activities/courses
    activities_dict = {}
    for activity_data in data.get('activities', []):
        activities_dict[activity_data.get(
            'code', activity_data.get('id'))] = Activity(**activity_data)

    # Generate time slots
    slots = []
    for day in ["MON", "TUE", "WED", "THU", "FRI"]:
        for i in range(1, 9):  # 8 slots per day
            slots.append(f"{day}{i}")

    return spaces_dict, groups_dict, activities_dict, lecturers_dict, slots

# === Hard Constraint Evaluation (Helper) ===


def evaluate_hard_constraints(timetable, activities_dict, groups_dict, spaces_dict):
    """Evaluate hard constraints and return a dictionary of violation counts."""
    violations = {
        'vacant': 0,
        'prof_conflicts': 0,
        'room_size_conflicts': 0,
        'sub_group_conflicts': 0,
        'unassigned': 0
    }
    activities_set = set()

    for slot in timetable:
        prof_set = set()
        sub_group_set = set()
        for room in timetable[slot]:
            activity = timetable[slot][room]

            if not isinstance(activity, Activity):
                violations['vacant'] += 1
                continue

            activities_set.add(activity.id)

            # Lecturer Conflicts
            if activity.teacher_id in prof_set:
                violations['prof_conflicts'] += 1
            prof_set.add(activity.teacher_id)

            # Student Group Conflicts
            for group_id in activity.group_ids:
                if group_id in sub_group_set:
                    violations['sub_group_conflicts'] += 1
                sub_group_set.add(group_id)

            # Room Capacity
            group_size = sum(
                groups_dict[g].size for g in activity.group_ids if g in groups_dict)
            # Use .size attribute from Space class
            if room in spaces_dict and group_size > spaces_dict[room].size:
                violations['room_size_conflicts'] += 1
            elif room not in spaces_dict:
                # Handle case where room ID might be invalid (optional logging)
                # print(f"Warning: Room '{room}' in timetable but not in spaces_dict during hard constraint check.")
                # Penalize if room doesn't exist
                violations['room_size_conflicts'] += 1

    violations['unassigned'] = len(activities_dict) - len(activities_set)
    return violations

# === Soft Constraint Evaluation (Helper) ===


def evaluate_soft_constraints(schedule, groups_dict, lecturers_dict, slots):
    """Evaluates soft constraints and returns a single score (higher is better)."""
    group_fatigue = {g: 0 for g in groups_dict.keys()}
    group_idle_time = {g: 0 for g in groups_dict.keys()}
    group_lecture_spread = {g: 0 for g in groups_dict.keys()}
    lecturer_fatigue = {l: 0 for l in lecturers_dict.keys()}
    lecturer_idle_time = {l: 0 for l in lecturers_dict.keys()}
    lecturer_lecture_spread = {l: 0 for l in lecturers_dict.keys()}
    lecturer_workload = {l: 0 for l in lecturers_dict.keys()}
    group_lecture_slots = {g: [] for g in groups_dict.keys()}
    lecturer_lecture_slots = {l: [] for l in lecturers_dict.keys()}

    for slot, rooms in schedule.items():
        for room, activity in rooms.items():
            if not isinstance(activity, Activity):
                continue

            # Process student groups
            if hasattr(activity, 'group_ids') and isinstance(activity.group_ids, list):
                for group_id in activity.group_ids:
                    if group_id in groups_dict:
                        group_fatigue[group_id] += 1
                        group_lecture_spread[group_id] += 2
                        group_lecture_slots[group_id].append(slot)

            # Process lecturers
            if hasattr(activity, 'teacher_id') and activity.teacher_id in lecturers_dict:
                lecturer_id = activity.teacher_id
                lecturer_fatigue[lecturer_id] += 1
                lecturer_lecture_spread[lecturer_id] += 2
                if hasattr(activity, 'duration'):
                    lecturer_workload[lecturer_id] += activity.duration
                lecturer_lecture_slots[lecturer_id].append(slot)

    # Calculate idle time
    slot_indices = {s: i for i, s in enumerate(slots)}
    num_slots = len(slots)
    if num_slots <= 1:  # Avoid division by zero if only one slot
        return 0.0

    for group_id, lectures in group_lecture_slots.items():
        if lectures:
            indices = sorted([slot_indices[s] for s in lectures])
            idle = sum((indices[i+1] - indices[i] - 1)
                       for i in range(len(indices)-1))
            group_idle_time[group_id] = idle / (num_slots - 1)

    for lecturer_id, lectures in lecturer_lecture_slots.items():
        if lectures:
            indices = sorted([slot_indices[s] for s in lectures])
            idle = sum((indices[i+1] - indices[i] - 1)
                       for i in range(len(indices)-1))
            lecturer_idle_time[lecturer_id] = idle / (num_slots - 1)

    # Normalize metrics
    def normalize(dictionary):
        max_val = max(dictionary.values()) if dictionary else 1
        # Ensure max_val is at least 1 to avoid division by zero
        max_val = max(max_val, 1)
        return {k: v / max_val for k, v in dictionary.items()}

    group_fatigue = normalize(group_fatigue)
    group_idle_time = normalize(group_idle_time)
    group_lecture_spread = normalize(group_lecture_spread)
    lecturer_fatigue = normalize(lecturer_fatigue)
    lecturer_idle_time = normalize(lecturer_idle_time)
    lecturer_lecture_spread = normalize(lecturer_lecture_spread)

    # Calculate workload balance
    workload_values = np.array(list(lecturer_workload.values()))
    lecturer_workload_balance = 1.0
    if len(workload_values) > 1:
        mean_workload = np.mean(workload_values)
        if mean_workload > 0:  # Avoid division by zero
            variance = np.var(workload_values)
            lecturer_workload_balance = max(0, 1 - (variance / mean_workload))

    # Calculate final scores
    student_fatigue_score = np.mean(
        list(group_fatigue.values())) if group_fatigue else 0
    student_idle_time_score = np.mean(
        list(group_idle_time.values())) if group_idle_time else 0
    student_lecture_spread_score = np.mean(
        list(group_lecture_spread.values())) if group_lecture_spread else 0
    lecturer_fatigue_score = np.mean(
        list(lecturer_fatigue.values())) if lecturer_fatigue else 0
    lecturer_idle_time_score = np.mean(
        list(lecturer_idle_time.values())) if lecturer_idle_time else 0
    lecturer_lecture_spread_score = np.mean(
        list(lecturer_lecture_spread.values())) if lecturer_lecture_spread else 0

    # Final weighted score (higher is better)
    final_score = (
        (1 - student_fatigue_score) * 0.2 +        # Minimize fatigue
        (1 - student_idle_time_score) * 0.2 +      # Minimize idle time
        # Minimize spread (prefer compact)
        (1 - student_lecture_spread_score) * 0.2 +
        (1 - lecturer_fatigue_score) * 0.1 +       # Minimize fatigue
        (1 - lecturer_idle_time_score) * 0.1 +     # Minimize idle time
        (1 - lecturer_lecture_spread_score) * 0.1 +  # Minimize spread
        lecturer_workload_balance * 0.1            # Maximize balance
    )
    return final_score


# === Multi-Objective Evaluator ===
# This evaluator returns a tuple suitable for multi-objective algorithms like NSGA-II
# Objectives are minimized: prof_conflicts, sub_group_conflicts, room_size_conflicts, unassigned, (1 - soft_score)
NUM_OBJECTIVES_GA = 5  # Define number of objectives for GA variants


def multi_objective_evaluator(timetable, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots):
    """Evaluate a timetable and return a tuple of objectives to be minimized."""
    hard_violations = evaluate_hard_constraints(
        timetable, activities_dict, groups_dict, spaces_dict)
    soft_score = evaluate_soft_constraints(
        timetable, groups_dict, lecturers_dict, slots)  # Higher is better

    # Objectives to minimize:
    # 1. Professor Conflicts
    # 2. Sub-group Conflicts
    # 3. Room Size Conflicts
    # 4. Unassigned Activities
    # 5. Inverted Soft Score (1 - score, so minimizing this maximizes the original score)
    return (
        hard_violations['prof_conflicts'],
        hard_violations['sub_group_conflicts'],
        hard_violations['room_size_conflicts'],
        hard_violations['unassigned'],
        1.0 - soft_score  # Invert soft score for minimization
    )

# === Helper: Get Class Size ===


def get_classsize(activity: Activity, groups_dict_local) -> int:
    """Calculate the total class size for an activity."""
    if not hasattr(activity, 'group_ids') or not activity.group_ids:
        return 0
    return sum(groups_dict_local[gid].size for gid in activity.group_ids if gid in groups_dict_local)

# === Generate Initial Population (Constrained Approach) ===


def generate_initial_population(pop_size, slots, activities_dict, spaces_dict, groups_dict_local):
    """Generate an initial population using a more constrained approach."""
    population = []
    all_activity_ids = list(activities_dict.keys())  # Use IDs

    for _ in range(pop_size):
        timetable = {slot: {room: None for room in spaces_dict}
            for slot in slots}
        activity_slots = {activity_id: [] for activity_id in all_activity_ids}
        # Create a list of (activity_id, duration) tuples to track remaining duration
        activities_to_schedule = []
        for act_id, activity in activities_dict.items():
            # Add activity ID for each hour of duration
            activities_to_schedule.extend([act_id] * activity.duration)

        # Shuffle the list of activity hours
        random.shuffle(activities_to_schedule)

        scheduled_count = 0
        for activity_id in activities_to_schedule:
            activity = activities_dict[activity_id]
            activity_size = get_classsize(activity, groups_dict_local)

            # Find potential slots/rooms
            potential_placements = []
            for slot in slots:
                # Check if this activity ID has already been scheduled in this slot (for multi-hour activities)
                if slot in activity_slots[activity_id]:
                     continue

                 # Check for lecturer/group conflicts in this slot
                lecturer_busy = False
                group_busy = False
                for r, assigned_act in timetable[slot].items():
                    if assigned_act:
                        if assigned_act.teacher_id == activity.teacher_id:
                            lecturer_busy = True
                            break
                        if any(g in assigned_act.group_ids for g in activity.group_ids):
                            group_busy = True
                            break
                if lecturer_busy or group_busy:
                    continue

                # Find suitable rooms
                for room_id, space in spaces_dict.items():
                    if timetable[slot][room_id] is None and space.size >= activity_size:
                        potential_placements.append((slot, room_id))

            # Assign if a placement is found
            if potential_placements:
                chosen_slot, chosen_room = random.choice(potential_placements)
                timetable[chosen_slot][chosen_room] = activity
                activity_slots[activity_id].append(chosen_slot)  # Mark slot as used for this activity ID
                scheduled_count += 1
            # else: activity remains unassigned for this hour

        # print(f"Initial individual: Scheduled {scheduled_count}/{len(activities_to_schedule)} activity hours.")
        population.append(timetable)
    return population

# === Crossover ===

def crossover(parent1, parent2):
    """Perform crossover by swapping time slots between two parents."""
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    slots_list = list(parent1.keys())
    if len(slots_list) <= 1:  # Cannot perform crossover with 0 or 1 slot
        return child1, child2
    split = random.randint(1, len(slots_list) - 1)  # Ensure split point is valid
    for i in range(split, len(slots_list)):
        slot = slots_list[i]
        # Ensure slots exist in both children before swapping
        if slot in child1 and slot in child2:
            child1[slot], child2[slot] = parent2.get(
                slot, {}), parent1.get(slot, {})
    return child1, child2

# === Mutation ===

def mutate(individual, activities_dict_local, slots_local, spaces_dict_local):
    """Perform mutation by randomly swapping activities in the timetable."""
    # Ensure individual is mutable
    mutated_individual = copy.deepcopy(individual)
    slots_list = list(mutated_individual.keys())

    if len(slots_list) < 2:  # Need at least two slots to swap
        return mutated_individual

    # Select two different random slots
    slot1, slot2 = random.sample(slots_list, 2)

    # Get available rooms in each slot
    rooms1 = list(mutated_individual.get(slot1, {}).keys())
    rooms2 = list(mutated_individual.get(slot2, {}).keys())

    if not rooms1 or not rooms2:  # Need rooms in both slots to swap
        return mutated_individual

    # Select a random room from each slot
    room1 = random.choice(rooms1)
    room2 = random.choice(rooms2)

    # Swap the activities (or None) between the selected room/slot combinations
    mutated_individual[slot1][room1], mutated_individual[slot2][room2] = \
        mutated_individual[slot2].get(
            room2), mutated_individual[slot1].get(room1)

    return mutated_individual

# === Pareto Front Helper ===


def pareto_frontier(points):
    """Compute the Pareto frontier for a set of points (for minimization)."""
    # Convert to NumPy array immediately
    pts = np.array(points)

    # Check if the NumPy array is empty by checking its size
    if pts.size == 0:
        return np.array([], dtype=bool)  # Return an empty boolean array

    # Ensure pts is 2D, even if only one point is passed
    if pts.ndim == 1:
        # This case might occur if only one solution is passed.
        # Reshape assumes objectives are columns.
        # If pts was originally [obj1, obj2, ...], reshape to [[obj1, obj2, ...]]
        if len(pts) > 0:  # Check if it's not truly empty after conversion
            pts = pts.reshape(1, -1)
        else:  # If conversion resulted in a truly empty 1D array
            return np.array([], dtype=bool)

    # Initialize all points as potentially being on the frontier
    is_pf = np.ones(pts.shape[0], dtype=bool)

    # Compare each point against all other points
    for i, p in enumerate(pts):
        # Check if point p is dominated by any other point q
        # A point p is dominated if there exists another point q such that:
        # q[k] <= p[k] for all objectives k, AND q[j] < p[j] for at least one objective j.
        # Vectorized check:
        # Check points before i
        if i > 0:
            if np.any(np.all(pts[:i] <= p, axis=1) & np.any(pts[:i] < p, axis=1)):
                is_pf[i] = False
                continue  # Move to the next point i if dominated
        # Check points after i
        if i < pts.shape[0] - 1:
            if np.any(np.all(pts[i+1:] <= p, axis=1) & np.any(pts[i+1:] < p, axis=1)):
                is_pf[i] = False

    return is_pf

# === Dominance Check ===

def dominates(fitness1, fitness2):
    """Return True if fitness1 dominates fitness2 (minimization)."""
    # Ensure both are tuples/lists of the same length
    if len(fitness1) != len(fitness2):
        raise ValueError(
            "Fitness vectors must have the same number of objectives.")
    # Check if fitness1 is better or equal in all objectives
    all_le = all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2))
    # Check if fitness1 is strictly better in at least one objective
    any_lt = any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))
    return all_le and any_lt

# === Non-dominated Sorting ===

def non_dominated_sort(population_indices, fitness_values):
    """Sort population indices into fronts based on dominance."""
    n = len(population_indices)
    if n == 0:
        return []

    # Use actual fitness values for comparison
    pop_fitness = [fitness_values[i] for i in population_indices]

    fronts = [[]]
    S = [[] for _ in range(n)]  # Solutions dominated by p
    n_p = [0] * n               # Domination count for p
    rank = [0] * n

    for p_idx in range(n):
        p = population_indices[p_idx]
        for q_idx in range(n):
            if p_idx == q_idx:
                continue
            q = population_indices[q_idx]

            if dominates(pop_fitness[p_idx], pop_fitness[q_idx]):
                S[p_idx].append(q_idx)  # Add index within the current subset
            elif dominates(pop_fitness[q_idx], pop_fitness[p_idx]):
                n_p[p_idx] += 1

        if n_p[p_idx] == 0:
            rank[p_idx] = 0
            fronts[0].append(population_indices[p_idx])  # Store original index

    i = 0
    while fronts[i]:
        next_front = []
        for p_orig_idx in fronts[i]:
            # Find the index of p_orig_idx within the current population_indices subset
            p_subset_idx = population_indices.index(p_orig_idx)
            for q_subset_idx in S[p_subset_idx]:
                n_p[q_subset_idx] -= 1
                if n_p[q_subset_idx] == 0:
                    rank[q_subset_idx] = i + 1
                    next_front.append(population_indices[q_subset_idx])  # Store original index
        i += 1
        if not next_front:  # Avoid adding empty list if last front was processed
            break
        fronts.append(next_front)

    return fronts

# === Crowding Distance ===

def crowding_distance(fitness_values, front_indices):
    """Calculate crowding distance for solutions in a front (list of indices)."""
    if not front_indices:
        return {}
    distances = {i: 0.0 for i in front_indices}
    num_objectives = len(fitness_values[front_indices[0]])
    num_in_front = len(front_indices)

    if num_in_front <= 2:
        for i in front_indices:
            distances[i] = float('inf')
        return distances

    for m in range(num_objectives):
        # Sort front indices by this objective
        sorted_indices = sorted(
            front_indices, key=lambda i: fitness_values[i][m])

        # Assign infinite distance to boundary points
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        # Get min/max values for normalization
        min_val = fitness_values[sorted_indices[0]][m]
        max_val = fitness_values[sorted_indices[-1]][m]
        range_val = max_val - min_val

        if range_val == 0:  # Avoid division by zero if all values are the same
            continue

        # Calculate distance for intermediate points
        for i in range(1, num_in_front - 1):
            # Check if the index is already infinite (boundary point)
            if distances[sorted_indices[i]] != float('inf'):
                 distances[sorted_indices[i]] += (fitness_values[sorted_indices[i+1]][m] -
                                                  fitness_values[sorted_indices[i-1]][m]) / range_val

    return distances

# === Selection (NSGA-II) ===

def selection(population, fitness_values, pop_size):
    """Select parents for the next generation using NSGA-II logic."""
    all_indices = list(range(len(population)))
    fronts = non_dominated_sort(all_indices, fitness_values)

    next_generation_indices = []
    current_front_idx = 0

    # Add individuals from fronts until population size is reached
    while len(next_generation_indices) + len(fronts[current_front_idx]) <= pop_size:
        next_generation_indices.extend(fronts[current_front_idx])
        current_front_idx += 1
        if current_front_idx >= len(fronts):  # Stop if all fronts are added
            break

    # If more individuals are needed, use crowding distance on the last front
    if len(next_generation_indices) < pop_size and current_front_idx < len(fronts):
        last_front_indices = fronts[current_front_idx]
        distances = crowding_distance(fitness_values, last_front_indices)

        # Sort by crowding distance (descending)
        sorted_last_front = sorted(
            last_front_indices, key=lambda i: distances[i], reverse=True)

        needed = pop_size - len(next_generation_indices)
        next_generation_indices.extend(sorted_last_front[:needed])

    # Return the selected individuals from the original population
    return [population[i] for i in next_generation_indices]

# === NSGA-II Main Loop ===

def nsga2(pop_size, generations, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots):
    """Main NSGA-II algorithm."""
    print(f"\n--- Running NSGA-II (Pop: {pop_size}, Gen: {generations}) ---")
    # Use global constants for rates or pass them as arguments
    CROSSOVER_RATE_NSGA = 0.8
    MUTATION_RATE_NSGA = 0.1

    # 1. Initialize population
    population = generate_initial_population(
        pop_size, slots, activities_dict, spaces_dict, groups_dict)
    fitness = [multi_objective_evaluator(
        ind, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots) for ind in population]

    for gen in range(generations):
        if gen % 10 == 0:
            print(f"NSGA-II Generation {gen + 1}/{generations}")

        # 2. Create Offspring
        offspring = []
        while len(offspring) < pop_size:
            # Select parents using binary tournament based on rank and crowding distance
            # (Simplified: random selection from current pop for this example)
            p1_idx, p2_idx = random.sample(range(len(population)), 2)
            p1, p2 = population[p1_idx], population[p2_idx]

            if random.random() < CROSSOVER_RATE_NSGA:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            if random.random() < MUTATION_RATE_NSGA:
                c1 = mutate(c1, activities_dict, slots, spaces_dict)
            if random.random() < MUTATION_RATE_NSGA:
                c2 = mutate(c2, activities_dict, slots, spaces_dict)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # 3. Evaluate offspring
        offspring_fitness = [multi_objective_evaluator(
            ind, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots) for ind in offspring]

        # 4. Combine parent and offspring populations
        combined_pop = population + offspring
        combined_fitness = fitness + offspring_fitness

        # 5. Select next generation using non-dominated sort and crowding distance
        population = selection(combined_pop, combined_fitness, pop_size)

        # 6. Update fitness for the new population
        fitness = [multi_objective_evaluator(
            ind, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots) for ind in population]

    print("NSGA-II Complete!")
    return population, fitness

# === SPEA2 Helpers (Copied/Adapted from scheduler_ga.py) ===
# (Assuming these functions correctly handle the 5-objective fitness tuple)


def calculate_strength_spea2(combined_fitness):
    """Calculate strength values for SPEA2 based on a list of fitness values."""
    n = len(combined_fitness)
    strength = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(combined_fitness[i], combined_fitness[j]):
                strength[i] += 1
    return strength


def calculate_raw_fitness_spea2(combined_fitness, strength):
    """Calculate raw fitness values for SPEA2."""
    n = len(combined_fitness)
    raw_fitness = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(combined_fitness[j], combined_fitness[i]):
                raw_fitness[i] += strength[j]
    return raw_fitness


def calculate_density_spea2(combined_fitness):
    """Calculate density values for SPEA2 using k-nearest neighbor."""
    n = len(combined_fitness)
    if n <= 1:
        return np.zeros(n)

    k = int(np.sqrt(n))  # k is typically sqrt(n)
    k = max(1, min(k, n - 1))  # Ensure k is valid

    density = np.zeros(n)
    # Calculate distances between all pairs
    distances_all = np.zeros((n, n))
    # Convert list of tuples to numpy array
    fitness_array = np.array(combined_fitness)

    # Efficiently calculate pairwise distances
    # This requires fitness_array to be purely numeric. Ensure evaluator returns numeric tuples.
    try:
        # Ensure fitness_array is float
        fitness_array = fitness_array.astype(float)
        diff = fitness_array[:, np.newaxis, :] - \
            fitness_array[np.newaxis, :, :]
        distances_all = np.sqrt(np.sum(diff**2, axis=-1))
    except (ValueError, TypeError) as e:
        print(
            f"Error calculating distances, ensure fitness values are numeric: {e}")
        # Fallback to slower method if vectorization fails
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(fitness_array[i] - fitness_array[j])
                distances_all[i, j] = distances_all[j, i] = dist

    # Calculate density based on k-th neighbor distance
    for i in range(n):
        sorted_distances = np.sort(distances_all[i])
        # k-th distance is at index k (since index 0 is distance to self=0)
        if len(sorted_distances) > k:
            kth_distance = sorted_distances[k]
            # Add constant to avoid division by zero
            density[i] = 1.0 / (kth_distance + 2.0)
        else:
            # Assign high density if less than k points
            density[i] = float('inf')

    return density

def environmental_selection(combined_pop, combined_fitness, archive_size):
    """Selects the next archive based on SPEA2 fitness."""
    n_combined = len(combined_pop)
    if n_combined == 0:
        return [], []

    # Calculate SPEA2 fitness components
    strength = calculate_strength_spea2(combined_fitness)
    raw_fitness = calculate_raw_fitness_spea2(combined_fitness, strength)
    density = calculate_density_spea2(combined_fitness)
    final_fitness = raw_fitness + density # Lower is better

    # Identify non-dominated solutions (raw_fitness == 0)
    non_dominated_indices = [i for i, raw in enumerate(raw_fitness) if raw == 0]

    next_archive_indices = []
    if len(non_dominated_indices) <= archive_size:
        # If fewer non-dominated than archive size, add them all
        next_archive_indices.extend(non_dominated_indices)
        # Fill remaining slots with best dominated solutions
        if len(non_dominated_indices) < archive_size:
            dominated_indices = [i for i, raw in enumerate(raw_fitness) if raw > 0]
            # Sort dominated by final_fitness (lower is better)
            dominated_indices.sort(key=lambda i: final_fitness[i])
            needed = archive_size - len(next_archive_indices)
            next_archive_indices.extend(dominated_indices[:needed])
    else:
        # If more non-dominated than archive size, truncate based on density
        # We need to remove the ones with the *lowest* density value (most crowded)
        # Density here is 1/(dist+2), so lower density means higher distance (less crowded)
        # We want to remove those with the *highest* density value (closest neighbors)
        non_dom_final_fitness = {i: final_fitness[i] for i in non_dominated_indices}
        # Sort by final fitness (includes density) - lower is better
        sorted_non_dom = sorted(non_dominated_indices, key=lambda i: non_dom_final_fitness[i])
        next_archive_indices = sorted_non_dom[:archive_size] # Keep the best ones

    # Return the selected individuals and their fitness
    next_archive = [combined_pop[i] for i in next_archive_indices]
    next_archive_fitness = [combined_fitness[i] for i in next_archive_indices]

    return next_archive, next_archive_fitness

def select_mating_pool(archive, archive_fitness, pool_size):
    """Selects mating pool from the archive using binary tournament."""
    mating_pool = []
    n_archive = len(archive)

    if n_archive == 0:
        # Handle empty archive - maybe return random individuals from population?
        # For now, return empty pool. This needs robust handling.
        print("Warning: SPEA2 archive is empty during mating selection.")
        return []
    if n_archive == 1:
        # If only one in archive, duplicate it for the pool
        return [archive[0]] * pool_size

    # Calculate fitness for tournament (lower is better)
    # Need strength, raw, density for archive members relative to themselves + population?
    # Simpler: Use the final_fitness calculated during environmental selection if available
    # For now, use a simple tournament based on the objectives directly
    # This assumes archive_fitness contains the multi-objective tuples
    for _ in range(pool_size):
        idx1, idx2 = random.sample(range(n_archive), 2)
        # Compare based on dominance, then crowding (or just dominance for simplicity)
        if dominates(archive_fitness[idx1], archive_fitness[idx2]):
            mating_pool.append(archive[idx1])
        elif dominates(archive_fitness[idx2], archive_fitness[idx1]):
            mating_pool.append(archive[idx2])
        else: # If non-dominating, choose randomly or based on density (not calculated here)
            mating_pool.append(random.choice([archive[idx1], archive[idx2]]))

    return mating_pool

# === SPEA2 Main Loop ===


def spea2(pop_size, archive_size, generations, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots):
    """SPEA2 implementation."""
    print(
        f"\n--- Running SPEA2 (Pop: {pop_size}, Archive: {archive_size}, Gen: {generations}) ---")
    CROSSOVER_RATE_SPEA = 0.8
    MUTATION_RATE_SPEA = 0.1

    # 1. Initialize population and empty archive (stores individuals)
    population = generate_initial_population(
        pop_size, slots, activities_dict, spaces_dict, groups_dict)
    archive = []  # Stores individuals
    archive_fitness = []  # Stores fitness corresponding to archive individuals

    for gen in range(generations):
        if gen % 10 == 0:
            print(f"SPEA2 Generation {gen + 1}/{generations}")

        # 2. Evaluate current population
        population_fitness = [multi_objective_evaluator(
            ind, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots) for ind in population]

        # 3. Combine population and archive individuals
        combined_individuals = population + archive
        combined_fitness = population_fitness + archive_fitness
        n_combined = len(combined_individuals)

        if n_combined == 0:  # Handle case where both pop and archive might be empty initially or after selection
            print(
                f"Warning: Combined population empty in generation {gen}. Re-initializing.")
            population = generate_initial_population(
                pop_size, slots, activities_dict, spaces_dict, groups_dict)
            archive = []
            archive_fitness = []
            continue  # Skip to next generation

        # 4. Calculate Fitness Components for combined set using CORRECT function names
        strength = calculate_strength_spea2(
            combined_fitness)  # FIX: Added _spea2
        raw_fitness = calculate_raw_fitness_spea2(
            combined_fitness, strength)  # FIX: Added _spea2
        density = calculate_density_spea2(
            combined_fitness)  # FIX: Added _spea2
        final_fitness = raw_fitness + density  # Lower is better

        # Map fitness values to individuals for easier lookup during selection
        individual_final_fitness = {
            i: final_fitness[i] for i in range(n_combined)}

        # 5. Environmental Selection: Select next archive from combined set
        archive, archive_fitness = environmental_selection(
            combined_individuals, combined_fitness, archive_size)

        # 6. Mating Selection: Select mating pool from the *new* archive
        mating_pool = select_mating_pool(archive, archive_fitness, pop_size)

        # Handle empty mating pool
        if not mating_pool:
            print(
                f"Warning: Mating pool empty in generation {gen}. Using current population for next gen.")
            temp_mating_pool = []
            if population:
                 for _ in range(pop_size):
                     temp_mating_pool.append(random.choice(population))
                 mating_pool = temp_mating_pool
            else:
                 print(
                     "Error: Both population and archive are empty after selection. Re-initializing population.")
                 population = generate_initial_population(
                     pop_size, slots, activities_dict, spaces_dict, groups_dict)
                 archive = []
                 archive_fitness = []
                 continue

        # 7. Create Offspring for the next population
        next_population = []
        while len(next_population) < pop_size:
            if len(mating_pool) >= 2:
                p1, p2 = random.sample(mating_pool, 2)
            elif len(mating_pool) == 1:
                p1 = p2 = mating_pool[0]
            else:
                print(
                    "Error: Mating pool became unexpectedly empty during offspring creation.")
                break  # Avoid infinite loop

            if random.random() < CROSSOVER_RATE_SPEA:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            if random.random() < MUTATION_RATE_SPEA:
                c1 = mutate(c1, activities_dict, slots, spaces_dict)
            if random.random() < MUTATION_RATE_SPEA:
                c2 = mutate(c2, activities_dict, slots, spaces_dict)

            next_population.append(c1)
            if len(next_population) < pop_size:
                next_population.append(c2)

        population = next_population  # Update population for the next iteration

    # Final evaluation and return
    print("SPEA2 Complete!")
    # Return the final archive members and their fitness
    return archive, archive_fitness
# === MOEA/D Helpers (Copied/Adapted from scheduler_ga.py) ===
# (Assuming these functions correctly handle the 5-objective fitness tuple)

def generate_weight_vectors(n_weights, n_objectives):
    """Generate weight vectors using Dirichlet distribution."""
    # Use numpy's newer random Generator for better randomness
    rng = np.random.default_rng(42)  # Seed for reproducibility
    weights = rng.dirichlet(np.ones(n_objectives), size=n_weights)
    return weights


def update_ideal_point(fitness_values, ideal_point):
    """Update the ideal point (minimum values for each objective)."""
    current_ideal = np.array(ideal_point)
    for fitness in fitness_values:
        current_ideal = np.minimum(current_ideal, np.array(fitness))
    return current_ideal.tolist()  # Return as list


def calculate_tchebycheff(weight, fitness, ideal_point):
    """Calculate the Tchebycheff scalarization value."""
    # Weighted difference from the ideal point
    weighted_diff = weight * np.abs(np.array(fitness) - np.array(ideal_point))
    return np.max(weighted_diff)

# === MOEA/D Main Loop ===


def moead(pop_size, generations, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots, n_neighbors=15, max_replacements=2):
    """MOEA/D implementation."""
    print(
        # Added Max Replace to print
        f"\n--- Running MOEA/D (Pop: {pop_size}, Gen: {generations}, Neighbors: {n_neighbors}, Max Replace: {max_replacements}) ---")
    NUM_OBJECTIVES_MOEAD = 5  # Match the multi-objective evaluator
    CROSSOVER_RATE_MOEA = 0.8
    MUTATION_RATE_MOEA = 0.1
    # MODIFICATION: Removed hardcoded MAX_REPLACEMENTS = 2

    # ... (rest of initialization remains the same) ...
    population = generate_initial_population(
        pop_size, slots, activities_dict, spaces_dict, groups_dict)
    fitness_values = [multi_objective_evaluator(
        ind, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots) for ind in population]
    weight_vectors = generate_weight_vectors(pop_size, NUM_OBJECTIVES_MOEAD)
    ideal_point = [float('inf')] * NUM_OBJECTIVES_MOEAD
    ideal_point = update_ideal_point(fitness_values, ideal_point)
    distances = np.zeros((pop_size, pop_size))
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dist = np.linalg.norm(weight_vectors[i] - weight_vectors[j])
            distances[i, j] = distances[j, i] = dist
    neighborhoods = [np.argsort(distances[i])[:n_neighbors].tolist()
                     for i in range(pop_size)]
    archive = []
    archive_fitness = []
    MAX_ARCHIVE_SIZE = pop_size

    # 2. Main Loop
    for gen in range(generations):
        if gen % 10 == 0:
            print(f"MOEA/D Generation {gen + 1}/{generations}")

        permutation = list(range(pop_size))
        random.shuffle(permutation)

        for i in permutation:  # Iterate through subproblems
            # ... (parent selection, crossover, mutation, child evaluation, ideal point update remain the same) ...
            if len(neighborhoods[i]) < 2:
                k, l = random.sample(range(pop_size), 2)
            else:
                k, l = random.sample(neighborhoods[i], 2)

            if random.random() < CROSSOVER_RATE_MOEA:
                c1, c2 = crossover(population[k], population[l])
                child = c1
            else:
                child = copy.deepcopy(population[k])

            if random.random() < MUTATION_RATE_MOEA:
                child = mutate(child, activities_dict, slots, spaces_dict)

            child_fitness = multi_objective_evaluator(
                child, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots)

            ideal_point = update_ideal_point([child_fitness], ideal_point)

            # Update neighborhood solutions
            replacements_done = 0
            for j in neighborhoods[i]:
                child_scalar = calculate_tchebycheff(
                    weight_vectors[j], child_fitness, ideal_point)
                neighbor_scalar = calculate_tchebycheff(
                    weight_vectors[j], fitness_values[j], ideal_point)

                if child_scalar < neighbor_scalar:
                    population[j] = copy.deepcopy(child)
                    fitness_values[j] = child_fitness
                    replacements_done += 1
                    # MODIFICATION: Use the max_replacements argument
                    if replacements_done >= max_replacements:
                        break

            # ... (archive update logic remains the same) ...
            is_dominated_by_archive = False
            indices_to_remove = []
            for arc_idx, arc_fit in enumerate(archive_fitness):
                if dominates(arc_fit, child_fitness):
                    is_dominated_by_archive = True
                    break
                if dominates(child_fitness, arc_fit):
                    indices_to_remove.append(arc_idx)

            for idx in sorted(indices_to_remove, reverse=True):
                archive.pop(idx)
                archive_fitness.pop(idx)

            if not is_dominated_by_archive:
                archive.append(copy.deepcopy(child))
                archive_fitness.append(child_fitness)
                if len(archive) > MAX_ARCHIVE_SIZE:
                    idx_to_remove = random.randrange(len(archive))
                    archive.pop(idx_to_remove)
                    archive_fitness.pop(idx_to_remove)

    # ... (Combine final population, sort, return Pareto front remain the same) ...
    final_population = population + archive
    final_fitness = fitness_values + archive_fitness
    final_indices = list(range(len(final_population)))
    final_fronts = non_dominated_sort(final_indices, final_fitness)
    pareto_indices = final_fronts[0] if final_fronts else []
    pareto_solutions = [final_population[i] for i in pareto_indices]
    pareto_fitness = [final_fitness[i] for i in pareto_indices]

    print("MOEA/D Complete!")
    return pareto_solutions, pareto_fitness

# === Plotting Function ===


def plot_pareto_2d_projection(fitness_values, name, output_dir,
                               obj_x_index, obj_y_index,
                               obj_x_label, obj_y_label):
    """
    Plots a 2D projection of the multi-objective fitness values,
    highlighting the solutions on the true multi-dimensional Pareto front.

    Args:
        fitness_values: List of multi-objective fitness tuples/lists.
        name: Name of the algorithm for the title.
        output_dir: Directory to save the plot.
        obj_x_index: Index of the objective for the X-axis.
        obj_y_index: Index of the objective for the Y-axis.
        obj_x_label: Label for the X-axis.
        obj_y_label: Label for the Y-axis.
    """
    if not fitness_values:
        print(
            f"No fitness values to plot for {name} ({obj_x_label} vs {obj_y_label}).")
        return

    fit = np.array(fitness_values)

    if fit.ndim == 1:  # Handle case with only one solution
        if fit.shape[0] >= max(obj_x_index, obj_y_index) + 1:
            fit = fit.reshape(1, -1)
        else:
            print(
                f"Warning: Single fitness vector for {name} has too few objectives ({fit.shape[0]}) for requested plot ({obj_x_index} vs {obj_y_index}). Skipping plot.")
            return
    elif fit.shape[1] < max(obj_x_index, obj_y_index) + 1:
        print(
            f"Warning: Fitness values for {name} have too few objectives ({fit.shape[1]}) for requested plot ({obj_x_index} vs {obj_y_index}). Skipping plot.")
        return

    # --- Identify the TRUE Pareto Front in the original N-dimensional space ---
    all_indices = list(range(len(fit)))
    # Use non_dominated_sort to find the indices of the first front
    # Pass original fitness list
    fronts = non_dominated_sort(all_indices, fitness_values)
    pareto_indices = fronts[0] if fronts else []
    # ---------------------------------------------------------------------

    # Extract the specific objectives for plotting
    all_x = fit[:, obj_x_index]
    all_y = fit[:, obj_y_index]

    pareto_x = fit[pareto_indices, obj_x_index] if pareto_indices else np.array([
    ])
    pareto_y = fit[pareto_indices, obj_y_index] if pareto_indices else np.array([
    ])

    plt.figure(figsize=(10, 7))  # Slightly larger figure

    # Plot all solutions
    plt.scatter(all_x, all_y, alpha=0.6, s=40,
                label='All Solutions (Projected)')  # Adjusted alpha/size

    # Plot the true Pareto front solutions
    if pareto_x.size > 0:  # Check if Pareto front is not empty
        plt.scatter(pareto_x, pareto_y, c='red', s=60,
                    label='True Pareto Front (Projected)')  # Larger size

    plt.xlabel(obj_x_label)
    plt.ylabel(obj_y_label)
    plt.title(f'{name} Pareto Front ({obj_x_label} vs {obj_y_label})')
    plt.legend()
    plt.grid(True)

    # Create safe filename
    safe_filename = name.lower().replace(
        '/', '_').replace('\\', '_').replace(' ', '_')
    plot_filename = f'{safe_filename}_pareto_{obj_x_index}v{obj_y_index}.png'
    plot_path = os.path.join(output_dir, plot_filename)

    try:
        plt.savefig(plot_path)
        print(f"Pareto plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving plot {plot_path}: {e}")

    # plt.show() # Comment out to prevent blocking in scripts
    plt.close()  # Close figure to free memory


def plot_combined_pareto(fitness_sets, names, colors, markers, output_dir,
                         obj_x_index, obj_y_index,
                         obj_x_label, obj_y_label):
    """
    Plots a 2D projection comparing the Pareto fronts from multiple algorithms.

    Args:
        fitness_sets: A list where each element is the list of fitness tuples
                      for one algorithm's final non-dominated set.
        names: A list of algorithm names corresponding to fitness_sets.
        colors: A list of colors for plotting each algorithm.
        markers: A list of markers for plotting each algorithm.
        output_dir: Directory to save the plot.
        obj_x_index, obj_y_index: Indices for X and Y axes.
        obj_x_label, obj_y_label: Labels for axes.
    """
    plt.figure(figsize=(10, 7))

    if len(fitness_sets) != len(names) or len(fitness_sets) != len(colors) or len(fitness_sets) != len(markers):
        print("Error: Mismatch in lengths of fitness_sets, names, colors, or markers.")
        return

    for i, fitness_values in enumerate(fitness_sets):
        if not fitness_values:
            print(f"No fitness values to plot for {names[i]}.")
            continue

        fit = np.array(fitness_values)

        if fit.ndim == 1:  # Handle single solution case
            if fit.shape[0] >= max(obj_x_index, obj_y_index) + 1:
                fit = fit.reshape(1, -1)
            else:
                continue  # Skip if not enough objectives
        elif fit.shape[0] == 0:  # Handle empty array
            continue
        elif fit.shape[1] < max(obj_x_index, obj_y_index) + 1:
            print(f"Warning: Fitness values for {names[i]} have too few objectives. Skipping.")
            continue

        # Extract the specific objectives for plotting
        plot_x = fit[:, obj_x_index]
        plot_y = fit[:, obj_y_index]

        # Plot only the Pareto points for this algorithm
        plt.scatter(plot_x, plot_y, alpha=0.8, s=60,  # Slightly larger points
                    c=colors[i], marker=markers[i], label=f'{names[i]} Pareto Front')

    plt.xlabel(obj_x_label)
    plt.ylabel(obj_y_label)
    plt.title(
        f'Comparison of EA Pareto Fronts ({obj_x_label} vs {obj_y_label})')
    # Only add legend if there's something to plot
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True)

    # Create safe filename
    plot_filename = f'combined_ea_pareto_{obj_x_index}v{obj_y_index}.png'
    plot_path = os.path.join(output_dir, plot_filename)

    try:
        plt.savefig(plot_path)
        print(f"Combined Pareto plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving combined plot {plot_path}: {e}")

    plt.close()  # Close figure


# === Main Execution Block ===
if __name__ == "__main__":
    # Configuration
    DATA_FILE = 'sliit_computing_dataset.json'
    OUTPUT_DIR_BASE = 'ga_results_multi_obj'
    POP_SIZE_MAIN = 50
    GENERATIONS_MAIN = 50
    ARCHIVE_SIZE_MAIN = 50  # For SPEA2

    # --- Parameters for Investigation ---
    # STEP 4 & 5: Modify these values and re-run the script to experiment
    N_NEIGHBORS_MAIN = 15  # MOEA/D: Try values like 5, 15, 25
    MAX_REPLACEMENTS_MAIN = 2  # MOEA/D: Try values like 2, 5, 10, or N_NEIGHBORS_MAIN
    # ------------------------------------

    # Load data
    try:
        print("Loading data...")
        spaces_dict, groups_dict, activities_dict, lecturers_dict, slots = load_data(
            DATA_FILE)
        print(f"Loaded {len(activities_dict)} activities, {len(spaces_dict)} spaces, {len(groups_dict)} groups, {len(lecturers_dict)} lecturers")
        print(f"Defined {len(slots)} time slots")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    objective_labels = [
        "Prof Conflicts", "Sub-group Conflicts", "Room Size Conflicts",
        "Unassigned Activities", "1.0 - Soft Score"
    ]

    # --- Run Algorithms ---
    # NSGA-II
    # ... (NSGA-II code remains the same) ...
    output_dir_nsga2 = os.path.join(OUTPUT_DIR_BASE, 'nsga2')
    os.makedirs(output_dir_nsga2, exist_ok=True)
    nsga2_final_pop, nsga2_final_fitness = nsga2(
        pop_size=POP_SIZE_MAIN, generations=GENERATIONS_MAIN, activities_dict=activities_dict,
        groups_dict=groups_dict, lecturers_dict=lecturers_dict, spaces_dict=spaces_dict, slots=slots
    )
    nsga2_indices = list(range(len(nsga2_final_fitness)))
    nsga2_fronts = non_dominated_sort(nsga2_indices, nsga2_final_fitness)
    nsga2_pareto_indices = nsga2_fronts[0] if nsga2_fronts else []
    nsga2_pareto_fitness = [nsga2_final_fitness[i]
                            for i in nsga2_pareto_indices]

    # SPEA2
    # ... (SPEA2 code remains the same) ...
    output_dir_spea2 = os.path.join(OUTPUT_DIR_BASE, 'spea2')
    os.makedirs(output_dir_spea2, exist_ok=True)
    spea2_final_archive_pop, spea2_final_archive_fitness = spea2(
        pop_size=POP_SIZE_MAIN, archive_size=ARCHIVE_SIZE_MAIN, generations=GENERATIONS_MAIN,
        activities_dict=activities_dict, groups_dict=groups_dict, lecturers_dict=lecturers_dict,
        spaces_dict=spaces_dict, slots=slots
    )
    spea2_indices = list(range(len(spea2_final_archive_fitness)))
    spea2_fronts = non_dominated_sort(
        spea2_indices, spea2_final_archive_fitness)
    spea2_pareto_indices = spea2_fronts[0] if spea2_fronts else []
    spea2_pareto_fitness = [spea2_final_archive_fitness[i]
                            for i in spea2_pareto_indices]

    # MOEA/D
    output_dir_moead = os.path.join(OUTPUT_DIR_BASE, 'moead')
    os.makedirs(output_dir_moead, exist_ok=True)
    # Pass the new max_replacements argument
    moead_pareto_pop, moead_pareto_fitness = moead(
        pop_size=POP_SIZE_MAIN,
        generations=GENERATIONS_MAIN,
        activities_dict=activities_dict,
        groups_dict=groups_dict,
        lecturers_dict=lecturers_dict,
        spaces_dict=spaces_dict,
        slots=slots,
        n_neighbors=N_NEIGHBORS_MAIN,
        max_replacements=MAX_REPLACEMENTS_MAIN  # Pass the parameter
    )

    # STEP 2: Add detailed printout for MOEA/D results
    print("\n--- MOEA/D Pareto Fitness Values ---")
    if moead_pareto_fitness:
        print(f"Found {len(moead_pareto_fitness)} Pareto solutions:")
        # Limit printing if the list is very long, e.g., first 20
        limit = min(20, len(moead_pareto_fitness))
        print(f"(Showing first {limit}):")
        for i, fit in enumerate(moead_pareto_fitness[:limit]):
            # Format the tuple for better readability
            formatted_fit = tuple(f"{x:.4f}" if isinstance(
                x, float) else x for x in fit)
            print(f"  Solution {i}: {formatted_fit}")
        if len(moead_pareto_fitness) > limit:
            print(
                f"  ... ({len(moead_pareto_fitness) - limit} more not shown)")
    else:
        print("MOEA/D returned an empty Pareto front.")
    # -----------------------------------------

    # --- Generate Combined Plot ---
    # ... (Plotting code remains the same) ...
    print("\n--- Generating Combined EA Pareto Front Plot ---")
    plot_combined_pareto(
        fitness_sets=[nsga2_pareto_fitness,
                      spea2_pareto_fitness, moead_pareto_fitness],
        names=["NSGA-II", "SPEA2", "MOEA/D"],
        colors=['blue', 'green', 'red'],
        markers=['o', 's', '^'],  # Circle, Square, Triangle
        output_dir=OUTPUT_DIR_BASE,  # Save in the base results dir
        obj_x_index=3, obj_y_index=4,  # Unassigned vs 1-SoftScore
        obj_x_label=objective_labels[3], obj_y_label=objective_labels[4]
    )
    plot_combined_pareto(
        fitness_sets=[nsga2_pareto_fitness,
                      spea2_pareto_fitness, moead_pareto_fitness],
        names=["NSGA-II", "SPEA2", "MOEA/D"],
        colors=['blue', 'green', 'red'],
        markers=['o', 's', '^'],
        output_dir=OUTPUT_DIR_BASE,
        obj_x_index=0, obj_y_index=3,  # Prof Conflicts vs Unassigned
        obj_x_label=objective_labels[0], obj_y_label=objective_labels[3]
    )

    # --- Final Comparison ---
    # ... (Final comparison print statements remain the same) ...
    print("\n--- Final Pareto Front Sizes ---")
    print(f"NSGA-II: {len(nsga2_pareto_fitness)}")
    print(f"SPEA2:   {len(spea2_pareto_fitness)}")
    print(f"MOEA/D:  {len(moead_pareto_fitness)}")

    print(f"\nCombined plot saved to '{OUTPUT_DIR_BASE}' directory.")
