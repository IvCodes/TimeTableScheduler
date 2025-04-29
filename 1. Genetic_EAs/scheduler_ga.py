import json
import random
import numpy as np
import matplotlib.pyplot as plt
import copy  # For deep copying objects
from pprint import pprint

# === Data Classes ===


class Space:
    def __init__(self, code=None, capacity=None, id=None, size=None, **kwargs):
        self.code = code or id
        self.size = size or capacity

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
                 teacher_id=None, lecturer=None,
                 group_ids=None, group=None,
                 duration=1, **kwargs):
        self.id = id or code
        self.subject = subject or name
        self.teacher_id = teacher_id or (lecturer.id if lecturer else None)
        self.group_ids = group_ids or ([group.id] if group else [])
        self.duration = duration
        self.lecturer = lecturer
        self.group = group

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
    for space in data.get('spaces', []):
        spaces_dict[space.get('code', space.get('id'))] = Space(**space)

    # Load student groups
    groups_dict = {}
    for group in data.get('years', data.get('groups', [])):
        groups_dict[group['id']] = Group(**group)

    # Load lecturers/teachers
    lecturers_dict = {}
    for user in data.get('users', []):
        if user.get('role') == 'lecturer':
            lecturers_dict[user['id']] = Lecturer(**user)

    # Load activities/courses
    activities_dict = {}
    for activity in data.get('activities', []):
        # Handle different data structures
        if 'teacher_ids' in activity:
            activity['teacher_id'] = activity['teacher_ids'][0]
        if 'subgroup_ids' in activity:
            activity['group_ids'] = activity['subgroup_ids']

        activities_dict[activity.get(
            'code', activity.get('id'))] = Activity(**activity)

    # Generate time slots
    slots = []
    for day in ["MON", "TUE", "WED", "THU", "FRI"]:
        for i in range(1, 9):  # 8 slots per day
            slots.append(f"{day}{i}")

    return spaces_dict, groups_dict, activities_dict, lecturers_dict, slots

# === Hard Constraint Evaluation ===


def evaluate_hard_constraints(timetable, activities_dict, groups_dict, spaces_dict):
    """Evaluate hard constraints (must be satisfied)."""
    violations = {
        'vacant': 0,          # Number of vacant slots
        'prof_conflicts': 0,  # Number of lecturer double-bookings
        'room_size_conflicts': 0,  # Number of room capacity violations
        'sub_group_conflicts': 0,  # Number of student group conflicts
        'unassigned': 0       # Number of unassigned activities
    }

    activities_set = set()  # Track assigned activities

    for slot in timetable:
        prof_set = set()      # Track lecturers in this slot
        sub_group_set = set()  # Track student groups in this slot

        for room in timetable[slot]:
            activity = timetable[slot][room]

            # Check if slot is vacant or invalid activity
            if not isinstance(activity, Activity):
                violations['vacant'] += 1
                continue

            # Track assigned activity
            activities_set.add(activity.id)

            # Check for lecturer conflicts
            if activity.teacher_id in prof_set:
                violations['prof_conflicts'] += 1
            prof_set.add(activity.teacher_id)

            # Check for student group conflicts
            for group_id in activity.group_ids:
                if group_id in sub_group_set:
                    violations['sub_group_conflicts'] += 1
                sub_group_set.add(group_id)

            # Check room capacity
            group_size = sum(
                groups_dict[g].size for g in activity.group_ids if g in groups_dict)
            if group_size > spaces_dict[room].size:
                violations['room_size_conflicts'] += 1

    # Calculate unassigned activities
    violations['unassigned'] = len(activities_dict) - len(activities_set)

    # Calculate total violations
    violations['total_hard'] = (
        violations['prof_conflicts'] +
        violations['sub_group_conflicts'] +
        violations['room_size_conflicts'] +
        violations['unassigned']
    )

    return violations

# === Soft Constraint Evaluation ===


def evaluate_soft_constraints(schedule, groups_dict, lecturers_dict, slots):
    """
    Evaluates the soft constraints of a given schedule, handling missing (None) activities.
    This function measures:
    - Student group metrics: fatigue, idle time, lecture spread.
    - Lecturer metrics: fatigue, idle time, lecture spread, and workload balance.
    
    Returns:
    - final_score (float): Computed soft constraint score
    """
    # Initialize student group metrics
    group_fatigue = {g: 0 for g in groups_dict.keys()}
    group_idle_time = {g: 0 for g in groups_dict.keys()}
    group_lecture_spread = {g: 0 for g in groups_dict.keys()}

    # Initialize lecturer metrics
    lecturer_fatigue = {l: 0 for l in lecturers_dict.keys()}
    lecturer_idle_time = {l: 0 for l in lecturers_dict.keys()}
    lecturer_lecture_spread = {l: 0 for l in lecturers_dict.keys()}
    lecturer_workload = {l: 0 for l in lecturers_dict.keys()}

    # Track the lecture slots assigned to each group and lecturer
    group_lecture_slots = {g: [] for g in groups_dict.keys()}
    lecturer_lecture_slots = {l: [] for l in lecturers_dict.keys()}

    # Process the schedule
    for slot, rooms in schedule.items():
        for room, activity in rooms.items():
            if activity is None:
                continue

            # Process student groups
            if not isinstance(activity, Activity):
                continue

            if not isinstance(activity.group_ids, list):
                continue

            for group_id in activity.group_ids:
                if group_id in groups_dict:
                    group_fatigue[group_id] += 1
                    group_lecture_spread[group_id] += 2
                    group_lecture_slots[group_id].append(slot)

            # Process lecturers
            lecturer_id = activity.teacher_id
            if lecturer_id in lecturers_dict:
                lecturer_fatigue[lecturer_id] += 1
                lecturer_lecture_spread[lecturer_id] += 2
                lecturer_workload[lecturer_id] += activity.duration
                lecturer_lecture_slots[lecturer_id].append(slot)

    # Calculate idle time
    for group_id, lectures in group_lecture_slots.items():
        if lectures:
            lecture_indices = sorted([slots.index(s) for s in lectures])
            idle_time = sum(
                (lecture_indices[i+1] - lecture_indices[i] - 1) for i in range(len(lecture_indices)-1)
            )
            group_idle_time[group_id] = idle_time / (len(slots) - 1)

    for lecturer_id, lectures in lecturer_lecture_slots.items():
        if lectures:
            lecture_indices = sorted([slots.index(s) for s in lectures])
            idle_time = sum(
                (lecture_indices[i+1] - lecture_indices[i] - 1) for i in range(len(lecture_indices)-1)
            )
            lecturer_idle_time[lecturer_id] = idle_time / (len(slots) - 1)

    # Normalize metrics
    def normalize(dictionary):
        max_val = max(dictionary.values(), default=1)
        return {k: v/max_val if max_val else 0 for k, v in dictionary.items()}

    # Normalize all metrics
    group_fatigue = normalize(group_fatigue)
    group_idle_time = normalize(group_idle_time)
    group_lecture_spread = normalize(group_lecture_spread)
    lecturer_fatigue = normalize(lecturer_fatigue)
    lecturer_idle_time = normalize(lecturer_idle_time)
    lecturer_lecture_spread = normalize(lecturer_lecture_spread)

    # Calculate workload balance
    workload_values = np.array(list(lecturer_workload.values()))
    lecturer_workload_balance = 1  # Default balance
    if len(workload_values) > 1 and np.mean(workload_values) != 0:
        lecturer_workload_balance = max(
            0, 1 - (np.var(workload_values) / np.mean(workload_values)))

    # Calculate final metrics
    student_fatigue_score = np.mean(list(group_fatigue.values()))
    student_idle_time_score = np.mean(list(group_idle_time.values()))
    student_lecture_spread_score = np.mean(list(group_lecture_spread.values()))
    lecturer_fatigue_score = np.mean(list(lecturer_fatigue.values()))
    lecturer_idle_time_score = np.mean(list(lecturer_idle_time.values()))
    lecturer_lecture_spread_score = np.mean(
        list(lecturer_lecture_spread.values()))

    # Calculate weighted final score
    final_score = (
        student_fatigue_score * 0.2 +
        (1 - student_idle_time_score) * 0.2 +
        (1 - student_lecture_spread_score) * 0.2 +
        (1 - lecturer_fatigue_score) * 0.1 +
        (1 - lecturer_idle_time_score) * 0.1 +
        (1 - lecturer_lecture_spread_score) * 0.1 +
        lecturer_workload_balance * 0.1
    )

    return final_score

# === Evaluator Function ===


def evaluator(timetable, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots):
    """Evaluate a timetable and return (hard_violations, soft_score)."""
    hard = evaluate_hard_constraints(
        timetable, activities_dict, groups_dict, spaces_dict)
    soft = evaluate_soft_constraints(
        timetable, groups_dict, lecturers_dict, slots)
    return (hard['total_hard'], soft)

# === Generate Initial Population ===


def generate_initial_population(pop_size, slots, activities_dict, spaces_dict):
    """Generate an initial population of timetables."""
    population = []
    all_activities = list(activities_dict.values())

    for _ in range(pop_size):
        timetable = {}
        for slot in slots:
            timetable[slot] = {}
            for room in spaces_dict:
                timetable[slot][room] = random.choice(all_activities)
        population.append(timetable)

    return population

# === Crossover ===


def crossover(parent1, parent2):
    """Perform crossover by swapping time slots between two parents."""
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    slots = list(parent1.keys())
    split = random.randint(0, len(slots) - 1)
    for i in range(split, len(slots)):
        child1[slots[i]], child2[slots[i]
                                 ] = parent2[slots[i]], parent1[slots[i]]
    return child1, child2

# === Mutation ===


def mutate(individual):
    """Perform mutation by randomly swapping activities in the timetable."""
    slots = list(individual.keys())
    slot1, slot2 = random.sample(slots, 2)
    room1, room2 = random.choice(
        list(individual[slot1])), random.choice(list(individual[slot2]))
    individual[slot1][room1], individual[slot2][room2] = individual[slot2][room2], individual[slot1][room1]
    return individual

# === Pareto Front Helper ===


def pareto_frontier(points):
    """Compute the Pareto frontier for a set of points."""
    pts = np.array(points)
    is_pf = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        for j, q in enumerate(pts):
            if j != i and (q <= p).all() and (q < p).any():
                is_pf[i] = False
                break
    return is_pf


# === SPEA2 Helpers ===
def calculate_strength(population, fitness):
    """Calculate strength values for SPEA2."""
    n = len(population)
    strength = np.zeros(n)
    
    # For each individual, count how many others it dominates
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # If i dominates j
            if all(fitness[i][k] <= fitness[j][k] for k in range(len(fitness[i]))) and \
               any(fitness[i][k] < fitness[j][k] for k in range(len(fitness[i]))):
                strength[i] += 1
                
    return strength


def calculate_raw_fitness(population, fitness, strength):
    """Calculate raw fitness values for SPEA2."""
    n = len(population)
    raw_fitness = np.zeros(n)
    
    # Sum up strengths of dominating solutions
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # If j dominates i
            if all(fitness[j][k] <= fitness[i][k] for k in range(len(fitness[i]))) and \
               any(fitness[j][k] < fitness[i][k] for k in range(len(fitness[i]))):
                raw_fitness[i] += strength[j]
                
    return raw_fitness


def calculate_density(population, fitness):
    """Calculate density values for SPEA2 using k-nearest neighbor method."""
    n = len(population)
    k = int(np.sqrt(n))  # k is typically sqrt(n)
    density = np.zeros(n)
    
    # Calculate distances between all pairs
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # Euclidean distance in objective space
            dist = np.sqrt(sum((fitness[i][k] - fitness[j][k])**2 for k in range(len(fitness[i]))))
            distances[i, j] = distances[j, i] = dist
    
    # For each individual, find the kth nearest neighbor
    for i in range(n):
        sorted_distances = np.sort(distances[i])
        # k+1 because the closest is itself (distance 0)
        kth_distance = sorted_distances[min(k, n-1)]
        # Density is inverse of distance to avoid division by zero
        density[i] = 1.0 / (kth_distance + 2.0)
        
    return density


def truncate_archive(archive, archive_fitness, max_size):
    """Truncate archive to maintain maximum size in SPEA2."""
    if len(archive) <= max_size:
        return archive, archive_fitness
    
    # Calculate distances between all pairs
    n = len(archive)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # Euclidean distance in objective space
            dist = np.sqrt(sum((archive_fitness[i][k] - archive_fitness[j][k])**2 for k in range(len(archive_fitness[i]))))
            distances[i, j] = distances[j, i] = dist
    
    # Remove solutions one by one
    while len(archive) > max_size:
        # Find solution with minimum distance to nearest neighbor
        min_dist = float('inf')
        to_remove = -1
        
        for i in range(len(archive)):
            # Find nearest neighbor (excluding self)
            nn_dist = min(distances[i, j] for j in range(len(archive)) if i != j)
            
            if nn_dist < min_dist:
                min_dist = nn_dist
                to_remove = i
        
        # Remove the solution
        archive.pop(to_remove)
        archive_fitness.pop(to_remove)
        
        # Update distances matrix
        distances = np.delete(distances, to_remove, axis=0)
        distances = np.delete(distances, to_remove, axis=1)
    
    return archive, archive_fitness


# === MOEA/D Helpers ===
def generate_weight_vectors(n_weights, n_objectives):
    """Generate well-distributed weight vectors for MOEA/D decomposition.
    Uses a systematic approach to generate more uniform coverage of the weight space.
    """
    if n_objectives == 2:
        # For bi-objective problems, use non-uniform distribution to better explore the Pareto front
        # This helps concentrate more weight vectors in potentially interesting regions
        weights = []
        # Use non-linear distribution to better explore the Pareto front
        for i in range(n_weights):
            # Power distribution gives more points near the extremes
            # which is often where interesting trade-offs occur
            alpha = 0.5  # Parameter controlling the distribution (0.5 gives more emphasis to extremes)
            t = i / (n_weights - 1) if n_weights > 1 else 0.5
            w1 = t ** alpha
            w2 = 1.0 - w1
            weights.append([w1, w2])
            
            # Also add the inverse to ensure symmetry
            if 0 < i < n_weights - 1:  # Skip the first and last to avoid duplicates
                weights.append([w2, w1])
                
        # Ensure we don't exceed the target population size
        if len(weights) > n_weights:
            # Randomly sample to get exactly n_weights
            indices = np.random.choice(len(weights), n_weights, replace=False)
            weights = [weights[i] for i in indices]
        
        return np.array(weights)
    else:
        # For higher dimensions, use simplex lattice design with controlled density
        # This is more systematic than random Dirichlet sampling
        H = 1
        while choose(H + n_objectives - 1, n_objectives - 1) <= n_weights:
            H += 1
        H -= 1  # Back up to the largest valid H
        
        # Generate the weight vectors using the simplex-lattice method
        weights = []
        def generate_weights_recursive(weight, left, depth):
            if depth == n_objectives - 1:
                weight.append(left / H)
                weights.append(weight.copy())
                weight.pop()
                return
            for i in range(left + 1):
                weight.append(i / H)
                generate_weights_recursive(weight, left - i, depth + 1)
                weight.pop()
        
        generate_weights_recursive([], H, 0)
        
        # If we generated too many, randomly select n_weights of them
        if len(weights) > n_weights:
            indices = np.random.choice(len(weights), n_weights, replace=False)
            weights = [weights[i] for i in indices]
        
        return np.array(weights)


def choose(n, k):
    """Calculate the binomial coefficient (n choose k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


def calculate_tchebycheff(weight, fitness, ideal_point):
    """Calculate the Tchebycheff scalarization value (to be minimized)."""
    # Add small epsilon to avoid division by zero with zero weights
    epsilon = 1e-6
    # Calculate the weighted max deviation from ideal point
    return np.max([(fitness[i] - ideal_point[i]) / (weight[i] + epsilon) for i in range(len(weight))])

# === Non-dominated Sorting ===


def non_dominated_sort(population, fitness_values):
    """Sort population into fronts based on dominance."""
    fronts = [[]]
    S = [[] for _ in range(len(population))]
    n = [0] * len(population)
    rank = [0] * len(population)

    for p in range(len(population)):
        for q in range(len(population)):
            if p == q:
                continue

            # p dominates q
            if all(fitness_values[p][k] <= fitness_values[q][k] for k in range(len(fitness_values[p]))) and \
               any(fitness_values[p][k] < fitness_values[q][k] for k in range(len(fitness_values[p]))):
                S[p].append(q)
            # q dominates p
            elif all(fitness_values[q][k] <= fitness_values[p][k] for k in range(len(fitness_values[p]))) and \
                    any(fitness_values[q][k] < fitness_values[p][k] for k in range(len(fitness_values[p]))):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]  # Remove empty last front

# === Crowding Distance ===


def crowding_distance(fitness_values, front):
    """Calculate crowding distance for solutions in a front."""
    distances = {i: 0 for i in front}
    num_objectives = len(fitness_values[0])

    for m in range(num_objectives):
        # Sort front by this objective
        sorted_front = sorted(front, key=lambda i: fitness_values[i][m])

        # Assign infinite distance to boundary points
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')

        # Skip if not enough points or all values are the same
        if len(sorted_front) <= 2:
            continue

        min_val = fitness_values[sorted_front[0]][m]
        max_val = fitness_values[sorted_front[-1]][m]
        if max_val == min_val:
            continue

        # Calculate distance for intermediate points
        for i in range(1, len(sorted_front) - 1):
            distances[sorted_front[i]] += (fitness_values[sorted_front[i+1]][m] -
                                           fitness_values[sorted_front[i-1]][m]) / (max_val - min_val)

    return distances

# === Selection ===


def selection(population, fitness, pop_size, fronts, distances):
    """Select parents for next generation."""
    selected = []

    # First prioritize by front rank
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            # Add all of this front
            for i in front:
                selected.append(population[i])
        else:
            # Sort this front by crowding distance
            sorted_front = sorted(
                front, key=lambda i: distances[i], reverse=True)
            needed = pop_size - len(selected)
            for i in sorted_front[:needed]:
                selected.append(population[i])
            break

    return selected

# === NSGA-II Main Loop ===


def nsga2(pop_size, generations, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots):
    """Main NSGA-II algorithm."""
    # Constants
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1

    # 1. Initialize population
    population = generate_initial_population(
        pop_size, slots, activities_dict, spaces_dict)
    fitness = [evaluator(ind, activities_dict, groups_dict,
                         lecturers_dict, spaces_dict, slots) for ind in population]

    for gen in range(generations):
        if gen % 10 == 0:
            print(f"NSGA-II Generation {gen + 1}/{generations}")

        # 2. Non-Dominated Sort & Crowding Distance
        fronts = non_dominated_sort(population, fitness)
        distances = {}
        for front in fronts:
            distances.update(crowding_distance(fitness, front))

        # 3. Selection
        parents = selection(population, fitness, pop_size, fronts, distances)

        # 4. Crossover & Mutation
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(parents, 2)
            if random.random() < CROSSOVER_RATE:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            if random.random() < MUTATION_RATE:
                mutate(c1)
            if random.random() < MUTATION_RATE:
                mutate(c2)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # 5. Evaluate offspring
        offspring_fitness = [evaluator(
            ind, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots) for ind in offspring]

        # 6. Combine & sort populations
        combined_pop = population + offspring
        combined_fitness = fitness + offspring_fitness

        # 7. Select next generation
        new_population = []
        new_fitness = []

        for front in non_dominated_sort(combined_pop, combined_fitness):
            if len(new_population) + len(front) <= pop_size:
                for i in front:
                    new_population.append(combined_pop[i])
                    new_fitness.append(combined_fitness[i])
            else:
                front_distances = crowding_distance(combined_fitness, front)
                sorted_front = sorted(
                    front, key=lambda i: front_distances[i], reverse=True)

                needed = pop_size - len(new_population)
                for i in sorted_front[:needed]:
                    new_population.append(combined_pop[i])
                    new_fitness.append(combined_fitness[i])
                break

        population = new_population
        fitness = new_fitness

    return population, fitness

# === Plotting Function ===


def plot_pareto(fitness, name):
    """Plot the Pareto front from the fitness values."""
    fit = np.array(fitness)
    is_pf = pareto_frontier(fit[:, :2])

    plt.figure(figsize=(8, 6))
    plt.scatter(fit[:, 0], fit[:, 1], alpha=0.5, label='All')
    plt.scatter(fit[is_pf, 0], fit[is_pf, 1], c='red', label='Pareto Front')
    plt.xlabel('Hard Constraint Violations')
    plt.ylabel('Soft Constraint Score')
    plt.title(f'{name} Pareto Front')
    plt.legend()
    # Create safe filename by replacing problematic characters
    safe_filename = name.lower().replace('/', '_').replace('\\', '_').replace(' ', '_')
    plt.savefig(f'{safe_filename}_pareto.png')
    plt.show()


# === SPEA2 Implementation ===
def spea2(pop_size, archive_size, generations, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots):
    """SPEA2 (Strength Pareto Evolutionary Algorithm 2) implementation."""
    # Constants
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1
    
    # Initialize population and empty archive
    population = generate_initial_population(pop_size, slots, activities_dict, spaces_dict)
    fitness = [evaluator(ind, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots) for ind in population]
    archive = []
    archive_fitness = []
    
    for gen in range(generations):
        if gen % 10 == 0:
            print(f"SPEA2 Generation {gen + 1}/{generations}")
        
        # Combine population and archive
        combined_pop = population + archive
        combined_fitness = fitness + archive_fitness
        n_combined = len(combined_pop)
        
        # Calculate strength and raw fitness
        strength = calculate_strength(combined_pop, combined_fitness)
        raw_fitness = calculate_raw_fitness(combined_pop, combined_fitness, strength)
        
        # Calculate density (for tie-breaking)
        density = calculate_density(combined_pop, combined_fitness)
        
        # Calculate final fitness: smaller is better (raw + density)
        final_fitness = raw_fitness + density
        
        # Select new archive (non-dominated solutions)
        new_archive = []
        new_archive_fitness = []
        
        for i in range(n_combined):
            if raw_fitness[i] == 0:  # Non-dominated solutions
                new_archive.append(combined_pop[i])
                new_archive_fitness.append(combined_fitness[i])
        
        # If archive underflows, fill with best dominated solutions
        if len(new_archive) < archive_size:
            # Sort by final fitness (smaller is better)
            remaining = [(i, final_fitness[i]) for i in range(n_combined) if raw_fitness[i] > 0]
            remaining.sort(key=lambda x: x[1])
            
            # Add solutions until archive is filled
            i = 0
            while len(new_archive) < archive_size and i < len(remaining):
                idx = remaining[i][0]
                new_archive.append(combined_pop[idx])
                new_archive_fitness.append(combined_fitness[idx])
                i += 1
        
        # If archive overflows, truncate using density
        if len(new_archive) > archive_size:
            new_archive, new_archive_fitness = truncate_archive(
                new_archive, new_archive_fitness, archive_size)
        
        # Update archive
        archive = new_archive
        archive_fitness = new_archive_fitness
        
        # Binary tournament selection from archive
        mating_pool = []
        for _ in range(pop_size):
            i, j = random.sample(range(len(archive)), 2)
            # Choose the better one (smaller fitness value is better)
            selected = i if final_fitness[i] < final_fitness[j] else j
            mating_pool.append(archive[selected])
        
        # Create new population using crossover and mutation
        new_population = []
        while len(new_population) < pop_size:
            p1, p2 = random.sample(mating_pool, 2)
            
            if random.random() < CROSSOVER_RATE:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                
            if random.random() < MUTATION_RATE:
                mutate(c1)
            if random.random() < MUTATION_RATE:
                mutate(c2)
                
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)
        
        # Update population and fitness
        population = new_population
        fitness = [evaluator(ind, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots) 
                  for ind in population]
    
    # Combine final population and archive
    combined_pop = population + archive
    combined_fitness = fitness + archive_fitness
    
    return combined_pop, combined_fitness


# === MOEA/D Implementation ===
def moead(pop_size, generations, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots, n_neighbors=15):
    """MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) implementation.
    Enhanced with improved weight generation, adaptive neighborhood, and more effective Tchebycheff approach.
    """
    # Algorithm parameters
    CROSSOVER_RATE = 0.9  # Increased to promote more exploration
    MUTATION_RATE = 0.15  # Increased slightly to escape local optima
    N_OBJECTIVES = 2  # Hard and soft constraint objectives
    MAX_REPLACEMENTS = 2  # Limit replacements per generation to maintain diversity
    
    # Ensure population size is reasonable
    pop_size = max(pop_size, 30)  # Need enough for good weight vector distribution
    
    # Initialize population with some diversity
    population = []
    for _ in range(pop_size):
        # Create an individual with some randomization to ensure diversity
        timetable = {}
        for slot in slots:
            timetable[slot] = {}
            for room in spaces_dict:
                # Sometimes leave slots empty to help with hard constraint violations
                if random.random() < 0.05:  # 5% chance of empty slot
                    timetable[slot][room] = None
                else:
                    timetable[slot][room] = random.choice(list(activities_dict.values()))
        population.append(timetable)
    
    # Evaluate initial population
    fitness_values = [evaluator(ind, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots) 
                     for ind in population]
    
    # Generate well-distributed weight vectors
    weight_vectors = generate_weight_vectors(pop_size, N_OBJECTIVES)
    
    # Initialize ideal point (to be minimized)
    ideal_point = np.array([min(f[i] for f in fitness_values) for i in range(N_OBJECTIVES)])
    
    # Initialize nadir point (worst values, to be used for normalization)
    nadir_point = np.array([max(f[i] for f in fitness_values) for i in range(N_OBJECTIVES)])
    
    # Compute neighborhood of each weight vector
    neighborhoods = []
    for i in range(pop_size):
        # Use Euclidean distance between weight vectors
        distances = [np.sqrt(np.sum((weight_vectors[i] - weight_vectors[j])**2)) for j in range(pop_size)]
        # Get indices of n_neighbors closest weights
        sorted_idx = np.argsort(distances)
        neighborhoods.append(sorted_idx[:n_neighbors].tolist())  # Convert to list
    
    # External archive for storing non-dominated solutions
    archive = []
    archive_fitness = []
    MAX_ARCHIVE_SIZE = pop_size * 2
    
    # Main algorithm loop
    for gen in range(generations):
        if gen % 10 == 0:
            print(f"MOEA/D Generation {gen + 1}/{generations}")
            
            # Periodically update nadir point for better scaling
            nadir_point = np.array([max(f[i] for f in fitness_values) for i in range(N_OBJECTIVES)])
        
        # Keep track of replacements to limit them
        replacement_count = np.zeros(pop_size, dtype=int)
        
        # For each subproblem
        for i in range(pop_size):
            # Dynamically determine neighborhood size based on generation
            # Start with smaller neighborhoods and gradually expand
            dynamic_n_neighbors = max(2, min(n_neighbors, int(n_neighbors * (1 + gen/generations))))
            
            # Occasionally use the entire population for global search (10% probability)
            use_global = random.random() < 0.1
            
            if use_global or len(neighborhoods[i]) < 2:
                # Global selection from entire population
                selection_pool = list(range(pop_size))
            else:
                # Local selection from neighborhood
                selection_pool = neighborhoods[i][:dynamic_n_neighbors]
            
            # Select parents
            k, l = random.sample(selection_pool, 2)
            
            # Perform differential evolution style recombination
            # with higher probability for exploitation of good solutions
            if random.random() < CROSSOVER_RATE:
                if random.random() < 0.7:  # 70% standard crossover
                    c1, c2 = crossover(population[k], population[l])
                    child = c1  # Take first child
                else:  # 30% more exploratory three-parent crossover
                    # Find a third parent
                    m = random.choice([p for p in selection_pool if p != k and p != l])
                    # Create child by combining all three parents
                    child = copy.deepcopy(population[k])
                    slots_list = list(child.keys())
                    # Take 1/3 from each parent
                    for s in range(len(slots_list)):
                        if s % 3 == 0:
                            child[slots_list[s]] = copy.deepcopy(population[k][slots_list[s]])
                        elif s % 3 == 1:
                            child[slots_list[s]] = copy.deepcopy(population[l][slots_list[s]])
                        else:
                            child[slots_list[s]] = copy.deepcopy(population[m][slots_list[s]])
            else:
                # Slight preference for better parent
                if sum(fitness_values[k]) < sum(fitness_values[l]):
                    child = copy.deepcopy(population[k])
                else:
                    child = copy.deepcopy(population[l])
            
            # Apply adaptive mutation
            if random.random() < MUTATION_RATE:
                # Higher mutation rate in early generations
                mutation_intensity = 1 + (1 - gen/generations) * 2  # Starts at 3, ends at 1
                # Apply multiple mutations based on intensity
                for _ in range(int(mutation_intensity)):
                    mutate(child)
            
            # Evaluate offspring
            child_fitness = evaluator(child, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots)
            
            # Update ideal point
            for j in range(N_OBJECTIVES):
                ideal_point[j] = min(ideal_point[j], child_fitness[j])
            
            # Normalize update neighborhood
            # We'll use a mix of random and closest neighbors for better exploration
            update_indices = set(neighborhoods[i][:int(n_neighbors * 0.7)])  # 70% closest neighbors
            # Add 30% random neighbors from the rest of the population
            remaining = [j for j in range(pop_size) if j not in update_indices]
            if remaining and n_neighbors > 5:  # Only if enough neighbors and population
                random_count = min(int(n_neighbors * 0.3), len(remaining))
                update_indices.update(random.sample(remaining, random_count))
            
            # Update neighborhood solutions using improved Tchebycheff approach with normalization
            updates_made = 0
            for j in update_indices:
                # Skip if this solution has been replaced too many times
                if replacement_count[j] >= MAX_REPLACEMENTS:
                    continue
                    
                # Normalize objectives for more balanced comparison
                range_obj = [max(1e-6, nadir_point[k] - ideal_point[k]) for k in range(N_OBJECTIVES)]
                
                # Calculate normalized Tchebycheff values
                parent_norm = [(fitness_values[j][k] - ideal_point[k]) / range_obj[k] for k in range(N_OBJECTIVES)]
                child_norm = [(child_fitness[k] - ideal_point[k]) / range_obj[k] for k in range(N_OBJECTIVES)]
                
                wt_parent = max(weight_vectors[j][k] * parent_norm[k] for k in range(N_OBJECTIVES))
                wt_child = max(weight_vectors[j][k] * child_norm[k] for k in range(N_OBJECTIVES))
                
                # If offspring is better, replace solution
                if wt_child < wt_parent:  # Strictly better
                    population[j] = copy.deepcopy(child)
                    fitness_values[j] = child_fitness
                    replacement_count[j] += 1
                    updates_made += 1
                    if updates_made >= 3:  # Limit updates per offspring
                        break
            
            # Update external archive with non-dominated solutions
            add_to_archive = True
            i = 0
            while i < len(archive_fitness):
                # Check if child is dominated by archive solution
                if (all(archive_fitness[i][k] <= child_fitness[k] for k in range(N_OBJECTIVES)) and
                    any(archive_fitness[i][k] < child_fitness[k] for k in range(N_OBJECTIVES))):
                    add_to_archive = False
                    break
                # Check if child dominates archive solution
                elif (all(child_fitness[k] <= archive_fitness[i][k] for k in range(N_OBJECTIVES)) and
                      any(child_fitness[k] < archive_fitness[i][k] for k in range(N_OBJECTIVES))):
                    # Remove dominated solution
                    archive.pop(i)
                    archive_fitness.pop(i)
                else:
                    i += 1
            
            # Add non-dominated child to archive
            if add_to_archive:
                archive.append(copy.deepcopy(child))
                archive_fitness.append(child_fitness)
                
                # Truncate archive if too large
                if len(archive) > MAX_ARCHIVE_SIZE:
                    # Remove crowded solutions to maintain diversity
                    # Calculate distances between solutions in objective space
                    distances = np.zeros((len(archive), len(archive)))
                    for i in range(len(archive)):
                        for j in range(i+1, len(archive)):
                            d = np.sqrt(sum((archive_fitness[i][k] - archive_fitness[j][k])**2 
                                           for k in range(N_OBJECTIVES)))
                            distances[i, j] = distances[j, i] = d
                    
                    # For each solution, find distance to nearest neighbor
                    nn_distances = [min(distances[i, j] for j in range(len(archive)) if i != j) 
                                  for i in range(len(archive))]
                    
                    # Remove solution with smallest nearest-neighbor distance (most crowded)
                    idx_to_remove = np.argmin(nn_distances)
                    archive.pop(idx_to_remove)
                    archive_fitness.pop(idx_to_remove)
    
    # Return both population and archive for a more diverse set of solutions
    combined_pop = population + archive
    combined_fitness = fitness_values + archive_fitness
    
    return combined_pop, combined_fitness


# === Main Execution Block ===
if __name__ == "__main__":
    # Configuration
    DATA_FILE = 'sliit_computing_dataset.json'
    POP_SIZE = 50
    GENERATIONS = 25  # Reduced generations for each algorithm to save time
    ARCHIVE_SIZE = 50  # For SPEA2
    N_NEIGHBORS = 15   # For MOEA/D

    # Load data
    try:
        print("Loading data...")
        spaces_dict, groups_dict, activities_dict, lecturers_dict, slots = load_data(
            DATA_FILE)
        print(f"Loaded {len(activities_dict)} activities, {len(spaces_dict)} spaces, {len(groups_dict)} groups, {len(lecturers_dict)} lecturers")
        print(f"Defined {len(slots)} time slots")
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # Run NSGA-II
    print("\n--- Running NSGA-II ---")
    nsga2_pop, nsga2_fitness = nsga2(POP_SIZE, GENERATIONS,
                                     activities_dict, groups_dict,
                                     lecturers_dict, spaces_dict, slots)
    print("NSGA-II Complete!")
    print("\n--- Generating NSGA-II Pareto Front Plot ---")
    plot_pareto(nsga2_fitness, "NSGA-II")
    
    # Run SPEA2
    print("\n--- Running SPEA2 ---")
    spea2_pop, spea2_fitness = spea2(POP_SIZE, ARCHIVE_SIZE, GENERATIONS,
                                    activities_dict, groups_dict,
                                    lecturers_dict, spaces_dict, slots)
    print("SPEA2 Complete!")
    print("\n--- Generating SPEA2 Pareto Front Plot ---")
    plot_pareto(spea2_fitness, "SPEA2")
    
    # Run MOEA/D
    print("\n--- Running MOEA/D ---")
    moead_pop, moead_fitness = moead(POP_SIZE, GENERATIONS,
                                   activities_dict, groups_dict,
                                   lecturers_dict, spaces_dict, slots, N_NEIGHBORS)
    print("MOEA/D Complete!")
    print("\n--- Generating MOEA/D Pareto Front Plot ---")
    plot_pareto(moead_fitness, "MOEA/D")
    
    # Compare best solutions from each algorithm
    print("\n--- Best Solutions Comparison ---")
    
    nsga2_best = np.argmin([f[0] for f in nsga2_fitness])
    spea2_best = np.argmin([f[0] for f in spea2_fitness])
    moead_best = np.argmin([f[0] for f in moead_fitness])
    
    print(f"NSGA-II best: Hard violations = {nsga2_fitness[nsga2_best][0]}, Soft score = {nsga2_fitness[nsga2_best][1]:.6f}")
    print(f"SPEA2 best:  Hard violations = {spea2_fitness[spea2_best][0]}, Soft score = {spea2_fitness[spea2_best][1]:.6f}")
    print(f"MOEA/D best: Hard violations = {moead_fitness[moead_best][0]}, Soft score = {moead_fitness[moead_best][1]:.6f}")
