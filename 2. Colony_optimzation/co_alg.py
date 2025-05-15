"""Colony Optimization algorithms implementation for timetable scheduling.

This module implements three colony-based metaheuristic algorithms:
1. Ant Colony Optimization (ACO)
2. Bee Colony Optimization (BCO)
3. Particle Swarm Optimization (PSO)

Each algorithm is implemented with multi-objective optimization capabilities
for solving the University Timetable Scheduling Problem.
"""

import os
import copy
import random
import numpy as np
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional

# Fix Python import path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import common infrastructure modules
from common.data_loader import load_data, Activity, Group, Space, Lecturer, is_space_suitable
from common.evaluator import multi_objective_evaluator
from common.metrics import calculate_hypervolume, calculate_igd, calculate_spread
from common.visualization import (
    plot_pareto_front_2d, plot_pareto_front_3d, plot_parallel_coordinates,
    plot_metrics_comparison, plot_constraint_breakdown, plot_resource_utilization
)
from common.resource_tracker import track_computational_resources, ResourceTracker

# For plotting
import matplotlib.pyplot as plt

# Constants for frequently used strings
PROFESSOR_CONFLICTS = "Professor Conflicts"
GROUP_CONFLICTS = "Group Conflicts"
ROOM_CONFLICTS = "Room Conflicts"
UNASSIGNED_ACTIVITIES = "Unassigned Activities"
SOFT_CONSTRAINTS = "Soft Constraints"

#===============================================================================
# === ACO: Ant Colony Optimization Implementation ===
#===============================================================================

class ACOParameters:
    """Parameters for the ACO algorithm."""
    
    def __init__(self, 
                 num_ants: int = 30,
                 num_iterations: int = 20,
                 evaporation_rate: float = 0.5,
                 alpha: float = 1.0,  # Pheromone importance
                 beta: float = 2.0,   # Heuristic importance
                 q_value: float = 100.0,  # Pheromone deposit quantity
                 elitist_weight: float = 2.0):  # Weight for elitist ant
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q_value = q_value
        self.elitist_weight = elitist_weight


class AntColonyOptimization:
    """
    Ant Colony Optimization implementation for university timetable scheduling.
    
    This class implements a multi-objective ACO approach for generating
    optimal timetables with focus on constraint satisfaction and performance.
    """
    
    def __init__(self, params: ACOParameters = None):
        """
        Initialize the ACO algorithm.
        
        Args:
            params: Parameters for the ACO algorithm
        """
        self.params = params or ACOParameters()
        
        # Data containers
        self.activities_dict = {}
        self.groups_dict = {}
        self.spaces_dict = {}
        self.lecturers_dict = {}
        self.slots = []
        
        # Activity and space attributes (cached)
        self.activity_sizes = {}
        self.suitable_spaces = {}
        
        # Pheromone and heuristic information
        self.pheromone = {}  # Nested dict: activity_id -> slot -> space_id -> value
        self.heuristic = {}  # Nested dict: activity_id -> slot -> space_id -> value
        
        # Results tracking
        self.best_solution = None
        self.best_fitness = None
        self.pareto_front = []
        self.archive = []
        self.iteration_history = []
        
        # Resource tracking
        self.resource_tracker = None
    
    def load_data(self, dataset_path: str = None):
        """
        Load data from the dataset file and prepare all necessary data structures.
        
        Args:
            dataset_path: Path to the dataset file
        """
        # Load dataset
        data = load_data(dataset_path)
        
        self.activities_dict = data[0]
        self.groups_dict = data[1]
        self.spaces_dict = data[2]
        self.lecturers_dict = data[3]
        self.slots = data[12]
        
        # Initialize cached data for performance
        self._initialize_cached_data()
        
        # Initialize pheromone and heuristic matrices
        self._initialize_pheromone()
        self._initialize_heuristic()
        
        print(f"ACO: Data loaded with {len(self.activities_dict)} activities, "
              f"{len(self.spaces_dict)} spaces, {len(self.slots)} slots")
    
    def _initialize_cached_data(self):
        """Initialize cached data for faster access during algorithm execution."""
        # Calculate and cache activity sizes
        self.activity_sizes = {}
        for activity_id, activity in self.activities_dict.items():
            self.activity_sizes[activity_id] = sum(
                self.groups_dict[group_id].size 
                for group_id in activity.group_ids 
                if group_id in self.groups_dict
            )
        
        # Calculate and cache suitable spaces for each activity
        self.suitable_spaces = {}
        for activity_id, activity in self.activities_dict.items():
            self.suitable_spaces[activity_id] = {}
            for space_id, space in self.spaces_dict.items():
                is_suitable = is_space_suitable(space, activity, self.groups_dict)
                if is_suitable:
                    # Also check space size vs. activity size
                    activity_size = self.activity_sizes[activity_id]
                    if activity_size <= space.size:
                        self.suitable_spaces[activity_id][space_id] = True
    
    def _initialize_pheromone(self):
        """Initialize pheromone matrix with default values."""
        # Use nested dictionaries for fast access
        self.pheromone = {}
        for activity_id in self.activities_dict:
            self.pheromone[activity_id] = {}
            for slot in self.slots:
                self.pheromone[activity_id][slot] = {}
                for space_id in self.suitable_spaces.get(activity_id, {}):
                    # Only initialize pheromones for suitable spaces
                    self.pheromone[activity_id][slot][space_id] = 1.0
    
    def _initialize_heuristic(self):
        """Initialize heuristic matrix based on problem instance."""
        # Use nested dictionaries for fast access
        self.heuristic = {}
        
        for activity_id in self.activities_dict:
            self.heuristic[activity_id] = {}
            for slot in self.slots:
                self.heuristic[activity_id][slot] = {}
                
                for space_id in self.suitable_spaces.get(activity_id, {}):
                    # Calculate heuristic value based on space fitness
                    activity_size = self.activity_sizes[activity_id]
                    space_size = self.spaces_dict[space_id].size
                    
                    # Room size fitness: 1.0 if perfect fit, lower if too large
                    size_ratio = activity_size / max(1, space_size)  # Avoid division by zero
                    
                    # Penalize if room is too large (efficient use of resources)
                    size_fitness = 1.0 - max(0, 0.5 - size_ratio * 0.5)  # Between 0.5 and 1.0
                    
                    self.heuristic[activity_id][slot][space_id] = size_fitness
    
    def construct_solution(self) -> Dict:
        """
        Construct a single timetable solution using ACO principles.
        
        Returns:
            Dict: A complete timetable solution
        """
        # Initialize empty schedule
        schedule = {slot: {space_id: None for space_id in self.spaces_dict} 
                   for slot in self.slots}
        
        # Track available slots and spaces
        available_slots = {slot: set(self.spaces_dict.keys()) for slot in self.slots}
        
        # Create a list of activities to schedule, prioritizing larger ones
        activities_to_schedule = sorted(
            list(self.activities_dict.keys()),
            key=lambda x: -len(self.suitable_spaces.get(x, {}))  # Activities with fewer options first
        )
        
        # Shuffle slightly to avoid deterministic behavior
        if random.random() < 0.3:  # 30% chance of shuffling
            random.shuffle(activities_to_schedule)
        
        scheduled_activities = set()
        
        # For each activity, find a suitable slot and space
        for activity_id in activities_to_schedule:
            # Skip if already scheduled
            if activity_id in scheduled_activities:
                continue
            
            # Skip if no suitable spaces
            if activity_id not in self.suitable_spaces or not self.suitable_spaces[activity_id]:
                continue
            
            valid_assignments = []
            probabilities = []
            total_probability = 0.0
            
            # Find valid slot-space combinations
            for slot in self.slots:
                # Get available spaces in this slot
                available_spaces = available_slots[slot]
                
                # Check suitable spaces that are available
                for space_id in self.suitable_spaces[activity_id]:
                    if space_id not in available_spaces:
                        continue
                    
                    # Get pheromone and heuristic values
                    pheromone = self.pheromone[activity_id][slot].get(space_id, 1.0)
                    heuristic = self.heuristic[activity_id][slot].get(space_id, 0.0)
                    
                    # Skip if unsuitable
                    if heuristic <= 0:
                        continue
                    
                    # Calculate probability
                    probability = pow(pheromone, self.params.alpha) * pow(heuristic, self.params.beta)
                    
                    if probability > 0:
                        valid_assignments.append((slot, space_id))
                        probabilities.append(probability)
                        total_probability += probability
            
            # If no valid assignments, skip this activity
            if not valid_assignments:
                continue
            
            # Select a slot-space combination based on probabilities
            if total_probability > 0:
                # Use roulette wheel selection (faster than random.choices)
                r = random.random() * total_probability
                cum_prob = 0.0
                selected_idx = 0
                
                for i, prob in enumerate(probabilities):
                    cum_prob += prob
                    if cum_prob >= r:
                        selected_idx = i
                        break
            else:
                # If all probabilities are zero, select randomly
                selected_idx = random.randint(0, len(valid_assignments) - 1)
            
            # Get the selected assignment
            selected_slot, selected_space = valid_assignments[selected_idx]
            
            # Assign activity to the selected slot and space
            schedule[selected_slot][selected_space] = activity_id
            scheduled_activities.add(activity_id)
            
            # Update available slots and spaces
            available_slots[selected_slot].remove(selected_space)
        
        return schedule
    
    def evaluate_solution(self, solution: Dict) -> Tuple:
        """
        Evaluate a timetable solution using the multi-objective evaluator.
        
        Args:
            solution: Timetable solution to evaluate
            
        Returns:
            Tuple: Multi-objective fitness values
        """
        return multi_objective_evaluator(
            solution, 
            self.activities_dict, 
            self.groups_dict, 
            self.lecturers_dict, 
            self.spaces_dict, 
            self.slots
        )
    
    def update_pheromones(self, solutions: List[Dict], fitness_values: List[Tuple], best_solution: Dict):
        """
        Update pheromone matrix based on solution quality.
        
        Args:
            solutions: List of solutions
            fitness_values: Corresponding fitness values
            best_solution: The best solution found so far
        """
        # Evaporate existing pheromones
        evap_factor = 1.0 - self.params.evaporation_rate
        for activity_id in self.pheromone:
            for slot in self.pheromone[activity_id]:
                for space_id in self.pheromone[activity_id][slot]:
                    self.pheromone[activity_id][slot][space_id] *= evap_factor
        
        # Deposit pheromones for each solution proportional to quality
        for solution, fitness in zip(solutions, fitness_values):
            # Calculate solution quality (lower fitness means higher quality)
            quality = 1.0 / (1.0 + sum(fitness[:4]) + fitness[4] * 10.0)
            deposit_amount = quality * self.params.q_value
            
            # Deposit pheromones
            for slot, space_assignments in solution.items():
                for space_id, activity_id in space_assignments.items():
                    if activity_id is not None and activity_id in self.pheromone and slot in self.pheromone[activity_id] and space_id in self.pheromone[activity_id][slot]:
                        self.pheromone[activity_id][slot][space_id] += deposit_amount
        
        # Elitist strategy: deposit extra pheromone on the best solution
        if best_solution:
            # Evaluate best solution if needed
            if not hasattr(self, 'best_solution_fitness') or self.best_solution != best_solution:
                self.best_solution_fitness = self.evaluate_solution(best_solution)
                
            best_quality = 1.0 / (1.0 + sum(self.best_solution_fitness[:4]) + self.best_solution_fitness[4] * 10.0)
            best_deposit = best_quality * self.params.q_value * self.params.elitist_weight
            
            # Deposit extra pheromones
            for slot, space_assignments in best_solution.items():
                for space_id, activity_id in space_assignments.items():
                    if activity_id is not None and activity_id in self.pheromone and slot in self.pheromone[activity_id] and space_id in self.pheromone[activity_id][slot]:
                        self.pheromone[activity_id][slot][space_id] += best_deposit
    
    def update_archive(self, solution: Dict, fitness: Tuple):
        """
        Update the archive of non-dominated solutions (Pareto front).
        
        Args:
            solution: Solution to consider for the archive
            fitness: Fitness values of the solution
        """
        # Check if this solution is dominated by any in the archive
        is_dominated = False
        dominated_indices = []
        
        for i, (archived_solution, archived_fitness) in enumerate(self.archive):
            # Check if current solution dominates archived one
            if self._dominates(fitness, archived_fitness):
                dominated_indices.append(i)
            # Check if archived solution dominates current one
            elif self._dominates(archived_fitness, fitness):
                is_dominated = True
                break
        
        # If not dominated, add to archive
        if not is_dominated:
            # Remove solutions that are dominated by this one
            for idx in sorted(dominated_indices, reverse=True):
                del self.archive[idx]
            
            # Add solution to archive (make a deep copy to avoid reference issues)
            self.archive.append((copy.deepcopy(solution), fitness))
    
    def _dominates(self, fitness1: Tuple, fitness2: Tuple) -> bool:
        """
        Check if fitness1 dominates fitness2 (for minimization).
        
        Args:
            fitness1: First fitness tuple
            fitness2: Second fitness tuple
            
        Returns:
            bool: True if fitness1 dominates fitness2
        """
        better_in_one = False
        for f1, f2 in zip(fitness1, fitness2):
            if f1 > f2:  # For minimization, lower is better
                return False
            if f1 < f2:
                better_in_one = True
        return better_in_one
    
    @track_computational_resources
    def run(self, dataset_path: str = None, output_dir: str = "results/aco"):
        """
        Run the ACO algorithm.
        
        Args:
            dataset_path: Path to dataset file
            output_dir: Directory to save results
            
        Returns:
            Tuple: Best solution, Pareto front, and metrics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize resource tracker
        self.resource_tracker = ResourceTracker("ACO")
        self.resource_tracker.start()
        
        # Load data if needed
        if not self.activities_dict:
            self.load_data(dataset_path)
        
        # Initialize archive and history
        self.archive = []
        self.iteration_history = []
        
        # Initialize best solution tracking
        best_solution = None
        best_fitness_value = float('inf')  # Lower is better
        
        # Main ACO loop
        for iteration in range(self.params.num_iterations):
            iteration_start = time.time()
            
            # Construct solutions in parallel using vectorized operations where possible
            solutions = []
            fitness_values = []
            
            # Generate solutions with all ants
            for _ in range(self.params.num_ants):
                # Construct a solution
                solution = self.construct_solution()
                fitness = self.evaluate_solution(solution)
                
                # Calculate a simple aggregated fitness for comparison
                simple_fitness = sum(fitness[:4]) + fitness[4] * 10.0
                
                solutions.append(solution)
                fitness_values.append(fitness)
                
                # Update best solution
                if simple_fitness < best_fitness_value:
                    best_fitness_value = simple_fitness
                    best_solution = copy.deepcopy(solution)
                
                # Update Pareto front archive
                self.update_archive(solution, fitness)
            
            # Update pheromones
            self.update_pheromones(solutions, fitness_values, best_solution)
            
            # Record iteration data
            iteration_time = time.time() - iteration_start
            best_in_iteration = min(fitness_values, key=lambda x: sum(x[:4]) + x[4] * 10.0)
            
            self.iteration_history.append({
                'iteration': iteration + 1,
                'best_fitness': best_in_iteration,
                'archive_size': len(self.archive),
                'time': iteration_time
            })
            
            # Log performance metrics
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"Iteration {iteration + 1}/{self.params.num_iterations}: "
                      f"Best Fitness = {best_fitness_value:.4f}, "
                      f"Pareto Front Size = {len(self.archive)}, "
                      f"Time = {iteration_time:.2f}s")
        
        # Store best solution and extract Pareto front
        self.best_solution = best_solution
        self.pareto_front = [fitness for _, fitness in self.archive]
        
        # Stop resource tracking
        self.resource_tracker.stop()
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Generate visualizations
        try:
            self._generate_visualizations(output_dir)
        except Exception as e:
            print(f"Warning: Error generating visualizations: {e}")
        
        return best_solution, self.pareto_front, metrics
    
    def _calculate_metrics(self):
        """Calculate performance metrics for the algorithm."""
        metrics = {}
        
        # Basic metrics
        metrics['iterations'] = self.params.num_iterations
        metrics['pareto_front_size'] = len(self.pareto_front)
        metrics['execution_time'] = self.resource_tracker.execution_time
        metrics['peak_memory_usage'] = self.resource_tracker.peak_memory_usage
        
        # Calculate hypervolume if possible
        if self.pareto_front:
            try:
                # Define reference point (worst possible values + margin)
                reference_point = (50, 100, 100, 200, 1.0, 1.0, 1.0, 1.0, 1.0)
                metrics['hypervolume'] = calculate_hypervolume(self.pareto_front, reference_point)
            except Exception as e:
                print(f"Warning: Error calculating hypervolume: {e}")
                metrics['hypervolume'] = 0.0
        else:
            metrics['hypervolume'] = 0.0
        
        # Calculate spread/diversity
        if len(self.pareto_front) >= 2:
            try:
                metrics['spread'] = calculate_spread(self.pareto_front)
            except Exception as e:
                print(f"Warning: Error calculating spread: {e}")
                metrics['spread'] = 0.0
        else:
            metrics['spread'] = 0.0
        
        return metrics
    
    def _generate_visualizations(self, output_dir: str):
        """Generate visualizations for the results."""
        # Create visualization directory
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Only generate visualizations if we have solutions
        if not self.pareto_front:
            return
        
        # 2D Pareto front plot
        try:
            plot_pareto_front_2d(
                [self.pareto_front],
                ["ACO"],
                ["red"],
                ["o"],
                0, 4,  # Professor conflicts vs Soft constraints
                PROFESSOR_CONFLICTS, SOFT_CONSTRAINTS,
                "ACO: Professor Conflicts vs Soft Constraints",
                vis_dir,
                "pareto_2d_prof_vs_soft.png"
            )
        except Exception as e:
            print(f"Warning: Error generating 2D Pareto plot: {e}")
        
        # Plot convergence history if available
        if self.iteration_history:
            try:
                # Extract data
                iterations = [entry['iteration'] for entry in self.iteration_history]
                obj_values = [entry['best_fitness'][0] for entry in self.iteration_history]
                
                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(iterations, obj_values, 'r-', linewidth=2, marker='o')
                plt.title('ACO Convergence')
                plt.xlabel('Iteration')
                plt.ylabel(PROFESSOR_CONFLICTS)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(vis_dir, "convergence.png"), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Warning: Error generating convergence plot: {e}")


#===============================================================================
# === BCO: Bee Colony Optimization Implementation ===
#===============================================================================

class BCOParameters:
    """Parameters for the BCO algorithm."""
    
    def __init__(self, 
                 num_bees: int = 50,
                 num_iterations: int = 50,
                 employed_ratio: float = 0.5,  # Ratio of employed bees
                 onlooker_ratio: float = 0.3,  # Ratio of onlooker bees
                 scout_ratio: float = 0.2,     # Ratio of scout bees
                 limit_trials: int = 5,        # Limit of unsuccessful trials before abandonment
                 alpha: float = 1.0):          # Exploitation parameter
        self.num_bees = num_bees
        self.num_iterations = num_iterations
        self.employed_ratio = employed_ratio
        self.onlooker_ratio = onlooker_ratio
        self.scout_ratio = scout_ratio
        self.limit_trials = limit_trials
        self.alpha = alpha


class BeeColonyOptimization:
    """
    Bee Colony Optimization implementation for university timetable scheduling.
    
    This class implements a multi-objective BCO approach with employed bees,
    onlooker bees, and scout bees working together to find optimal solutions.
    """
    
    def __init__(self, params: BCOParameters = None):
        """
        Initialize the BCO algorithm.
        
        Args:
            params: Parameters for the BCO algorithm
        """
        self.params = params or BCOParameters()
        
        # Data containers
        self.activities_dict = {}
        self.groups_dict = {}
        self.spaces_dict = {}
        self.lecturers_dict = {}
        self.slots = []
        
        # Solution tracking
        self.food_sources = []  # List of (solution, fitness, trials) tuples
        self.best_solution = None
        self.best_fitness = None
        self.archive = []  # Pareto front archive
        self.pareto_front = []
        self.iteration_history = []
        
        # Resource tracking
        self.resource_tracker = None
    
    def load_data(self, dataset_path: str = None):
        """
        Load data from the dataset file and prepare all necessary data structures.
        
        Args:
            dataset_path: Path to the dataset file
        """
        # Load dataset
        data = load_data(dataset_path)
        
        self.activities_dict = data[0]
        self.groups_dict = data[1]
        self.spaces_dict = data[2]
        self.lecturers_dict = data[3]
        self.slots = data[12]
        
        print(f"BCO: Data loaded with {len(self.activities_dict)} activities, "
              f"{len(self.spaces_dict)} spaces, {len(self.slots)} slots")
    
    def initialize_food_sources(self):
        """
        Initialize food sources (solutions) for the BCO algorithm.
        
        Each food source is a tuple of (solution, fitness, trials).
        """
        self.food_sources = []
        
        # Number of employed bees
        num_employed_bees = int(self.params.num_bees * self.params.employed_ratio)
        
        # Generate initial random solutions
        for _ in range(num_employed_bees):
            solution = self.generate_random_solution()
            fitness = self.evaluate_solution(solution)
            trials = 0  # Initialize trials to zero
            
            self.food_sources.append((solution, fitness, trials))
            
            # Update archive
            self.update_archive(solution, fitness)
        
        # Initialize best solution
        self.update_best_solution()
    
    def generate_random_solution(self) -> Dict:
        """
        Generate a random timetable solution.
        
        Returns:
            Dict: A randomly generated timetable solution
        """
        # Initialize empty schedule
        schedule = {slot: {space_id: None for space_id in self.spaces_dict} 
                   for slot in self.slots}
        
        # Get random order of activities to schedule
        activities = list(self.activities_dict.keys())
        random.shuffle(activities)
        
        # Track available slots and spaces
        available_slots = {slot: set(self.spaces_dict.keys()) for slot in self.slots}
        
        # Try to assign each activity
        for activity_id in activities:
            activity = self.activities_dict[activity_id]
            assigned = False
            
            # Try slots in random order
            slot_order = list(self.slots)
            random.shuffle(slot_order)
            
            for slot in slot_order:
                if not available_slots[slot]:  # Skip if no spaces available in this slot
                    continue
                
                # Find suitable spaces
                suitable_spaces = []
                for space_id in available_slots[slot]:
                    space = self.spaces_dict[space_id]
                    is_suitable = True
                    
                    # Check room capacity
                    activity_size = sum(self.groups_dict[g_id].size for g_id in activity.group_ids if g_id in self.groups_dict)
                    if activity_size > space.size:
                        is_suitable = False
                    
                    # Add space if suitable
                    if is_suitable:
                        suitable_spaces.append(space_id)
                
                if suitable_spaces:  # If suitable spaces found
                    # Randomly select a space
                    selected_space = random.choice(suitable_spaces)
                    
                    # Assign activity
                    schedule[slot][selected_space] = activity_id
                    available_slots[slot].remove(selected_space)
                    assigned = True
                    break
            
            # If activity couldn't be assigned, just continue
            if not assigned:
                continue
        
        return schedule
    
    def employed_bee_phase(self):
        """
        Employed bee phase of the BCO algorithm.
        
        Each employed bee searches for a better solution around its assigned food source.
        """
        new_food_sources = []
        
        # For each food source
        for solution, fitness, trials in self.food_sources:
            # Generate a neighbor solution
            neighbor = self.generate_neighbor_solution(solution)
            neighbor_fitness = self.evaluate_solution(neighbor)
            
            # If neighbor is better, replace the current solution
            if self.is_better(neighbor_fitness, fitness):
                new_food_sources.append((neighbor, neighbor_fitness, 0))  # Reset trials
                
                # Update archive
                self.update_archive(neighbor, neighbor_fitness)
            else:
                new_food_sources.append((solution, fitness, trials + 1))  # Increment trials
        
        self.food_sources = new_food_sources
        self.update_best_solution()
    
    def onlooker_bee_phase(self):
        """
        Onlooker bee phase of the BCO algorithm.
        
        Onlooker bees select food sources based on their fitness and search around them.
        """
        # Calculate selection probabilities
        total_fitness = 0.0
        inverse_fitness = []
        
        for _, fitness, _ in self.food_sources:
            # For minimization, use inverse of the fitness
            inv_fit = 1.0 / (1.0 + sum(fitness[:4]) + fitness[4] * 10.0)
            inverse_fitness.append(inv_fit)
            total_fitness += inv_fit
        
        if total_fitness > 0:
            probabilities = [fit / total_fitness for fit in inverse_fitness]
        else:
            # If all fitness values are extremely poor, use uniform distribution
            probabilities = [1.0 / len(self.food_sources) for _ in self.food_sources]
        
        # Number of onlooker bees
        num_onlooker_bees = int(self.params.num_bees * self.params.onlooker_ratio)
        
        # For each onlooker bee
        for _ in range(num_onlooker_bees):
            # Select a food source
            selected_idx = random.choices(range(len(self.food_sources)), probabilities)[0]
            solution, fitness, trials = self.food_sources[selected_idx]
            
            # Generate a neighbor solution
            neighbor = self.generate_neighbor_solution(solution)
            neighbor_fitness = self.evaluate_solution(neighbor)
            
            # If neighbor is better, replace the current solution
            if self.is_better(neighbor_fitness, fitness):
                self.food_sources[selected_idx] = (neighbor, neighbor_fitness, 0)  # Reset trials
                
                # Update archive
                self.update_archive(neighbor, neighbor_fitness)
            else:
                self.food_sources[selected_idx] = (solution, fitness, trials + 1)  # Increment trials
        
        self.update_best_solution()
    
    def scout_bee_phase(self):
        """
        Scout bee phase of the BCO algorithm.
        
        Scout bees abandon food sources that have been tried too many times without improvement.
        """
        for i, (solution, fitness, trials) in enumerate(self.food_sources):
            if trials >= self.params.limit_trials:
                # Generate a new random solution
                new_solution = self.generate_random_solution()
                new_fitness = self.evaluate_solution(new_solution)
                
                # Replace the abandoned solution
                self.food_sources[i] = (new_solution, new_fitness, 0)  # Reset trials
                
                # Update archive
                self.update_archive(new_solution, new_fitness)
        
        self.update_best_solution()
    
    def generate_neighbor_solution(self, solution: Dict) -> Dict:
        """
        Generate a neighboring solution by making a small change to the given solution.
        
        Args:
            solution: The current solution
            
        Returns:
            Dict: A neighboring solution
        """
        # Make a deep copy of the solution
        neighbor = copy.deepcopy(solution)
        
        # Determine the number of swaps to make (between 1 and 3)
        num_swaps = random.randint(1, 3)
        
        for _ in range(num_swaps):
            # Select two random slots
            slots = list(neighbor.keys())
            if len(slots) < 2:  # Need at least 2 slots to swap
                break
                
            slot1, slot2 = random.sample(slots, 2)
            
            # Get activities in these slots
            activities1 = [activity_id for space_id, activity_id in neighbor[slot1].items() if activity_id is not None]
            activities2 = [activity_id for space_id, activity_id in neighbor[slot2].items() if activity_id is not None]
            
            if not activities1 or not activities2:  # Skip if any slot is empty
                continue
            
            # Select random activities to swap
            activity1 = random.choice(activities1)
            activity2 = random.choice(activities2)
            
            # Find the spaces containing these activities
            space1 = next(space_id for space_id, act_id in neighbor[slot1].items() if act_id == activity1)
            space2 = next(space_id for space_id, act_id in neighbor[slot2].items() if act_id == activity2)
            
            # Swap activities
            neighbor[slot1][space1], neighbor[slot2][space2] = neighbor[slot2][space2], neighbor[slot1][space1]
        
        return neighbor
    
    def evaluate_solution(self, solution: Dict) -> Tuple:
        """
        Evaluate a timetable solution using the multi-objective evaluator.
        
        Args:
            solution: Timetable solution to evaluate
            
        Returns:
            Tuple: Multi-objective fitness values
        """
        return multi_objective_evaluator(
            solution, 
            self.activities_dict, 
            self.groups_dict, 
            self.lecturers_dict, 
            self.spaces_dict, 
            self.slots
        )
    
    def update_archive(self, solution: Dict, fitness: Tuple):
        """
        Update the archive of non-dominated solutions (Pareto front).
        
        Args:
            solution: Solution to consider for the archive
            fitness: Fitness values of the solution
        """
        # Check if this solution is dominated by any in the archive
        is_dominated = False
        dominated_indices = []
        
        for i, (archived_solution, archived_fitness) in enumerate(self.archive):
            # Check if current solution dominates archived one
            if self.dominates(fitness, archived_fitness):
                dominated_indices.append(i)
            # Check if archived solution dominates current one
            elif self.dominates(archived_fitness, fitness):
                is_dominated = True
                break
        
        # If not dominated, add to archive
        if not is_dominated:
            # Remove solutions that are dominated by this one
            for idx in sorted(dominated_indices, reverse=True):
                del self.archive[idx]
            
            # Add solution to archive (make a deep copy to avoid reference issues)
            self.archive.append((copy.deepcopy(solution), fitness))
    
    def is_better(self, fitness1: Tuple, fitness2: Tuple) -> bool:
        """
        Check if fitness1 is better than fitness2 for a minimization problem.
        
        Args:
            fitness1: First fitness tuple
            fitness2: Second fitness tuple
            
        Returns:
            bool: True if fitness1 is better than fitness2
        """
        # Simple aggregation of objectives for comparison
        # Lower is better for our minimization problem
        value1 = sum(fitness1[:4]) + fitness1[4] * 10.0
        value2 = sum(fitness2[:4]) + fitness2[4] * 10.0
        
        return value1 < value2
    
    def dominates(self, fitness1: Tuple, fitness2: Tuple) -> bool:
        """
        Check if fitness1 dominates fitness2 (for minimization).
        
        Args:
            fitness1: First fitness tuple
            fitness2: Second fitness tuple
            
        Returns:
            bool: True if fitness1 dominates fitness2
        """
        better_in_one = False
        for f1, f2 in zip(fitness1, fitness2):
            if f1 > f2:  # For minimization, lower is better
                return False
            if f1 < f2:
                better_in_one = True
        return better_in_one
    
    def update_best_solution(self):
        """
        Update the best solution found so far.
        """
        if not self.food_sources:
            return
        
        # Find the solution with the best fitness
        best_idx = 0
        best_val = float('inf')
        
        for i, (_, fitness, _) in enumerate(self.food_sources):
            # Simple aggregation for comparison
            val = sum(fitness[:4]) + fitness[4] * 10.0
            
            if val < best_val:
                best_val = val
                best_idx = i
        
        # Update best solution
        self.best_solution = copy.deepcopy(self.food_sources[best_idx][0])
        self.best_fitness = self.food_sources[best_idx][1]
    
    @track_computational_resources
    def run(self, dataset_path: str = None, output_dir: str = "results/bco"):
        """
        Run the BCO algorithm.
        
        Args:
            dataset_path: Path to dataset file
            output_dir: Directory to save results
            
        Returns:
            Tuple: Best solution, Pareto front, and metrics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize resource tracker
        self.resource_tracker = ResourceTracker("BCO")
        self.resource_tracker.start()
        
        # Load data if needed
        if not self.activities_dict:
            self.load_data(dataset_path)
        
        # Initialize food sources and archive
        self.archive = []
        self.iteration_history = []
        self.initialize_food_sources()
        
        # Main BCO loop
        for iteration in range(self.params.num_iterations):
            iteration_start = time.time()
            
            # Employed bee phase
            self.employed_bee_phase()
            
            # Onlooker bee phase
            self.onlooker_bee_phase()
            
            # Scout bee phase
            self.scout_bee_phase()
            
            # Record iteration data
            iteration_time = time.time() - iteration_start
            
            self.iteration_history.append({
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'archive_size': len(self.archive),
                'time': iteration_time
            })
            
            # Log performance metrics
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"Iteration {iteration + 1}/{self.params.num_iterations}: "
                      f"Best Fitness = {sum(self.best_fitness[:4]) + self.best_fitness[4] * 10.0:.4f}, "
                      f"Pareto Front Size = {len(self.archive)}, "
                      f"Time = {iteration_time:.2f}s")
        
        # Extract Pareto front
        self.pareto_front = [fitness for _, fitness in self.archive]
        
        # Stop resource tracking
        self.resource_tracker.stop()
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Generate visualizations
        try:
            self._generate_visualizations(output_dir)
        except Exception as e:
            print(f"Warning: Error generating visualizations: {e}")
        
        return self.best_solution, self.pareto_front, metrics
    
    def _calculate_metrics(self):
        """
        Calculate performance metrics for the algorithm.
        
        Returns:
            Dict: Performance metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['iterations'] = self.params.num_iterations
        metrics['pareto_front_size'] = len(self.pareto_front)
        metrics['execution_time'] = self.resource_tracker.execution_time
        metrics['peak_memory_usage'] = self.resource_tracker.peak_memory_usage
        
        # Calculate hypervolume if possible
        if self.pareto_front:
            try:
                # Define reference point (worst possible values + margin)
                reference_point = (50, 100, 100, 200, 1.0, 1.0, 1.0, 1.0, 1.0)
                metrics['hypervolume'] = calculate_hypervolume(self.pareto_front, reference_point)
            except Exception as e:
                print(f"Warning: Error calculating hypervolume: {e}")
                metrics['hypervolume'] = 0.0
        else:
            metrics['hypervolume'] = 0.0
        
        # Calculate spread/diversity
        if len(self.pareto_front) >= 2:
            try:
                metrics['spread'] = calculate_spread(self.pareto_front)
            except Exception as e:
                print(f"Warning: Error calculating spread: {e}")
                metrics['spread'] = 0.0
        else:
            metrics['spread'] = 0.0
        
        return metrics
    
    def _generate_visualizations(self, output_dir: str):
        """
        Generate visualizations for the results.
        
        Args:
            output_dir: Directory to save visualizations
        """
        # Create visualization directory
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Only generate visualizations if we have solutions
        if not self.pareto_front:
            return
        
        # 2D Pareto front plot
        try:
            plot_pareto_front_2d(
                [self.pareto_front],
                ["BCO"],
                ["blue"],
                ["o"],
                0, 4,  # Professor conflicts vs Soft constraints
                PROFESSOR_CONFLICTS, SOFT_CONSTRAINTS,
                "BCO: Professor Conflicts vs Soft Constraints",
                vis_dir,
                "pareto_2d_prof_vs_soft.png"
            )
        except Exception as e:
            print(f"Warning: Error generating 2D Pareto plot: {e}")
        
        # Plot convergence history if available
        if self.iteration_history:
            try:
                # Extract data
                iterations = [entry['iteration'] for entry in self.iteration_history]
                obj_values = [entry['best_fitness'][0] for entry in self.iteration_history]
                
                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(iterations, obj_values, 'b-', linewidth=2, marker='o')
                plt.title('BCO Convergence')
                plt.xlabel('Iteration')
                plt.ylabel(PROFESSOR_CONFLICTS)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(vis_dir, "convergence.png"), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Warning: Error generating convergence plot: {e}")


#===============================================================================
# === PSO: Particle Swarm Optimization Implementation ===
#===============================================================================

class PSOParameters:
    """Parameters for the PSO algorithm."""
    
    def __init__(self, 
                 num_particles: int = 50,
                 num_iterations: int = 50,
                 inertia_weight: float = 0.7,     # Inertia weight
                 cognitive_weight: float = 1.5,   # Cognitive weight (personal best)
                 social_weight: float = 1.5,      # Social weight (global best)
                 velocity_clamp: float = 0.1):    # Velocity clamping factor
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.velocity_clamp = velocity_clamp


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization implementation for university timetable scheduling.
    
    This class implements a multi-objective PSO approach for timetable optimization.
    """
    
    def __init__(self, params: PSOParameters = None):
        """
        Initialize the PSO algorithm.
        
        Args:
            params: Parameters for the PSO algorithm
        """
        self.params = params or PSOParameters()
        
        # Data containers
        self.activities_dict = {}
        self.groups_dict = {}
        self.spaces_dict = {}
        self.lecturers_dict = {}
        self.slots = []
        
        # PSO particles
        self.particles = []  # List of (position, velocity, fitness) tuples
        self.personal_best = []  # List of (position, fitness) tuples
        self.global_best = None  # (position, fitness) tuple
        self.archive = []  # Pareto front archive
        self.pareto_front = []
        self.iteration_history = []
        
        # Resource tracking
        self.resource_tracker = None
    
    def load_data(self, dataset_path: str = None):
        """
        Load data from the dataset file and prepare all necessary data structures.
        
        Args:
            dataset_path: Path to the dataset file
        """
        # Load dataset
        data = load_data(dataset_path)
        
        self.activities_dict = data[0]
        self.groups_dict = data[1]
        self.spaces_dict = data[2]
        self.lecturers_dict = data[3]
        self.slots = data[12]
        
        print(f"PSO: Data loaded with {len(self.activities_dict)} activities, "
              f"{len(self.spaces_dict)} spaces, {len(self.slots)} slots")
    
    def initialize_particles(self):
        """
        Initialize particles for the PSO algorithm.
        
        Each particle has a position (timetable solution), velocity, and fitness.
        """
        self.particles = []
        self.personal_best = []
        
        for _ in range(self.params.num_particles):
            # Generate random position (timetable solution)
            position = self.generate_random_solution()
            fitness = self.evaluate_solution(position)
            
            # Initialize velocity (list of potential swaps)
            velocity = self.initialize_velocity()
            
            # Add particle
            self.particles.append((position, velocity, fitness))
            
            # Initialize personal best
            self.personal_best.append((copy.deepcopy(position), fitness))
            
            # Update archive
            self.update_archive(position, fitness)
        
        # Initialize global best
        self.update_global_best()
    
    def generate_random_solution(self) -> Dict:
        """
        Generate a random timetable solution.
        
        Returns:
            Dict: A randomly generated timetable solution
        """
        # Initialize empty schedule
        schedule = {slot: {space_id: None for space_id in self.spaces_dict} 
                   for slot in self.slots}
        
        # Get random order of activities to schedule
        activities = list(self.activities_dict.keys())
        random.shuffle(activities)
        
        # Track available slots and spaces
        available_slots = {slot: set(self.spaces_dict.keys()) for slot in self.slots}
        
        # Try to assign each activity
        for activity_id in activities:
            activity = self.activities_dict[activity_id]
            assigned = False
            
            # Try slots in random order
            slot_order = list(self.slots)
            random.shuffle(slot_order)
            
            for slot in slot_order:
                if not available_slots[slot]:  # Skip if no spaces available in this slot
                    continue
                
                # Find suitable spaces
                suitable_spaces = []
                for space_id in available_slots[slot]:
                    space = self.spaces_dict[space_id]
                    if is_space_suitable(space, activity, self.groups_dict):
                        suitable_spaces.append(space_id)
                
                if suitable_spaces:  # If suitable spaces found
                    # Randomly select a space
                    selected_space = random.choice(suitable_spaces)
                    
                    # Assign activity
                    schedule[slot][selected_space] = activity_id
                    available_slots[slot].remove(selected_space)
                    assigned = True
                    break
            
            # If activity couldn't be assigned, just continue
            if not assigned:
                continue
        
        return schedule
    
    def initialize_velocity(self) -> List:
        """
        Initialize velocity for a particle.
        
        In the context of timetable scheduling, velocity is represented as a list of
        potential swaps (slot1, space1, slot2, space2) with associated probabilities.
        
        Returns:
            List: Initial velocity as a list of potential swaps
        """
        # For simplicity, initialize with empty velocity
        return []
    
    def update_velocity(self, position: Dict, velocity: List, personal_best: Dict, global_best: Dict) -> List:
        """
        Update velocity of a particle based on PSO rules.
        
        Args:
            position: Current position (timetable solution)
            velocity: Current velocity (list of potential swaps)
            personal_best: Personal best position
            global_best: Global best position
            
        Returns:
            List: Updated velocity
        """
        # For timetable scheduling, velocity update is based on swaps needed to move
        # from current position towards personal best and global best
        new_velocity = []
        
        # Inertia component: Keep some of the previous velocity
        inertia_velocity = velocity[:int(len(velocity) * self.params.inertia_weight)]
        new_velocity.extend(inertia_velocity)
        
        # Cognitive component: Move towards personal best
        cognitive_velocity = self.get_swaps_between(position, personal_best)
        cognitive_sample = random.sample(
            cognitive_velocity, 
            min(len(cognitive_velocity), int(len(cognitive_velocity) * self.params.cognitive_weight))
        )
        new_velocity.extend(cognitive_sample)
        
        # Social component: Move towards global best
        social_velocity = self.get_swaps_between(position, global_best)
        social_sample = random.sample(
            social_velocity, 
            min(len(social_velocity), int(len(social_velocity) * self.params.social_weight))
        )
        new_velocity.extend(social_sample)
        
        # Clamp velocity size
        max_velocity_size = int(self.params.velocity_clamp * len(self.activities_dict))
        if len(new_velocity) > max_velocity_size:
            new_velocity = random.sample(new_velocity, max_velocity_size)
        
        return new_velocity
    
    def get_swaps_between(self, solution1: Dict, solution2: Dict) -> List:
        """
        Determine swaps needed to transform solution1 into solution2.
        
        Args:
            solution1: First solution
            solution2: Second solution
            
        Returns:
            List: List of swaps (slot1, space1, slot2, space2)
        """
        swaps = []
        
        # Find activities that are assigned differently in the two solutions
        for slot1, space_assignments1 in solution1.items():
            for space1, activity1 in space_assignments1.items():
                if activity1 is None:
                    continue
                
                # Find where this activity is in solution2
                found = False
                for slot2, space_assignments2 in solution2.items():
                    for space2, activity2 in space_assignments2.items():
                        if activity2 == activity1 and (slot1 != slot2 or space1 != space2):
                            # Found different assignment, add swap
                            swaps.append((slot1, space1, slot2, space2))
                            found = True
                            break
                    if found:
                        break
        
        return swaps
    
    def update_position(self, position: Dict, velocity: List) -> Dict:
        """
        Update position of a particle based on its velocity.
        
        Args:
            position: Current position (timetable solution)
            velocity: Current velocity (list of potential swaps)
            
        Returns:
            Dict: Updated position
        """
        # Make a copy of the current position
        new_position = copy.deepcopy(position)
        
        # Apply swaps from velocity
        for swap in velocity:
            if len(swap) != 4:  # Skip invalid swaps
                continue
                
            slot1, space1, slot2, space2 = swap
            
            # Skip if any slot or space doesn't exist
            if (slot1 not in new_position or 
                slot2 not in new_position or 
                space1 not in new_position[slot1] or 
                space2 not in new_position[slot2]):
                continue
            
            # Skip if any slot-space combination has no activity
            if (new_position[slot1][space1] is None or 
                new_position[slot2][space2] is None):
                continue
            
            # Swap activities
            new_position[slot1][space1], new_position[slot2][space2] = \
                new_position[slot2][space2], new_position[slot1][space1]
        
        return new_position
    
    def evaluate_solution(self, solution: Dict) -> Tuple:
        """
        Evaluate a timetable solution using the multi-objective evaluator.
        
        Args:
            solution: Timetable solution to evaluate
            
        Returns:
            Tuple: Multi-objective fitness values
        """
        return multi_objective_evaluator(
            solution, 
            self.activities_dict, 
            self.groups_dict, 
            self.lecturers_dict, 
            self.spaces_dict, 
            self.slots
        )
    
    def update_archive(self, solution: Dict, fitness: Tuple):
        """
        Update the archive of non-dominated solutions (Pareto front).
        
        Args:
            solution: Solution to consider for the archive
            fitness: Fitness values of the solution
        """
        # Check if this solution is dominated by any in the archive
        is_dominated = False
        dominated_indices = []
        
        for i, (archived_solution, archived_fitness) in enumerate(self.archive):
            # Check if current solution dominates archived one
            if self.dominates(fitness, archived_fitness):
                dominated_indices.append(i)
            # Check if archived solution dominates current one
            elif self.dominates(archived_fitness, fitness):
                is_dominated = True
                break
        
        # If not dominated, add to archive
        if not is_dominated:
            # Remove solutions that are dominated by this one
            for idx in sorted(dominated_indices, reverse=True):
                del self.archive[idx]
            
            # Add solution to archive (make a deep copy to avoid reference issues)
            self.archive.append((copy.deepcopy(solution), fitness))
    
    def dominates(self, fitness1: Tuple, fitness2: Tuple) -> bool:
        """
        Check if fitness1 dominates fitness2 (for minimization).
        
        Args:
            fitness1: First fitness tuple
            fitness2: Second fitness tuple
            
        Returns:
            bool: True if fitness1 dominates fitness2
        """
        better_in_one = False
        for f1, f2 in zip(fitness1, fitness2):
            if f1 > f2:  # For minimization, lower is better
                return False
            if f1 < f2:
                better_in_one = True
        return better_in_one
    
    def is_better(self, fitness1: Tuple, fitness2: Tuple) -> bool:
        """
        Check if fitness1 is better than fitness2 for a minimization problem.
        
        Args:
            fitness1: First fitness tuple
            fitness2: Second fitness tuple
            
        Returns:
            bool: True if fitness1 is better than fitness2
        """
        # Simple aggregation of objectives for comparison
        # Lower is better for our minimization problem
        value1 = sum(fitness1[:4]) + fitness1[4] * 10.0
        value2 = sum(fitness2[:4]) + fitness2[4] * 10.0
        
        return value1 < value2
    
    def update_global_best(self):
        """
        Update the global best solution based on personal bests.
        """
        if not self.personal_best:
            return
        
        # Find the best personal best
        best_idx = 0
        best_val = float('inf')
        
        for i, (_, fitness) in enumerate(self.personal_best):
            # Simple aggregation for comparison
            val = sum(fitness[:4]) + fitness[4] * 10.0
            
            if val < best_val:
                best_val = val
                best_idx = i
        
        # Update global best
        self.global_best = (copy.deepcopy(self.personal_best[best_idx][0]), 
                           self.personal_best[best_idx][1])
    
    @track_computational_resources
    def run(self, dataset_path: str = None, output_dir: str = "results/pso"):
        """
        Run the PSO algorithm.
        
        Args:
            dataset_path: Path to dataset file
            output_dir: Directory to save results
            
        Returns:
            Tuple: Best solution, Pareto front, and metrics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize resource tracker
        self.resource_tracker = ResourceTracker("PSO")
        self.resource_tracker.start()
        
        # Load data if needed
        if not self.activities_dict:
            self.load_data(dataset_path)
        
        # Initialize particles and archive
        self.archive = []
        self.iteration_history = []
        self.initialize_particles()
        
        # Main PSO loop
        for iteration in range(self.params.num_iterations):
            iteration_start = time.time()
            
            # For each particle
            for i, (position, velocity, fitness) in enumerate(self.particles):
                # Update velocity
                new_velocity = self.update_velocity(
                    position, 
                    velocity, 
                    self.personal_best[i][0],  # Personal best position
                    self.global_best[0]   # Global best position
                )
                
                # Update position
                new_position = self.update_position(position, new_velocity)
                new_fitness = self.evaluate_solution(new_position)
                
                # Update particle
                self.particles[i] = (new_position, new_velocity, new_fitness)
                
                # Update personal best
                if self.is_better(new_fitness, self.personal_best[i][1]):
                    self.personal_best[i] = (copy.deepcopy(new_position), new_fitness)
                
                # Update archive
                self.update_archive(new_position, new_fitness)
            
            # Update global best
            self.update_global_best()
            
            # Record iteration data
            iteration_time = time.time() - iteration_start
            
            self.iteration_history.append({
                'iteration': iteration + 1,
                'best_fitness': self.global_best[1],
                'archive_size': len(self.archive),
                'time': iteration_time
            })
            
            # Log performance metrics
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"Iteration {iteration + 1}/{self.params.num_iterations}: "
                      f"Best Fitness = {sum(self.global_best[1][:4]) + self.global_best[1][4] * 10.0:.4f}, "
                      f"Pareto Front Size = {len(self.archive)}, "
                      f"Time = {iteration_time:.2f}s")
        
        # Extract Pareto front
        self.pareto_front = [fitness for _, fitness in self.archive]
        
        # Stop resource tracking
        self.resource_tracker.stop()
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Generate visualizations
        try:
            self._generate_visualizations(output_dir)
        except Exception as e:
            print(f"Warning: Error generating visualizations: {e}")
        
        return self.global_best[0], self.pareto_front, metrics
    
    def _calculate_metrics(self):
        """
        Calculate performance metrics for the algorithm.
        
        Returns:
            Dict: Performance metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['iterations'] = self.params.num_iterations
        metrics['pareto_front_size'] = len(self.pareto_front)
        metrics['execution_time'] = self.resource_tracker.execution_time
        metrics['peak_memory_usage'] = self.resource_tracker.peak_memory_usage
        
        # Calculate hypervolume if possible
        if self.pareto_front:
            try:
                # Define reference point (worst possible values + margin)
                reference_point = (50, 100, 100, 200, 1.0, 1.0, 1.0, 1.0, 1.0)
                metrics['hypervolume'] = calculate_hypervolume(self.pareto_front, reference_point)
            except Exception as e:
                print(f"Warning: Error calculating hypervolume: {e}")
                metrics['hypervolume'] = 0.0
        else:
            metrics['hypervolume'] = 0.0
        
        # Calculate spread/diversity
        if len(self.pareto_front) >= 2:
            try:
                metrics['spread'] = calculate_spread(self.pareto_front)
            except Exception as e:
                print(f"Warning: Error calculating spread: {e}")
                metrics['spread'] = 0.0
        else:
            metrics['spread'] = 0.0
        
        return metrics
    
    def _generate_visualizations(self, output_dir: str):
        """
        Generate visualizations for the results.
        
        Args:
            output_dir: Directory to save visualizations
        """
        # Create visualization directory
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Only generate visualizations if we have solutions
        if not self.pareto_front:
            return
        
        # 2D Pareto front plot
        try:
            plot_pareto_front_2d(
                [self.pareto_front],
                ["PSO"],
                ["green"],
                ["o"],
                0, 4,  # Professor conflicts vs Soft constraints
                PROFESSOR_CONFLICTS, SOFT_CONSTRAINTS,
                "PSO: Professor Conflicts vs Soft Constraints",
                vis_dir,
                "pareto_2d_prof_vs_soft.png"
            )
        except Exception as e:
            print(f"Warning: Error generating 2D Pareto plot: {e}")
        
        # Plot convergence history if available
        if self.iteration_history:
            try:
                # Extract data
                iterations = [entry['iteration'] for entry in self.iteration_history]
                obj_values = [entry['best_fitness'][0] for entry in self.iteration_history]
                
                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(iterations, obj_values, 'g-', linewidth=2, marker='o')
                plt.title('PSO Convergence')
                plt.xlabel('Iteration')
                plt.ylabel(PROFESSOR_CONFLICTS)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(vis_dir, "convergence.png"), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Warning: Error generating convergence plot: {e}")


#===============================================================================
# === Main Function ===
#===============================================================================

def run_algorithm(algorithm_name: str, dataset_path: str = None, output_dir: str = None, **kwargs):
    """
    Run a specified colony optimization algorithm with given parameters.
    
    Args:
        algorithm_name: Name of the algorithm to run ('aco', 'bco', or 'pso')
        dataset_path: Path to the dataset file
        output_dir: Directory to save results
        **kwargs: Additional parameters for the algorithm
        
    Returns:
        Tuple: Best solution, Pareto front, and metrics
    """
    if algorithm_name.lower() == 'aco':
        # Ant Colony Optimization
        aco_params = ACOParameters(
            num_ants=kwargs.get('num_ants', 30),
            num_iterations=kwargs.get('num_iterations', 20),
            evaporation_rate=kwargs.get('evaporation_rate', 0.5),
            alpha=kwargs.get('alpha', 1.0),
            beta=kwargs.get('beta', 2.0),
            q_value=kwargs.get('q_value', 100.0),
            elitist_weight=kwargs.get('elitist_weight', 2.0)
        )
        
        alg = AntColonyOptimization(aco_params)
        output_dir = output_dir or "results/aco"
        
    elif algorithm_name.lower() == 'bco':
        # Bee Colony Optimization
        bco_params = BCOParameters(
            num_bees=kwargs.get('num_bees', 50),
            num_iterations=kwargs.get('num_iterations', 50),
            employed_ratio=kwargs.get('employed_ratio', 0.5),
            onlooker_ratio=kwargs.get('onlooker_ratio', 0.3),
            scout_ratio=kwargs.get('scout_ratio', 0.2),
            limit_trials=kwargs.get('limit_trials', 5),
            alpha=kwargs.get('alpha', 1.0)
        )
        
        alg = BeeColonyOptimization(bco_params)
        output_dir = output_dir or "results/bco"
        
    elif algorithm_name.lower() == 'pso':
        # Particle Swarm Optimization
        pso_params = PSOParameters(
            num_particles=kwargs.get('num_particles', 50),
            num_iterations=kwargs.get('num_iterations', 50),
            inertia_weight=kwargs.get('inertia_weight', 0.7),
            cognitive_weight=kwargs.get('cognitive_weight', 1.5),
            social_weight=kwargs.get('social_weight', 1.5),
            velocity_clamp=kwargs.get('velocity_clamp', 0.1)
        )
        
        alg = ParticleSwarmOptimization(pso_params)
        output_dir = output_dir or "results/pso"
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Must be one of 'aco', 'bco', or 'pso'.")
    
    # Run the algorithm
    best_solution, pareto_front, metrics = alg.run(dataset_path, output_dir)
    
    print(f"\n{algorithm_name.upper()} Optimization Complete")
    print(f"Best Solution Fitness: {sum(alg.evaluate_solution(best_solution)[:4]) + alg.evaluate_solution(best_solution)[4] * 10.0:.4f}")
    print(f"Pareto Front Size: {len(pareto_front)}")
    print(f"Execution Time: {metrics['execution_time']:.2f} seconds")
    print(f"Peak Memory Usage: {metrics['peak_memory_usage']:.2f} MB")
    
    return best_solution, pareto_front, metrics


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Colony Optimization Algorithms for Timetable Scheduling")
    parser.add_argument("--algorithm", type=str, required=True, choices=["aco", "bco", "pso"], 
                        help="Algorithm to run (aco, bco, or pso)")
    parser.add_argument("--dataset", type=str, default=None, 
                        help="Path to dataset file (default: use environment variable or default path)")
    parser.add_argument("--output", type=str, default=None, 
                        help="Directory to save results (default: results/<algorithm>)")
    parser.add_argument("--iterations", type=int, default=None, 
                        help="Number of iterations to run (default: algorithm-specific)")
    parser.add_argument("--population", type=int, default=None, 
                        help="Population size (ants, bees, or particles) (default: algorithm-specific)")
    
    args = parser.parse_args()
    
    # Prepare kwargs based on algorithm
    kwargs = {}
    if args.iterations is not None:
        if args.algorithm == "aco":
            kwargs["num_iterations"] = args.iterations
        else:  # bco or pso
            kwargs["num_iterations"] = args.iterations
    
    if args.population is not None:
        if args.algorithm == "aco":
            kwargs["num_ants"] = args.population
        elif args.algorithm == "bco":
            kwargs["num_bees"] = args.population
        else:  # pso
            kwargs["num_particles"] = args.population
    
    # Run the selected algorithm
    run_algorithm(args.algorithm, args.dataset, args.output, **kwargs)
