"""
TimeTable Scheduler - Azure Fabric Notebook Script (Part 2: Evaluator Functions)
This script is formatted for easy conversion to a Jupyter notebook.
Each cell is marked with '# CELL: {description}' comments.
"""

# CELL: Import libraries
import numpy as np
import random
from typing import Dict, List, Tuple, Set, Any

# CELL: Hard constraint evaluation functions
def hard_constraint_space_capacity(schedule, activities_dict, groups_dict, spaces_dict, slots):
    """
    Check if the space capacity constraint is violated.
    The constraint is violated if the number of students exceeds the room capacity.
    
    Returns:
        int: Number of violations
    """
    violations = 0
    
    for slot_idx, slot in enumerate(slots):
        for space_id, activity_id in schedule[slot_idx].items():
            if activity_id and space_id in spaces_dict and activity_id in activities_dict:
                activity = activities_dict[activity_id]
                space = spaces_dict[space_id]
                
                # Calculate total students for activity
                total_students = sum(groups_dict[g_id].size for g_id in activity.group_ids if g_id in groups_dict)
                
                # Check if space capacity is exceeded
                if total_students > space.capacity:
                    violations += 1
    
    return violations

def hard_constraint_lecturer_clash(schedule, activities_dict, lecturers_dict, slots):
    """
    Check if there are any lecturer clashes in the schedule.
    A lecturer clash occurs when a lecturer is assigned to multiple activities at the same time.
    
    Returns:
        int: Number of violations
    """
    violations = 0
    
    for slot_idx, slot in enumerate(slots):
        # Track lecturers assigned in this slot
        assigned_lecturers = set()
        
        for space_id, activity_id in schedule[slot_idx].items():
            if activity_id and activity_id in activities_dict:
                activity = activities_dict[activity_id]
                lecturer_id = activity.lecturer_id
                
                # Check if lecturer is already assigned in this slot
                if lecturer_id in assigned_lecturers:
                    violations += 1
                else:
                    assigned_lecturers.add(lecturer_id)
    
    return violations

def hard_constraint_group_clash(schedule, activities_dict, groups_dict, slots):
    """
    Check if there are any student group clashes in the schedule.
    A group clash occurs when a group is assigned to multiple activities at the same time.
    
    Returns:
        int: Number of violations
    """
    violations = 0
    
    for slot_idx, slot in enumerate(slots):
        # Track groups assigned in this slot
        assigned_groups = set()
        
        for space_id, activity_id in schedule[slot_idx].items():
            if activity_id and activity_id in activities_dict:
                activity = activities_dict[activity_id]
                
                # Check for clashes for each group in the activity
                for group_id in activity.group_ids:
                    if group_id in assigned_groups:
                        violations += 1
                    else:
                        assigned_groups.add(group_id)
    
    return violations

def hard_constraint_space_clash(schedule, slots):
    """
    Check if there are any space clashes in the schedule.
    A space clash occurs when multiple activities are assigned to the same space at the same time.
    This function actually checks if the schedule is valid in terms of having at most one
    activity assigned to each space in each time slot.
    
    Returns:
        int: Number of violations (this should be 0 for a valid schedule)
    """
    violations = 0
    
    for slot_idx, slot in enumerate(slots):
        # Track how many activities are assigned to each space
        space_count = {}
        
        for space_id, activity_id in schedule[slot_idx].items():
            if activity_id:  # Only count if there's an activity assigned
                if space_id in space_count:
                    space_count[space_id] += 1
                else:
                    space_count[space_id] = 1
        
        # Count spaces with more than one activity
        for space_id, count in space_count.items():
            if count > 1:
                violations += (count - 1)  # Count as violations all activities beyond the first
    
    return violations

# CELL: Soft constraint evaluation functions
def soft_constraint_consecutive_activities(schedule, activities_dict, groups_dict, slots):
    """
    Evaluate the soft constraint for consecutive activities.
    The constraint prefers consecutive activities for the same group to minimize idle time.
    
    Returns:
        float: Penalty score (lower is better)
    """
    penalty = 0.0
    days = list(set(slot[0] for slot in slots))  # Extract unique days
    
    # For each day, check for idle periods per group
    for day in days:
        day_slots = [(idx, slot) for idx, slot in enumerate(slots) if slot[0] == day]
        
        # Get activities per group for this day
        group_activities = {}
        
        for slot_idx, slot in day_slots:
            for space_id, activity_id in schedule[slot_idx].items():
                if activity_id and activity_id in activities_dict:
                    activity = activities_dict[activity_id]
                    
                    for group_id in activity.group_ids:
                        if group_id not in group_activities:
                            group_activities[group_id] = []
                        
                        group_activities[group_id].append((slot_idx, slot[1]))  # (slot_idx, period)
        
        # Calculate idle periods for each group
        for group_id, activities in group_activities.items():
            # Sort by period
            activities.sort(key=lambda x: x[1])
            
            if len(activities) <= 1:
                continue
                
            # Calculate gaps
            for i in range(len(activities) - 1):
                current_period = activities[i][1]
                next_period = activities[i + 1][1]
                
                # Add penalty for gap (idle periods)
                gap = next_period - current_period - 1
                if gap > 0:
                    penalty += gap * 0.5  # Weight for this constraint
    
    return penalty

def soft_constraint_preferred_times(schedule, activities_dict, slots):
    """
    Evaluate the soft constraint for preferred activity times.
    The constraint checks if activities are scheduled during their preferred times.
    
    Returns:
        float: Penalty score (lower is better)
    """
    penalty = 0.0
    
    for slot_idx, slot in enumerate(slots):
        day, period = slot
        
        for space_id, activity_id in schedule[slot_idx].items():
            if activity_id and activity_id in activities_dict:
                activity = activities_dict[activity_id]
                
                # Check if activity has preferred times
                preferred_times = []
                for constraint in activity.constraints:
                    if constraint.get("type") == "preferred_times":
                        preferred_times = constraint.get("times", [])
                
                if preferred_times and (day, period) not in preferred_times:
                    penalty += 1.0  # Weight for this constraint
    
    return penalty

# CELL: Main schedule evaluation function
def evaluate_schedule(schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots):
    """
    Evaluate a schedule based on hard and soft constraints.
    
    Args:
        schedule: The schedule to evaluate
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of student groups
        spaces_dict: Dictionary of spaces
        lecturers_dict: Dictionary of lecturers
        slots: List of time slots
        
    Returns:
        Tuple[float, int, float]: Fitness score, hard violations count, soft penalty score
    """
    # Evaluate hard constraints
    hard_violations = (
        hard_constraint_space_capacity(schedule, activities_dict, groups_dict, spaces_dict, slots) +
        hard_constraint_lecturer_clash(schedule, activities_dict, lecturers_dict, slots) +
        hard_constraint_group_clash(schedule, activities_dict, groups_dict, slots) +
        hard_constraint_space_clash(schedule, slots)
    )
    
    # Evaluate soft constraints
    soft_violations = (
        soft_constraint_consecutive_activities(schedule, activities_dict, groups_dict, slots) +
        soft_constraint_preferred_times(schedule, activities_dict, slots)
    )
    
    # Calculate fitness score (negative because we're minimizing)
    # Hard constraints are given much higher weight
    fitness = -(hard_violations * 100 + soft_violations)
    
    return fitness, hard_violations, soft_violations

# CELL: Multi-objective evaluator for GA
def multi_objective_evaluator(schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots):
    """
    Multi-objective evaluation function for genetic algorithms.
    Returns multiple fitness values for NSGA-II algorithm.
    
    Returns:
        Tuple: (space_violation, lecturer_violation, group_violation, consecutive_violation, total)
    """
    # Evaluate individual hard constraints
    space_capacity_violations = hard_constraint_space_capacity(
        schedule, activities_dict, groups_dict, spaces_dict, slots
    )
    
    lecturer_clashes = hard_constraint_lecturer_clash(
        schedule, activities_dict, lecturers_dict, slots
    )
    
    group_clashes = hard_constraint_group_clash(
        schedule, activities_dict, groups_dict, slots
    )
    
    space_clashes = hard_constraint_space_clash(schedule, slots)
    
    # Evaluate individual soft constraints
    consecutive_penalty = soft_constraint_consecutive_activities(
        schedule, activities_dict, groups_dict, slots
    )
    
    preferred_times_penalty = soft_constraint_preferred_times(
        schedule, activities_dict, slots
    )
    
    # Sum hard violations
    hard_violations = space_capacity_violations + lecturer_clashes + group_clashes + space_clashes
    
    # Sum soft violations
    soft_violations = consecutive_penalty + preferred_times_penalty
    
    # Total fitness (negative because we minimize)
    total_fitness = -(hard_violations * 100 + soft_violations)
    
    # Return all objectives for multi-objective optimization
    return (
        space_capacity_violations,
        lecturer_clashes + group_clashes,
        space_clashes,
        consecutive_penalty + preferred_times_penalty,
        -total_fitness  # Convert to positive for easier analysis
    )

# CELL: Schedule parsing utility functions
def parse_ga_schedule(individual, activities_dict, spaces_dict, slots):
    """
    Convert a GA individual to a schedule format.
    
    Args:
        individual: GA individual (encoded schedule)
        activities_dict: Dictionary of activities
        spaces_dict: Dictionary of spaces
        slots: List of time slots
        
    Returns:
        List[Dict]: Schedule representation
    """
    schedule = [{} for _ in range(len(slots))]
    
    for activity_idx, (slot_idx, space_id) in enumerate(individual):
        if slot_idx < len(slots) and space_id in spaces_dict:
            activity_id = list(activities_dict.keys())[activity_idx]
            schedule[slot_idx][space_id] = activity_id
    
    return schedule

def parse_rl_schedule(rl_state, slots, spaces_ids):
    """
    Convert an RL state to a schedule format.
    
    Args:
        rl_state: RL state representation
        slots: List of time slots
        spaces_ids: List of space IDs
        
    Returns:
        List[Dict]: Schedule representation
    """
    schedule = [{} for _ in range(len(slots))]
    
    for slot_idx, slot_activities in enumerate(rl_state):
        for activity_idx, space_idx in enumerate(slot_activities):
            if space_idx < len(spaces_ids):
                space_id = spaces_ids[space_idx]
                activity_id = f"A{activity_idx+1}"
                schedule[slot_idx][space_id] = activity_id
    
    return schedule

# CELL: Visualization functions
def visualize_schedule(schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, title="Timetable Schedule"):
    """
    Visualize the schedule as a colorful grid.
    
    Args:
        schedule: The schedule to visualize
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of student groups
        spaces_dict: Dictionary of spaces
        lecturers_dict: Dictionary of lecturers
        slots: List of time slots
        title: Title for the plot
    """
    # Extract unique days and periods from slots
    days = list(set(slot[0] for slot in slots))
    periods = list(set(slot[1] for slot in slots))
    days.sort()
    periods.sort()
    
    # Create a mapping of slots to their indices
    slot_to_idx = {slot: idx for idx, slot in enumerate(slots)}
    
    # Create a 2D grid for each space
    spaces = list(spaces_dict.keys())
    
    # Create a figure with subplots for each space
    fig, axes = plt.subplots(len(spaces), 1, figsize=(12, 4 * len(spaces)))
    if len(spaces) == 1:
        axes = [axes]  # Ensure axes is a list even with only one space
    
    fig.suptitle(title, fontsize=16)
    
    # Define color map for activities
    unique_activities = list(set(
        activity_id for slot_data in schedule for activity_id in slot_data.values() 
        if activity_id
    ))
    cmap = plt.cm.get_cmap('tab20', len(unique_activities))
    activity_colors = {act_id: cmap(i) for i, act_id in enumerate(unique_activities)}
    
    # Plot each space's schedule
    for space_idx, space_id in enumerate(spaces):
        ax = axes[space_idx]
        
        # Create a grid for this space
        grid = np.zeros((len(days), len(periods)))
        grid.fill(np.nan)  # Fill with NaN for empty cells
        
        # Fill in the activities
        for slot_idx, slot_data in enumerate(schedule):
            if space_id in slot_data and slot_data[space_id]:
                day_idx = days.index(slots[slot_idx][0])
                period_idx = periods.index(slots[slot_idx][1])
                
                # Mark as occupied with activity index
                activity_id = slot_data[space_id]
                if activity_id in unique_activities:
                    grid[day_idx, period_idx] = unique_activities.index(activity_id)
        
        # Create a mask for empty slots
        mask = np.isnan(grid)
        
        # Plot the grid with custom colormap
        sns.heatmap(grid, cmap=cmap, mask=mask, cbar=False, 
                   linewidths=0.5, linecolor='gray', ax=ax)
        
        # Add activity labels
        for day_idx, day in enumerate(days):
            for period_idx, period in enumerate(periods):
                slot_idx = slot_to_idx.get((day, period))
                if slot_idx is not None and space_id in schedule[slot_idx]:
                    activity_id = schedule[slot_idx][space_id]
                    if activity_id:
                        activity = activities_dict.get(activity_id)
                        if activity:
                            # Format the text (subject, lecturer, groups)
                            lecturer = lecturers_dict.get(activity.lecturer_id)
                            lecturer_name = lecturer.name if lecturer else activity.lecturer_id
                            
                            groups = [groups_dict.get(g_id) for g_id in activity.group_ids]
                            group_names = ", ".join(g.name if g else g_id for g_id, g in zip(activity.group_ids, groups))
                            
                            text = f"{activity.subject}\n{lecturer_name}\n{group_names}"
                            ax.text(period_idx + 0.5, day_idx + 0.5, text, 
                                   ha='center', va='center', fontsize=8)
        
        # Set labels
        ax.set_title(f"Space: {spaces_dict[space_id].name or space_id}")
        ax.set_ylabel("Day")
        ax.set_xlabel("Period")
        ax.set_yticks(np.arange(len(days)) + 0.5)
        ax.set_xticks(np.arange(len(periods)) + 0.5)
        ax.set_yticklabels(days)
        ax.set_xticklabels(periods)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
    return fig

def visualize_pareto_front(fitness_values, metrics_names=['Obj1', 'Obj2', 'Obj3', 'Obj4', 'Total'], title="Pareto Front"):
    """
    Visualize the Pareto front for multi-objective optimization.
    Creates multiple pairwise plots for the objectives.
    
    Args:
        fitness_values: List of fitness tuples from multi-objective evaluation
        metrics_names: Names of the metrics for axis labels
        title: Title for the plot
    """
    num_objectives = len(fitness_values[0]) - 1  # Excluding the total
    
    # Create a figure with subplots for each pair of objectives
    rows = (num_objectives * (num_objectives - 1)) // 2
    fig, axes = plt.subplots(rows, 1, figsize=(10, 4 * rows))
    
    if rows == 1:
        axes = [axes]  # Ensure axes is a list
    
    fig.suptitle(title, fontsize=16)
    
    # Extract objective values
    objectives = [[individual[i] for individual in fitness_values] for i in range(num_objectives)]
    
    # Plot each pair of objectives
    subplot_idx = 0
    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            ax = axes[subplot_idx]
            
            # Scatter plot for this pair of objectives
            sc = ax.scatter(objectives[i], objectives[j], alpha=0.7, 
                           c=[ind[-1] for ind in fitness_values], cmap='viridis')
            
            # Add colorbar for total fitness
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Total Fitness')
            
            # Set labels
            ax.set_xlabel(metrics_names[i])
            ax.set_ylabel(metrics_names[j])
            ax.set_title(f"{metrics_names[i]} vs {metrics_names[j]}")
            
            # Identify and highlight non-dominated solutions (Pareto front)
            dominated = np.zeros(len(fitness_values), dtype=bool)
            
            for k, (f1, f2) in enumerate(zip(objectives[i], objectives[j])):
                dominated[k] = any((objectives[i][l] <= f1 and objectives[j][l] < f2) or 
                                  (objectives[i][l] < f1 and objectives[j][l] <= f2) 
                                  for l in range(len(fitness_values)) if l != k)
            
            # Highlight non-dominated solutions
            non_dominated_indices = np.where(~dominated)[0]
            non_dominated_x = [objectives[i][idx] for idx in non_dominated_indices]
            non_dominated_y = [objectives[j][idx] for idx in non_dominated_indices]
            
            # Sort non-dominated solutions for line plot
            if non_dominated_x:
                sorted_indices = np.argsort(non_dominated_x)
                sorted_x = [non_dominated_x[idx] for idx in sorted_indices]
                sorted_y = [non_dominated_y[idx] for idx in sorted_indices]
                
                ax.plot(sorted_x, sorted_y, 'r-', linewidth=2, label='Pareto Front')
                ax.scatter(non_dominated_x, non_dominated_y, s=100, facecolors='none', 
                          edgecolors='r', linewidth=2)
            
            ax.legend()
            subplot_idx += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
    return fig

def visualize_convergence(history, title="Algorithm Convergence"):
    """
    Visualize the convergence of the algorithm over generations/episodes.
    
    Args:
        history: Dictionary with history data (rewards, violations)
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards/fitness over time
    if 'rewards' in history:
        # For RL
        rewards = history['rewards']
        x_values = range(1, len(rewards) + 1)
        ax1.plot(x_values, rewards, 'b-')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Progression')
        
        # Add best reward line
        if 'best_reward' in history:
            ax1.axhline(y=history['best_reward'], color='r', linestyle='--', 
                        label=f'Best Reward: {history["best_reward"]:.2f}')
        
        ax1.legend()
        
    elif 'fitness' in history:
        # For GA
        fitness_history = history['fitness']
        x_values = range(1, len(fitness_history) + 1)
        
        # Plot average fitness
        avg_fitness = [np.mean([f[-1] for f in gen_fitness]) for gen_fitness in fitness_history]
        ax1.plot(x_values, avg_fitness, 'b-', label='Average Fitness')
        
        # Plot best fitness
        best_fitness = [min([f[-1] for f in gen_fitness]) for gen_fitness in fitness_history]
        ax1.plot(x_values, best_fitness, 'g-', label='Best Fitness')
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Progression')
        ax1.legend()
    
    # Plot violations over time
    if 'hard_violations' in history and 'soft_violations' in history:
        hard_violations = history['hard_violations']
        soft_violations = history['soft_violations']
        
        x_values = range(1, len(hard_violations) + 1)
        
        ax2.plot(x_values, hard_violations, 'r-', label='Hard Constraints')
        ax2.plot(x_values, soft_violations, 'y-', label='Soft Constraints')
        
        ax2.set_xlabel('Generation/Episode')
        ax2.set_ylabel('Violations')
        ax2.set_title('Constraint Violations')
        ax2.legend()
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
    return fig
