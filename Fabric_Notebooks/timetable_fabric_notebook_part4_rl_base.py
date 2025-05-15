"""
TimeTable Scheduler - Azure Fabric Notebook Script (Part 4: RL Base Implementation)
This script is formatted for easy conversion to a Jupyter notebook.
Each cell is marked with '# CELL: {description}' comments.
"""

# CELL: Import libraries
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
import pickle

# Try to import TensorFlow for DQN
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - DQN implementation will be limited")

# CELL: RL State and Action Representations

def initialize_state():
    """
    Initialize an empty state for RL.
    A state is represented as a list of lists, one for each timeslot.
    Each inner list contains the space assignments for activities in that slot.
    
    Returns:
        List[List]: Empty initial state
    """
    # Create empty state (one list per slot)
    return [[] for _ in range(len(slots))]

def get_available_actions(state, slot_idx, activities_dict, spaces_dict, groups_dict, slots):
    """
    Get available actions for a given state and slot.
    An action is a tuple (activity_idx, space_idx) that assigns an activity to a space.
    
    Args:
        state: Current state
        slot_idx: Index of the current slot
        activities_dict: Dictionary of activities
        spaces_dict: Dictionary of spaces
        groups_dict: Dictionary of student groups
        slots: List of time slots
        
    Returns:
        List[Tuple]: List of available actions
    """
    spaces = list(spaces_dict.keys())
    activities = list(activities_dict.keys())
    
    # Get activities and spaces already assigned in current slot
    assigned_activities = set()
    assigned_spaces = set()
    assigned_lecturers = set()
    assigned_groups = set()
    
    for slot_activities in state[slot_idx]:
        if slot_activities:
            activity_idx, space_idx = slot_activities
            if activity_idx < len(activities):
                assigned_activities.add(activity_idx)
                assigned_spaces.add(space_idx)
                
                # Track assigned lecturers and groups
                activity = activities_dict[activities[activity_idx]]
                assigned_lecturers.add(activity.lecturer_id)
                assigned_groups.update(activity.group_ids)
    
    # Get activities already assigned in other slots
    activities_in_other_slots = set()
    for s_idx, slot_activities in enumerate(state):
        if s_idx != slot_idx:
            for activity_data in slot_activities:
                if activity_data:
                    activity_idx, _ = activity_data
                    activities_in_other_slots.add(activity_idx)
    
    # Find available actions
    available_actions = []
    
    for activity_idx, activity_id in enumerate(activities):
        if activity_idx in assigned_activities or activity_idx in activities_in_other_slots:
            continue  # Skip already assigned activities
        
        activity = activities_dict[activity_id]
        
        # Check lecturer availability
        if activity.lecturer_id in assigned_lecturers:
            continue
        
        # Check group availability
        if any(group_id in assigned_groups for group_id in activity.group_ids):
            continue
        
        # Find available spaces
        for space_idx, space_id in enumerate(spaces):
            if space_idx in assigned_spaces:
                continue  # Skip already assigned spaces
            
            space = spaces_dict[space_id]
            
            # Check space capacity
            total_students = sum(groups_dict[g_id].size for g_id in activity.group_ids if g_id in groups_dict)
            if total_students > space.capacity:
                continue  # Skip if space is too small
            
            # This is a valid action
            available_actions.append((activity_idx, space_idx))
    
    return available_actions

def apply_action(state, action, slot_idx):
    """
    Apply an action to a state to get a new state.
    
    Args:
        state: Current state
        action: Action to apply (activity_idx, space_idx)
        slot_idx: Index of the current slot
        
    Returns:
        List[List]: New state after applying the action
    """
    new_state = [slot.copy() for slot in state]
    
    # Add the activity-space assignment to the slot
    new_state[slot_idx].append(action)
    
    return new_state

def sarsa_state_to_schedule(sarsa_state, slots, spaces_ids):
    """
    Convert a SARSA state to a schedule format.
    
    Args:
        sarsa_state: SARSA state representation
        slots: List of time slots
        spaces_ids: List of space IDs
        
    Returns:
        List[Dict]: Schedule representation
    """
    schedule = [{} for _ in range(len(slots))]
    
    # Go through each slot in the state
    for slot_idx, slot_activities in enumerate(sarsa_state):
        # Go through each activity-space assignment in the slot
        for activity_space in slot_activities:
            if activity_space:  # Skip empty assignments
                activity_idx, space_idx = activity_space
                
                if isinstance(space_idx, int) and space_idx < len(spaces_ids):
                    space_id = spaces_ids[space_idx]
                    activity_id = f"A{activity_idx+1}"
                    schedule[slot_idx][space_id] = activity_id
    
    return schedule

# CELL: RL Q-Learning Implementation
def q_learning(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
              alpha=0.1, gamma=0.6, epsilon=0.1, n_episodes=1000):
    """
    Implementation of Q-Learning algorithm for timetable scheduling.
    
    Args:
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of student groups
        spaces_dict: Dictionary of spaces
        lecturers_dict: Dictionary of lecturers
        slots: List of time slots
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        n_episodes: Number of episodes
        
    Returns:
        Tuple[Dict, Dict]: History and Q-table
    """
    print(f"\nRunning Q-Learning algorithm (alpha={alpha}, gamma={gamma}, epsilon={epsilon}, episodes={n_episodes})...")
    spaces_ids = list(spaces_dict.keys())
    
    # Initialize Q-table
    q_table = {}
    
    # Episode history for analysis
    history = {
        'rewards': [],
        'best_reward': float('-inf'),
        'best_schedule': None,
        'hard_violations': [],
        'soft_violations': []
    }
    
    for episode in range(n_episodes):
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes}", end='\r')
        
        # Initialize state as empty schedule
        state = [[] for _ in range(len(slots))]
        total_reward = 0
        
        for slot_idx in range(len(slots)):
            # Get available actions
            actions = get_available_actions(state, slot_idx, activities_dict, spaces_dict, groups_dict, slots)
            
            # If no valid actions, skip slot
            if not actions:
                continue
            
            # Convert state to string for Q-table lookup
            state_key = str(state)
            
            # If state is not in Q-table, add it
            if state_key not in q_table:
                q_table[state_key] = {str(a): 0 for a in actions}
            
            # Epsilon-greedy policy
            if random.uniform(0, 1) < epsilon:
                # Exploration: choose random action
                action = random.choice(actions)
            else:
                # Exploitation: choose best action
                action = max(actions, key=lambda a: q_table[state_key].get(str(a), 0))
            
            # Apply action to get next state
            next_state = apply_action(state, action, slot_idx)
            
            # Get intermediate reward for this step
            interim_schedule = sarsa_state_to_schedule(next_state, slots, spaces_ids)
            reward, hard_v, soft_v = evaluate_schedule(
                interim_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
            )
            
            # Get next slot and available actions
            next_slot_idx = slot_idx + 1
            next_actions = []
            
            if next_slot_idx < len(slots):
                next_actions = get_available_actions(next_state, next_slot_idx, activities_dict, spaces_dict, groups_dict, slots)
            
            # Convert next state to string for Q-table lookup
            next_state_key = str(next_state)
            
            # If next state is not in Q-table, add it
            if next_state_key not in q_table and next_actions:
                q_table[next_state_key] = {str(a): 0 for a in next_actions}
            
            # Update Q-value
            if next_actions:
                # Get max Q-value for next state
                next_max_q = max(q_table[next_state_key].values())
                
                # Update Q-value
                q_table[state_key][str(action)] = (1 - alpha) * q_table[state_key].get(str(action), 0) + \
                                                 alpha * (reward + gamma * next_max_q)
            else:
                # Terminal state
                q_table[state_key][str(action)] = (1 - alpha) * q_table[state_key].get(str(action), 0) + \
                                                 alpha * reward
            
            # Update state and reward
            state = next_state
            total_reward += reward
        
        # Store episode results
        history['rewards'].append(total_reward)
        
        # Evaluate final schedule
        final_schedule = sarsa_state_to_schedule(state, slots, spaces_ids)
        final_reward, hard_violations, soft_violations = evaluate_schedule(
            final_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        
        history['hard_violations'].append(hard_violations)
        history['soft_violations'].append(soft_violations)
        
        # Update best schedule
        if final_reward > history['best_reward']:
            history['best_reward'] = final_reward
            history['best_schedule'] = final_schedule
        
        # Report progress
        if episode % max(1, n_episodes // 10) == 0 or episode == n_episodes - 1:
            print(f"Episode {episode}: Reward = {total_reward}, Hard Violations = {hard_violations}, Soft Violations = {soft_violations}")
    
    print(f"Q-Learning completed. Best reward: {history['best_reward']}")
    return history, q_table
