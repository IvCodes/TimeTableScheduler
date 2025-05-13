#!/usr/bin/env python3
# RL_Scirpt.py - Implementation of RL algorithms for timetable scheduling

import json
import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Any, Optional
import copy
import warnings
warnings.filterwarnings('ignore')

# Data Classes
@dataclass
class Activity:
    id: str
    group_ids: List[str]
    lecturer_id: str
    duration: int
    activity_type: str
    subject: str
    constraints: List[Dict]

@dataclass
class Group:
    id: str
    name: str
    size: int
    constraints: List[Dict]

@dataclass
class Space:
    id: str
    name: str
    capacity: int
    constraints: List[Dict]

@dataclass
class Lecturer:
    id: str
    name: str
    constraints: List[Dict]

# Data Loading Function
def load_data(dataset_path=None):
    """Load data from the dataset file.
    
    Args:
        dataset_path: Path to the dataset file. If None, will try to locate the file.
        
    Returns:
        Tuple of (activities_dict, groups_dict, spaces_dict, lecturers_dict,
                activities_list, groups_list, spaces_list, lecturers_list,
                activity_types, timeslots_list, days_list, periods_list, slots)
    """
    # Try to find the dataset file if path not provided
    if dataset_path is None:
        possible_paths = [
            os.path.join(os.getcwd(), 'sliit_computing_dataset.json'),
            os.path.join(os.getcwd(), 'data', 'sliit_computing_dataset.json'),
            os.path.join(os.getcwd(), 'Dataset', 'sliit_computing_dataset.json'),
            os.path.join(os.getcwd(), '..', '..', 'data', 'sliit_computing_dataset.json'),
            os.path.join(os.getcwd(), '..', '..', 'Dataset', 'sliit_computing_dataset.json'),
            os.path.join(os.getcwd(), '..', '..', '1. Genetic_EAs', 'sliit_computing_dataset.json')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                print(f"Found dataset at: {path}")
                break
        
        if dataset_path is None:
            raise FileNotFoundError("Could not find the dataset file. Please specify the path.")
    
    # Load data from the dataset file
    try:
        with open(dataset_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")
    
    # Initialize dictionaries and lists
    activities_dict = {}
    groups_dict = {}
    spaces_dict = {}
    lecturers_dict = {}
    
    activities_list = []
    groups_list = []
    spaces_list = []
    lecturers_list = []
    
    # Process spaces (classrooms)
    for space_data in data.get('spaces', []):
        # Handle capacity/size field variations
        capacity = space_data.get('capacity', space_data.get('size', 0))
        space_id = space_data.get('id', space_data.get('code', ''))
        name = space_data.get('name', f"Space {space_id}")
        
        space = Space(
            id=space_id,
            name=name,
            capacity=capacity,
            constraints=space_data.get('constraints', [])
        )
        spaces_dict[space.id] = space
        spaces_list.append(space)
    
    # Process groups/years
    for group_data in data.get('groups', data.get('years', [])):
        group_id = group_data.get('id')
        if not group_id:
            continue
            
        group = Group(
            id=group_id,
            name=group_data.get('name', f"Group {group_id}"),
            size=group_data.get('size', 0),
            constraints=group_data.get('constraints', [])
        )
        groups_dict[group.id] = group
        groups_list.append(group)
    
    # Process lecturers/teachers/users
    lecturer_sources = [
        # Check dedicated lecturer list
        data.get('lecturers', []),
        # Check users with role=lecturer
        [u for u in data.get('users', []) if u.get('role') == 'lecturer']
    ]
    
    for source in lecturer_sources:
        for lecturer_data in source:
            lecturer_id = lecturer_data.get('id')
            if not lecturer_id or lecturer_id in lecturers_dict:
                continue
                
            # Handle various name formats
            name = lecturer_data.get('name')
            if not name and lecturer_data.get('first_name'):
                name = f"{lecturer_data.get('first_name', '')} {lecturer_data.get('last_name', '')}".strip()
            if not name:
                name = f"Lecturer {lecturer_id}"
                
            lecturer = Lecturer(
                id=lecturer_id,
                name=name,
                constraints=lecturer_data.get('constraints', [])
            )
            lecturers_dict[lecturer.id] = lecturer
            lecturers_list.append(lecturer)
    
    # Process activities/courses
    for activity_data in data.get('activities', []):
        # Handle different ID/code fields
        activity_id = activity_data.get('id', activity_data.get('code', ''))
        if not activity_id:
            continue
            
        # Handle different subject/name fields
        subject = activity_data.get('subject', activity_data.get('name', f"Activity {activity_id}"))
        
        # Handle different group ID formats
        group_ids = []
        if activity_data.get('group_id'):
            group_ids = [activity_data.get('group_id')]
        elif activity_data.get('group_ids'):
            group_ids = activity_data.get('group_ids')
        elif activity_data.get('subgroup_ids'):
            group_ids = activity_data.get('subgroup_ids')
            
        # Handle different teacher/lecturer ID formats
        lecturer_id = None
        if activity_data.get('lecturer_id'):
            lecturer_id = activity_data.get('lecturer_id')
        elif activity_data.get('teacher_id'):
            lecturer_id = activity_data.get('teacher_id')
        elif activity_data.get('teacher_ids') and len(activity_data.get('teacher_ids')) > 0:
            lecturer_id = activity_data.get('teacher_ids')[0]
            
        # Handle activity type
        activity_type = activity_data.get('activity_type', activity_data.get('type', 'Lecture'))
        
        activity = Activity(
            id=activity_id,
            group_ids=group_ids,
            lecturer_id=lecturer_id,
            duration=activity_data.get('duration', 1),
            activity_type=activity_type,
            subject=subject,
            constraints=activity_data.get('constraints', [])
        )
        activities_dict[activity.id] = activity
        activities_list.append(activity)
    
    # Extract activity types
    activity_types = list(set(activity.activity_type for activity in activities_list))
    
    # Define time slots
    days_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    # Use abbreviated day codes to match the dataset
    day_codes = ["MON", "TUE", "WED", "THU", "FRI"]
    periods_list = [f"Period {i}" for i in range(1, 9)]
    
    # Create slots as a list of day-period combinations
    slots = []
    timeslots_list = []
    for day_idx, day in enumerate(days_list):
        for period_idx, period in enumerate(periods_list):
            # Use the dataset format for slots (e.g., MON1, TUE2, etc.)
            slot = f"{day_codes[day_idx]}{period_idx + 1}"
            slots.append(slot)
            timeslots_list.append({"day": day, "period": period, "slot": slot})
    
    print(f"Successfully loaded: {len(activities_dict)} activities, {len(groups_dict)} groups, {len(spaces_dict)} spaces, {len(lecturers_dict)} lecturers")
    
    return (
        activities_dict, groups_dict, spaces_dict, lecturers_dict,
        activities_list, groups_list, spaces_list, lecturers_list,
        activity_types, timeslots_list, days_list, periods_list, slots
    )

# Constraint Evaluation Functions
def is_space_suitable(space: Space, activity: Activity, groups_dict: Dict[str, Group]) -> bool:
    """Check if a space is suitable for an activity."""
    # Handle empty or invalid group_ids
    if not activity.group_ids:
        return True
        
    # Calculate total size of all groups involved in the activity
    total_size = 0
    for group_id in activity.group_ids:
        group = groups_dict.get(group_id)
        if group:
            total_size += group.size
    
    return space.capacity >= total_size

def hard_constraint_space_capacity(schedule, activities_dict, groups_dict, spaces_dict):
    """Evaluate if any activities are scheduled in spaces with insufficient capacity."""
    violations = 0
    
    for slot, assignments in schedule.items():
        for space_id, activity_id in assignments.items():
            if activity_id is None:
                continue
                
            activity = activities_dict.get(activity_id)
            space = spaces_dict.get(space_id)
            
            if activity and space:
                if not is_space_suitable(space, activity, groups_dict):
                    violations += 1
    
    return violations

def hard_constraint_lecturer_clash(schedule, activities_dict):
    """Evaluate if any lecturer is assigned to multiple activities in the same timeslot."""
    violations = 0
    
    for slot, assignments in schedule.items():
        lecturer_activities = defaultdict(int)
        
        for space_id, activity_id in assignments.items():
            if activity_id is None:
                continue
                
            activity = activities_dict.get(activity_id)
            if activity:
                lecturer_activities[activity.lecturer_id] += 1
        
        # Count violations where a lecturer has more than one activity in a timeslot
        for lecturer_id, count in lecturer_activities.items():
            if count > 1:
                violations += (count - 1)
    
    return violations

def hard_constraint_group_clash(schedule, activities_dict):
    """Evaluate if any group is assigned to multiple activities in the same timeslot."""
    violations = 0
    group_assignments = defaultdict(list)
    
    # Collect all group assignments
    for slot, space_assignments in schedule.items():
        for space_id, activity_id in space_assignments.items():
            if activity_id:  # If there is an activity assigned
                activity = activities_dict.get(activity_id)
                if activity:
                    # Handle multiple group_ids for a single activity
                    for group_id in activity.group_ids:
                        group_assignments[(slot, group_id)].append(activity_id)
    
    # Check for violations
    for (slot, group_id), activity_ids in group_assignments.items():
        if len(activity_ids) > 1:  # Group is assigned to multiple activities in the same slot
            violations += len(activity_ids) - 1
    
    return violations

def hard_constraint_space_clash(schedule):
    """Evaluate if any space is assigned multiple activities in the same timeslot."""
    violations = 0
    
    for slot, assignments in schedule.items():
        for space_id, activity_id in assignments.items():
            if activity_id is None:
                continue
            
            # This is a simplification, as our data structure should prevent this
            # But included for completeness
            if isinstance(activity_id, list) and len(activity_id) > 1:
                violations += (len(activity_id) - 1)
    
    return violations

def soft_constraint_consecutive_activities(schedule, activities_dict, groups_dict, slots):
    """Evaluate preference for consecutive activities for the same group."""
    penalties = 0
    max_penalties = 0
    
    # Create a mapping from slots to their index
    slot_indices = {slot: idx for idx, slot in enumerate(slots)}
    
    # For each group, find its activities throughout the day
    group_daily_activities = defaultdict(lambda: defaultdict(list))
    
    for slot, space_assignments in schedule.items():
        # Extract day from slot (assuming format like 'Monday_Period 1' or 'MON1')
        if '_' in slot:
            day = slot.split('_')[0]  # Format: 'Monday_Period 1'
        else:
            day = slot[:3]  # Format: 'MON1'
        
        for space_id, activity_id in space_assignments.items():
            if activity_id:
                activity = activities_dict.get(activity_id)
                if activity:
                    # Handle multiple group_ids for a single activity
                    for group_id in activity.group_ids:
                        group_daily_activities[group_id][day].append((slot, activity_id))
    
    # For each group and day, check if activities are consecutive
    for group_id, daily_acts in group_daily_activities.items():
        for day, activities in daily_acts.items():
            if len(activities) <= 1:
                continue  # Skip if only one activity
            
            # Sort activities by slot index
            activities.sort(key=lambda x: slot_indices.get(x[0], 0))
            
            # Check for gaps between activities
            for i in range(len(activities) - 1):
                slot1, _ = activities[i]
                slot2, _ = activities[i + 1]
                
                idx1 = slot_indices.get(slot1, 0)
                idx2 = slot_indices.get(slot2, 0)
                
                if idx2 - idx1 > 1:  # There is a gap
                    penalties += 1
            
            # Maximum possible penalties for this group on this day
            max_penalties += len(activities) - 1
    
    # Normalize penalties to a 0-1 scale (0 is best, 1 is worst)
    normalized_score = penalties / max_penalties if max_penalties > 0 else 0
    
    return normalized_score

def soft_constraint_preferred_times(schedule, activities_dict, lecturers_dict):
    """Evaluate lecturer preferences for specific timeslots."""
    violations = 0
    
    for slot, assignments in schedule.items():
        for space_id, activity_id in assignments.items():
            if activity_id is None:
                continue
                
            activity = activities_dict.get(activity_id)
            if not activity:
                continue
                
            lecturer = lecturers_dict.get(activity.lecturer_id)
            if not lecturer:
                continue
            
            # Check lecturer constraints for preferred times
            for constraint in lecturer.constraints:
                if constraint.get('type') == 'preferred_times':
                    preferred_times = constraint.get('times', [])
                    if slot not in preferred_times:
                        violations += 1
    
    return violations

def evaluate_schedule(schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots):
    """Evaluate a schedule based on hard and soft constraints."""
    # Hard constraints (must be satisfied)
    hard_violations = (
        hard_constraint_space_capacity(schedule, activities_dict, groups_dict, spaces_dict) +
        hard_constraint_lecturer_clash(schedule, activities_dict) +
        hard_constraint_group_clash(schedule, activities_dict) +
        hard_constraint_space_clash(schedule)
    )
    
    # Soft constraints (preferences)
    soft_violations = (
        soft_constraint_consecutive_activities(schedule, activities_dict, groups_dict, slots) +
        soft_constraint_preferred_times(schedule, activities_dict, lecturers_dict)
    )
    
    # Calculate fitness (higher is better)
    # We prioritize hard constraints with a larger weight
    hard_weight = 100
    soft_weight = 1
    
    fitness = -1 * (hard_weight * hard_violations + soft_weight * soft_violations)
    
    return fitness, hard_violations, soft_violations

# RL Utilities and Schedule Manipulation
def initialize_empty_schedule(slots, spaces_ids):
    """Initialize an empty schedule."""
    return {slot: {space_id: None for space_id in spaces_ids} for slot in slots}

def get_available_actions(state, slot_idx, activities_dict, spaces_dict, groups_dict, slots):
    """Get valid actions for the current state in Q-Learning/SARSA."""
    actions = []
    current_slot = slots[slot_idx]
    spaces_ids = list(spaces_dict.keys())
    
    # For each space, consider each activity
    for space_id in spaces_ids:
        space = spaces_dict[space_id]
        
        # Option to leave the space empty
        actions.append((space_id, None))
        
        # Option to assign activities
        for activity_id, activity in activities_dict.items():
            # Check if the activity is already scheduled somewhere
            already_scheduled = False
            for s_idx, s_content in enumerate(state):
                for item in s_content:
                    if isinstance(item, tuple) and len(item) == 2 and item[1] == activity_id:
                        already_scheduled = True
                        break
                if already_scheduled:
                    break
            
            if already_scheduled:
                continue
            
            # Check if the space is suitable
            if is_space_suitable(space, activity, groups_dict):
                actions.append((space_id, activity_id))
    
    return actions

def apply_action(state, action, slot_idx):
    """Apply an action to the current state."""
    new_state = copy.deepcopy(state)
    space_id, activity_id = action
    
    # Add the action to the appropriate slot
    new_state[slot_idx].append((space_id, activity_id))
    
    return new_state

def sarsa_state_to_schedule(sarsa_state, slots, spaces_ids):
    """Convert a SARSA state representation to a schedule dictionary."""
    schedule = initialize_empty_schedule(slots, spaces_ids)
    
    for slot_idx, slot_content in enumerate(sarsa_state):
        slot_name = slots[slot_idx]
        for item in slot_content:
            if isinstance(item, tuple) and len(item) == 2:
                space_id, activity_id = item
                if activity_id is not None:  # Only set if there's an activity
                    schedule[slot_name][space_id] = activity_id
    
    return schedule

def convert_sarsa_state_to_schedule(sarsa_state_schedule, activities_dict_local, slots, spaces_ids):
    """Convert the final SARSA state back to a standard schedule format."""
    final_schedule = {slot: {space_id: None for space_id in spaces_ids} for slot in slots}
    for slot_idx, slot_content in enumerate(sarsa_state_schedule):
        slot_name = slots[slot_idx]
        for item in slot_content:
            if isinstance(item, tuple) and len(item) == 2:
                room_id, activity_id = item
                if activity_id is not None:
                    final_schedule[slot_name][room_id] = activity_id
    return final_schedule

# Q-Learning Algorithm
def q_learning(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
              alpha=0.1, gamma=0.6, epsilon=0.1, n_episodes=1000):
    """Implement Q-Learning for timetable scheduling."""
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
        # Initialize state as empty schedule in SARSA format (list of lists)
        state = [[] for _ in range(len(slots))]
        total_reward = 0
        
        # For each slot in the timetable
        for slot_idx in range(len(slots)):
            # State representation for Q-table (convert complex state to string)
            state_key = str(state)
            
            # Get available actions
            actions = get_available_actions(state, slot_idx, activities_dict, spaces_dict, groups_dict, slots)
            
            # If no valid actions, continue to next slot
            if not actions:
                continue
            
            # Epsilon-greedy policy
            if random.uniform(0, 1) < epsilon:
                # Exploration: choose random action
                action = random.choice(actions)
            else:
                # Exploitation: choose best action based on Q-values
                if state_key not in q_table:
                    q_table[state_key] = {str(a): 0 for a in actions}
                
                # Find action with highest Q-value
                action = max(actions, key=lambda a: q_table[state_key].get(str(a), 0))
            
            # Apply the action
            next_state = apply_action(state, action, slot_idx)
            next_state_key = str(next_state)
            
            # Get intermediate reward for this step
            interim_schedule = sarsa_state_to_schedule(next_state, slots, spaces_ids)
            reward, hard_v, soft_v = evaluate_schedule(
                interim_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
            )
            
            # Update Q-value
            if state_key not in q_table:
                q_table[state_key] = {}
            if str(action) not in q_table[state_key]:
                q_table[state_key][str(action)] = 0
            
            # Compute max Q-value for next state
            if slot_idx < len(slots) - 1:
                next_actions = get_available_actions(next_state, slot_idx + 1, activities_dict, spaces_dict, groups_dict, slots)
                if next_actions and next_state_key not in q_table:
                    q_table[next_state_key] = {str(a): 0 for a in next_actions}
                
                max_next_q = max([q_table.get(next_state_key, {}).get(str(a), 0) for a in next_actions], default=0)
            else:
                max_next_q = 0
            
            # Q-Learning update
            q_table[state_key][str(action)] = q_table[state_key][str(action)] + alpha * (
                reward + gamma * max_next_q - q_table[state_key][str(action)]
            )
            
            # Update state and reward
            state = next_state
            total_reward += reward
        
        # Record episode results
        history['rewards'].append(total_reward)
        
        # Convert final state to schedule format
        final_schedule = convert_sarsa_state_to_schedule(state, activities_dict, slots, spaces_ids)
        final_reward, hard_v, soft_v = evaluate_schedule(
            final_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        
        history['hard_violations'].append(hard_v)
        history['soft_violations'].append(soft_v)
        
        # Update best solution if current is better
        if final_reward > history['best_reward']:
            history['best_reward'] = final_reward
            history['best_schedule'] = final_schedule
        
        # Print progress occasionally
        if episode % 100 == 0 or episode == n_episodes - 1:
            print(f"Episode {episode}: Reward = {final_reward}, Hard Violations = {hard_v}, Soft Violations = {soft_v}")
    
    return history, q_table

# DQN Implementation
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a simple neural network model for DQN."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential()
            model.add(Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model
        except ImportError:
            print("TensorFlow/Keras not available. Using placeholder model.")
            # Return a placeholder model that does nothing
            class DummyModel:
                def predict(self, state):
                    return np.random.random((1, action_size))
                def fit(self, state, target, epochs=1, verbose=0):
                    pass
            return DummyModel()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Act based on the current state using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train the model using random samples from memory."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Simplified state encoding for DQN (as DQN requires fixed-size state inputs)
def encode_state_for_dqn(schedule, slots, spaces_ids, activities_dict):
    """Encode the schedule state for DQN input.
    
    A simplified version that represents each (slot, space) pair as a one-hot encoding of the
    assigned activity (or none). This is not optimal but serves as a placeholder.
    """
    # For simplicity, just count number of activities scheduled and violations
    total_assigned = 0
    total_spaces = len(spaces_ids) * len(slots)
    
    for slot in schedule:
        for space_id, activity_id in schedule[slot].items():
            if activity_id is not None:
                total_assigned += 1
    
    # This is a very simplified encoding and would need to be enhanced for real use
    encoded = np.array([total_assigned / total_spaces])
    return encoded.reshape(1, 1)  # Reshape for Keras input

# Simplified DQN training for timetable scheduling
def dqn_scheduling(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
                  n_episodes=100, batch_size=32):
    """Implement DQN for timetable scheduling (simplified version)."""
    spaces_ids = list(spaces_dict.keys())
    
    # Define action space (simplified)
    # Each action represents assigning a specific activity to a specific space in a specific slot
    # For simplicity, we'll just use a placeholder size
    action_size = 10  # This would need to be properly calculated based on activities, spaces, slots
    
    # Define state space (simplified)
    state_size = 1  # Just using a single feature for demonstration
    
    # Initialize DQN agent
    agent = DQNAgent(state_size, action_size)
    
    # Episode history
    history = {
        'rewards': [],
        'best_reward': float('-inf'),
        'best_schedule': None
    }
    
    print("Note: This is a simplified placeholder DQN implementation.")
    print("For actual use, a more sophisticated state/action encoding would be needed.")
    
    for episode in range(n_episodes):
        # Initialize schedule
        schedule = initialize_empty_schedule(slots, spaces_ids)
        total_reward = 0
        
        # Simplified episode loop
        for slot_idx, slot in enumerate(slots):
            # Get current state
            state = encode_state_for_dqn(schedule, slots, spaces_ids, activities_dict)
            
            # Choose action
            action_idx = agent.act(state)
            
            # Apply action (simplified for demonstration)
            # In a real implementation, action_idx would map to (activity_id, space_id, slot)
            # Here we just randomly assign an activity if the action is valid
            
            # Simplified action handling
            if action_idx % 2 == 0:  # Just an arbitrary condition for demonstration
                # Randomly choose an activity and space
                available_activities = list(activities_dict.keys())
                if available_activities:  # If there are activities left to schedule
                    activity_id = random.choice(available_activities)
                    space_id = random.choice(spaces_ids)
                    
                    # Assign activity to schedule
                    schedule[slot][space_id] = activity_id
            
            # Calculate reward
            reward, hard_v, soft_v = evaluate_schedule(
                schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
            )
            
            # Get next state
            next_state = encode_state_for_dqn(schedule, slots, spaces_ids, activities_dict)
            
            # Determine if episode is done
            done = (slot_idx == len(slots) - 1)
            
            # Remember the experience
            agent.remember(state, action_idx, reward, next_state, done)
            
            # Train the model
            agent.replay(batch_size)
            
            # Update total reward
            total_reward += reward
        
        # Record episode results
        history['rewards'].append(total_reward)
        
        # Evaluate final schedule
        final_reward, hard_v, soft_v = evaluate_schedule(
            schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        
        # Update best solution if current is better
        if final_reward > history['best_reward']:
            history['best_reward'] = final_reward
            history['best_schedule'] = copy.deepcopy(schedule)
        
        # Print progress occasionally
        if episode % 10 == 0 or episode == n_episodes - 1:
            print(f"Episode {episode}: Reward = {final_reward}, Hard Violations = {hard_v}, Soft Violations = {soft_v}")
    
    print(f"Best reward found: {history['best_reward']}")
    return history

# Main function
def main():
    """Main function to run the RL timetable scheduling experiments."""
    print("Starting Reinforcement Learning for Timetable Scheduling")
    
    # Load data
    try:
        data_tuple = load_data()
        (
            activities_dict, groups_dict, spaces_dict, lecturers_dict,
            activities_list, groups_list, spaces_list, lecturers_list,
            activity_types, timeslots_list, days_list, periods_list, slots
        ) = data_tuple
        print(f"Successfully loaded data: {len(activities_dict)} activities, {len(spaces_dict)} spaces, {len(slots)} slots")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Set algorithm parameters
    alpha = 0.1  # Learning rate
    gamma = 0.6  # Discount factor
    epsilon = 0.1  # Exploration rate
    n_episodes = 500  # Number of episodes
    
    # Choose algorithm to run
    algorithm_choice = input("Choose algorithm (1: Q-Learning, 2: SARSA, 3: DQN): ").strip()
    
    if algorithm_choice == '1':
        # Run Q-Learning
        print("\nRunning Q-Learning algorithm...")
        history_q, q_table = q_learning(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=alpha, gamma=gamma, epsilon=epsilon, n_episodes=n_episodes
        )
        
        # Print results
        print(f"\nQ-Learning completed. Best reward: {history_q['best_reward']}")
        best_schedule = history_q['best_schedule']
        final_reward, hard_v, soft_v = evaluate_schedule(
            best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        print(f"Final hard constraint violations: {hard_v}")
        print(f"Final soft constraint violations: {soft_v}")
        
        # Save results
        with open('q_learning_results.pkl', 'wb') as f:
            pickle.dump(history_q, f)
        
    elif algorithm_choice == '2':
        # Run SARSA
        print("\nRunning SARSA algorithm...")
        history_sarsa, sarsa_q_table = sarsa(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=alpha, gamma=gamma, epsilon=epsilon, n_episodes=n_episodes
        )
        
        # Print results
        print(f"\nSARSA completed. Best reward: {history_sarsa['best_reward']}")
        best_schedule = history_sarsa['best_schedule']
        final_reward, hard_v, soft_v = evaluate_schedule(
            best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        print(f"Final hard constraint violations: {hard_v}")
        print(f"Final soft constraint violations: {soft_v}")
        
        # Save results
        with open('sarsa_results.pkl', 'wb') as f:
            pickle.dump(history_sarsa, f)
        
    elif algorithm_choice == '3':
        # Run DQN
        print("\nRunning DQN algorithm (simplified version)...")
        history_dqn = dqn_scheduling(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            n_episodes=100  # Fewer episodes for DQN as it's more computationally intensive
        )
        
        # Print results
        print(f"\nDQN completed. Best reward: {history_dqn['best_reward']}")
        best_schedule = history_dqn['best_schedule']
        final_reward, hard_v, soft_v = evaluate_schedule(
            best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        print(f"Final hard constraint violations: {hard_v}")
        print(f"Final soft constraint violations: {soft_v}")
        
        # Save results
        with open('dqn_results.pkl', 'wb') as f:
            pickle.dump(history_dqn, f)
    
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")

# Run the main function if script is executed directly
if __name__ == "__main__":
    main()

# SARSA Algorithm
def sarsa(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
         alpha=0.1, gamma=0.6, epsilon=0.1, n_episodes=1000):
    """Implement SARSA for timetable scheduling."""
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
        # Initialize state as empty schedule in SARSA format
        state = [[] for _ in range(len(slots))]
        total_reward = 0
        
        # Get initial action for first slot
        state_key = str(state)
        actions = get_available_actions(state, 0, activities_dict, spaces_dict, groups_dict, slots)
        
        # If no valid actions, skip episode
        if not actions:
            continue
        
        # Epsilon-greedy policy for initial action
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            if state_key not in q_table:
                q_table[state_key] = {str(a): 0 for a in actions}
            action = max(actions, key=lambda a: q_table[state_key].get(str(a), 0))
        
        for slot_idx in range(len(slots)):
            # Apply the action
            next_state = apply_action(state, action, slot_idx)
            next_state_key = str(next_state)
            
            # Get intermediate reward for this step
            interim_schedule = sarsa_state_to_schedule(next_state, slots, spaces_ids)
            reward, hard_v, soft_v = evaluate_schedule(
                interim_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
            )
            
            # Get next action
            if slot_idx < len(slots) - 1:
                next_actions = get_available_actions(next_state, slot_idx + 1, activities_dict, spaces_dict, groups_dict, slots)
                
                if not next_actions:
                    next_action = None
                else:
                    # Epsilon-greedy policy for next action
                    if random.uniform(0, 1) < epsilon:
                        next_action = random.choice(next_actions)
                    else:
                        if next_state_key not in q_table:
                            q_table[next_state_key] = {str(a): 0 for a in next_actions}
                        next_action = max(next_actions, key=lambda a: q_table[next_state_key].get(str(a), 0))
            else:
                next_action = None
            
            # Update Q-value
            if state_key not in q_table:
                q_table[state_key] = {}
            if str(action) not in q_table[state_key]:
                q_table[state_key][str(action)] = 0
            
            # SARSA update formula
            if next_action is not None:
                if next_state_key not in q_table:
                    q_table[next_state_key] = {}
                if str(next_action) not in q_table[next_state_key]:
                    q_table[next_state_key][str(next_action)] = 0
                
                q_table[state_key][str(action)] = q_table[state_key][str(action)] + alpha * (
                    reward + gamma * q_table[next_state_key][str(next_action)] - q_table[state_key][str(action)]
                )
            else:
                q_table[state_key][str(action)] = q_table[state_key][str(action)] + alpha * (
                    reward - q_table[state_key][str(action)]
                )
            
            # Update state, action, and reward for next iteration
            state = next_state
            state_key = next_state_key
            action = next_action
            total_reward += reward
            
            if next_action is None:
                break
        
        # Record episode results
        history['rewards'].append(total_reward)
        
        # Convert final state to schedule format
        final_schedule = convert_sarsa_state_to_schedule(state, activities_dict, slots, spaces_ids)
        final_reward, hard_v, soft_v = evaluate_schedule(
            final_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        
        history['hard_violations'].append(hard_v)
        history['soft_violations'].append(soft_v)
        
        # Update best solution if current is better
        if final_reward > history['best_reward']:
            history['best_reward'] = final_reward
            history['best_schedule'] = final_schedule
        
        # Print progress occasionally
        if episode % 100 == 0 or episode == n_episodes - 1:
            print(f"Episode {episode}: Reward = {final_reward}, Hard Violations = {hard_v}, Soft Violations = {soft_v}")
    
    return history, q_table

# Main function
def main():
    """Main function to run the RL timetable scheduling experiments."""
    print("Starting Reinforcement Learning for Timetable Scheduling")
    
    # Load data
    try:
        data_tuple = load_data()
        (
            activities_dict, groups_dict, spaces_dict, lecturers_dict,
            activities_list, groups_list, spaces_list, lecturers_list,
            activity_types, timeslots_list, days_list, periods_list, slots
        ) = data_tuple
        print(f"Successfully loaded data: {len(activities_dict)} activities, {len(spaces_dict)} spaces, {len(slots)} slots")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Set algorithm parameters
    alpha = 0.1  # Learning rate
    gamma = 0.6  # Discount factor
    epsilon = 0.1  # Exploration rate
    n_episodes = 500  # Number of episodes
    
    # Choose algorithm to run
    algorithm_choice = input("Choose algorithm (1: Q-Learning, 2: SARSA, 3: DQN): ").strip()
    
    if algorithm_choice == '1':
        # Run Q-Learning
        print("\nRunning Q-Learning algorithm...")
        history_q, q_table = q_learning(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=alpha, gamma=gamma, epsilon=epsilon, n_episodes=n_episodes
        )
        
        # Print results
        print(f"\nQ-Learning completed. Best reward: {history_q['best_reward']}")
        best_schedule = history_q['best_schedule']
        final_reward, hard_v, soft_v = evaluate_schedule(
            best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        print(f"Final hard constraint violations: {hard_v}")
        print(f"Final soft constraint violations: {soft_v}")
        
        # Save results
        with open('q_learning_results.pkl', 'wb') as f:
            pickle.dump(history_q, f)
        
    elif algorithm_choice == '2':
        # Run SARSA
        print("\nRunning SARSA algorithm...")
        history_sarsa, sarsa_q_table = sarsa(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=alpha, gamma=gamma, epsilon=epsilon, n_episodes=n_episodes
        )
        
        # Print results
        print(f"\nSARSA completed. Best reward: {history_sarsa['best_reward']}")
        best_schedule = history_sarsa['best_schedule']
        final_reward, hard_v, soft_v = evaluate_schedule(
            best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        print(f"Final hard constraint violations: {hard_v}")
        print(f"Final soft constraint violations: {soft_v}")
        
        # Save results
        with open('sarsa_results.pkl', 'wb') as f:
            pickle.dump(history_sarsa, f)
        
    elif algorithm_choice == '3':
        # Run DQN
        print("\nRunning DQN algorithm (simplified version)...")
        history_dqn = dqn_scheduling(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            n_episodes=100  # Fewer episodes for DQN as it's more computationally intensive
        )
        
        # Print results
        print(f"\nDQN completed. Best reward: {history_dqn['best_reward']}")
        best_schedule = history_dqn['best_schedule']
        final_reward, hard_v, soft_v = evaluate_schedule(
            best_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
        )
        print(f"Final hard constraint violations: {hard_v}")
        print(f"Final soft constraint violations: {soft_v}")
        
        # Save results
        with open('dqn_results.pkl', 'wb') as f:
            pickle.dump(history_dqn, f)
    
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")

# Run the main function if script is executed directly
if __name__ == "__main__":
    main()