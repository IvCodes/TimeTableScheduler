#!/usr/bin/env python3
"""
RL_Script_Unified.py - Unified implementation of RL algorithms for timetable scheduling
Uses common data loading and evaluation functions to eliminate redundancy with GA implementations
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import copy
import warnings
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Any, Optional

# Import common modules
import sys
sys.path.append('../..')  # Add parent directory to path
from common.data_loader import load_data, initialize_empty_schedule, is_space_suitable
from common.evaluator import evaluate_schedule

# Suppress warnings
warnings.filterwarnings('ignore')

# RL State Manipulation Functions
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
            # Make sure item is a tuple to avoid type errors
            if isinstance(item, tuple) and len(item) == 2:
                room_id, activity_id = item
                if activity_id is not None:
                    final_schedule[slot_name][room_id] = activity_id
    return final_schedule

# Q-Learning Algorithm
def q_learning(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
              alpha=0.1, gamma=0.6, epsilon=0.1, n_episodes=1000):
    """Implement Q-Learning for timetable scheduling."""
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
    
    print(f"Q-Learning completed. Best reward: {history['best_reward']}")
    return history, q_table

# SARSA Algorithm
def sarsa(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
         alpha=0.1, gamma=0.6, epsilon=0.1, n_episodes=1000):
    """Implement SARSA for timetable scheduling."""
    print(f"\nRunning SARSA algorithm (alpha={alpha}, gamma={gamma}, epsilon={epsilon}, episodes={n_episodes})...")
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
    
    print(f"SARSA completed. Best reward: {history['best_reward']}")
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
                    return np.random.random((1, self.action_size))
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
    """Encode the schedule state for DQN input."""
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
    print(f"\nRunning DQN algorithm (episodes={n_episodes}, batch_size={batch_size})...")
    print("Note: This is a simplified placeholder DQN implementation.")
    print("For actual use, a more sophisticated state/action encoding would be needed.")
    
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
    
    print(f"DQN completed. Best reward: {history['best_reward']}")
    return history

# Plot history function
def plot_training_history(history, algorithm_name, output_dir='results'):
    """Plot the training history and save the figures."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot rewards over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(history['rewards'])
    plt.title(f'{algorithm_name} - Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{algorithm_name.lower()}_rewards.png'))
    plt.close()
    
    # Plot hard violations if available
    if 'hard_violations' in history:
        plt.figure(figsize=(12, 6))
        plt.plot(history['hard_violations'])
        plt.title(f'{algorithm_name} - Hard Constraint Violations')
        plt.xlabel('Episode')
        plt.ylabel('Violations')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{algorithm_name.lower()}_hard_violations.png'))
        plt.close()
    
    # Plot soft violations if available
    if 'soft_violations' in history:
        plt.figure(figsize=(12, 6))
        plt.plot(history['soft_violations'])
        plt.title(f'{algorithm_name} - Soft Constraint Violations')
        plt.xlabel('Episode')
        plt.ylabel('Violations')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{algorithm_name.lower()}_soft_violations.png'))
        plt.close()

# Main function to run all algorithms
def run_algorithms(algorithm=None, n_episodes=500, alpha=0.1, gamma=0.6, epsilon=0.1):
    """Run the RL timetable scheduling algorithms."""
    print("Starting Reinforcement Learning for Timetable Scheduling")
    
    # Set default output directory
    output_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # If algorithm is None or 'all', run all algorithms
    if algorithm is None or algorithm.lower() == 'all':
        # Run Q-Learning
        history_q, q_table = q_learning(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=alpha, gamma=gamma, epsilon=epsilon, n_episodes=n_episodes
        )
        
        # Plot and save Q-Learning results
        plot_training_history(history_q, 'Q-Learning', output_dir)
        with open(os.path.join(output_dir, 'q_learning_results.pkl'), 'wb') as f:
            pickle.dump(history_q, f)
        
        # Run SARSA
        history_sarsa, sarsa_q_table = sarsa(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=alpha, gamma=gamma, epsilon=epsilon, n_episodes=n_episodes
        )
        
        # Plot and save SARSA results
        plot_training_history(history_sarsa, 'SARSA', output_dir)
        with open(os.path.join(output_dir, 'sarsa_results.pkl'), 'wb') as f:
            pickle.dump(history_sarsa, f)
        
        # Run DQN (with fewer episodes as it's more computationally intensive)
        history_dqn = dqn_scheduling(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            n_episodes=min(100, n_episodes)  # Limit DQN episodes
        )
        
        # Plot and save DQN results
        plot_training_history(history_dqn, 'DQN', output_dir)
        with open(os.path.join(output_dir, 'dqn_results.pkl'), 'wb') as f:
            pickle.dump(history_dqn, f)
        
        print("\nAll algorithms completed. Results saved in the 'results' directory.")
        return history_q, history_sarsa, history_dqn
    
    # Run specific algorithm
    elif algorithm.lower() == 'qlearning':
        history_q, q_table = q_learning(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=alpha, gamma=gamma, epsilon=epsilon, n_episodes=n_episodes
        )
        
        # Plot and save Q-Learning results
        plot_training_history(history_q, 'Q-Learning', output_dir)
        with open(os.path.join(output_dir, 'q_learning_results.pkl'), 'wb') as f:
            pickle.dump(history_q, f)
        
        return history_q
    
    elif algorithm.lower() == 'sarsa':
        history_sarsa, sarsa_q_table = sarsa(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=alpha, gamma=gamma, epsilon=epsilon, n_episodes=n_episodes
        )
        
        # Plot and save SARSA results
        plot_training_history(history_sarsa, 'SARSA', output_dir)
        with open(os.path.join(output_dir, 'sarsa_results.pkl'), 'wb') as f:
            pickle.dump(history_sarsa, f)
        
        return history_sarsa
    
    elif algorithm.lower() == 'dqn':
        history_dqn = dqn_scheduling(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            n_episodes=min(100, n_episodes)  # Limit DQN episodes
        )
        
        # Plot and save DQN results
        plot_training_history(history_dqn, 'DQN', output_dir)
        with open(os.path.join(output_dir, 'dqn_results.pkl'), 'wb') as f:
            pickle.dump(history_dqn, f)
        
        return history_dqn
    
    else:
        print(f"Unknown algorithm: {algorithm}. Please use 'qlearning', 'sarsa', 'dqn', or 'all'.")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run RL algorithms for timetable scheduling')
    parser.add_argument('--algorithm', type=str, default='all',
                        help="Algorithm to run ('qlearning', 'sarsa', 'dqn', or 'all')")
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes to run')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.6,
                        help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Exploration rate')
    
    args = parser.parse_args()
    
    # Run the algorithms with the specified parameters
    run_algorithms(
        algorithm=args.algorithm,
        n_episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon
    )
