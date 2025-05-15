"""
TimeTable Scheduler - Azure Fabric Notebook Script (Part 4: RL SARSA Implementation)
This script is formatted for easy conversion to a Jupyter notebook.
Each cell is marked with '# CELL: {description}' comments.
"""

# CELL: RL SARSA Implementation
def sarsa(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
         alpha=0.1, gamma=0.6, epsilon=0.1, n_episodes=1000):
    """
    Implementation of SARSA algorithm for timetable scheduling.
    
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
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes}", end='\r')
        
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
            
            # Add state to Q-table if not present
            if state_key not in q_table:
                q_table[state_key] = {}
            
            # Add action to Q-table if not present
            if str(action) not in q_table[state_key]:
                q_table[state_key][str(action)] = 0
            
            # Update Q-value using SARSA update rule
            if next_action is not None:
                if next_state_key not in q_table:
                    q_table[next_state_key] = {}
                
                if str(next_action) not in q_table[next_state_key]:
                    q_table[next_state_key][str(next_action)] = 0
                
                q_table[state_key][str(action)] = (1 - alpha) * q_table[state_key][str(action)] + \
                                                alpha * (reward + gamma * q_table[next_state_key][str(next_action)])
            else:
                q_table[state_key][str(action)] = (1 - alpha) * q_table[state_key][str(action)] + \
                                                alpha * reward
            
            # Update state and action
            state = next_state
            state_key = next_state_key
            action = next_action
            
            # If no next action, break
            if action is None:
                break
            
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
    
    print(f"SARSA completed. Best reward: {history['best_reward']}")
    return history, q_table
