"""
TimeTable Scheduler - Azure Fabric Notebook Script (Part 4: RL DQN Implementation)
This script is formatted for easy conversion to a Jupyter notebook.
Each cell is marked with '# CELL: {description}' comments.
"""

# CELL: RL Helper Classes for DQN
class ExperienceReplay:
    """
    Experience Replay buffer for DQN algorithm.
    Stores transitions (state, action, reward, next_state, done) for training.
    """
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer with given capacity.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple[List]: Batch of transitions as separate lists
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            int: Current size of buffer
        """
        return len(self.buffer)

class StateEncoder:
    """
    Encodes RL states into a fixed-size vector for neural network input.
    """
    def __init__(self, activities_dict, spaces_dict, slots):
        """
        Initialize state encoder.
        
        Args:
            activities_dict: Dictionary of activities
            spaces_dict: Dictionary of spaces
            slots: List of time slots
        """
        self.activities = list(activities_dict.keys())
        self.spaces = list(spaces_dict.keys())
        self.slots = slots
        
        # Define encoding dimensions
        self.num_activities = len(self.activities)
        self.num_spaces = len(self.spaces)
        self.num_slots = len(self.slots)
        
        # Total size of encoded state
        self.encoded_size = self.num_slots * self.num_activities * self.num_spaces
    
    def encode(self, state):
        """
        Encode a state into a fixed-size vector.
        
        Args:
            state: State to encode
            
        Returns:
            np.ndarray: Encoded state vector
        """
        # Initialize empty vector
        encoded = np.zeros((self.num_slots, self.num_activities, self.num_spaces))
        
        # Fill in assigned activities
        for slot_idx, slot_activities in enumerate(state):
            for activity_data in slot_activities:
                if activity_data:
                    activity_idx, space_idx = activity_data
                    if activity_idx < self.num_activities and space_idx < self.num_spaces:
                        encoded[slot_idx, activity_idx, space_idx] = 1
        
        # Flatten to 1D vector
        return encoded.flatten()
    
    def encode_action(self, action):
        """
        Encode an action into an index for the Q-network.
        
        Args:
            action: Action to encode (activity_idx, space_idx)
            
        Returns:
            int: Action index
        """
        activity_idx, space_idx = action
        return activity_idx * self.num_spaces + space_idx
    
    def decode_action(self, action_idx):
        """
        Decode an action index into an action tuple.
        
        Args:
            action_idx: Action index to decode
            
        Returns:
            Tuple[int, int]: Action as (activity_idx, space_idx)
        """
        activity_idx = action_idx // self.num_spaces
        space_idx = action_idx % self.num_spaces
        return (activity_idx, space_idx)
    
    def get_action_space_size(self):
        """
        Get the size of the action space.
        
        Returns:
            int: Size of action space
        """
        return self.num_activities * self.num_spaces

# CELL: DQN Implementation
def create_dqn_model(state_size, action_size):
    """
    Create a Deep Q-Network model.
    
    Args:
        state_size: Size of state space
        action_size: Size of action space
        
    Returns:
        tf.keras.Model: DQN model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for DQN implementation")
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=(state_size,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def dqn_scheduling(activities_dict, groups_dict, spaces_dict, lecturers_dict, slots, 
                  gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                  batch_size=32, n_episodes=1000):
    """
    Implementation of Deep Q-Network algorithm for timetable scheduling.
    
    Args:
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of student groups
        spaces_dict: Dictionary of spaces
        lecturers_dict: Dictionary of lecturers
        slots: List of time slots
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Decay rate for exploration
        batch_size: Batch size for training
        n_episodes: Number of episodes
        
    Returns:
        Dict: History of training
    """
    print(f"\nRunning DQN algorithm (gamma={gamma}, epsilon={epsilon}->{epsilon_min}, episodes={n_episodes})...")
    
    # Check if TensorFlow is available
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - using Q-Learning as fallback")
        return q_learning(
            activities_dict, groups_dict, spaces_dict, lecturers_dict, slots,
            alpha=0.1, gamma=gamma, epsilon=0.1, n_episodes=n_episodes
        )
    
    spaces_ids = list(spaces_dict.keys())
    
    # Initialize state encoder
    encoder = StateEncoder(activities_dict, spaces_dict, slots)
    
    # Create DQN model
    state_size = encoder.encoded_size
    action_size = encoder.get_action_space_size()
    model = create_dqn_model(state_size, action_size)
    
    # Create target network
    target_model = create_dqn_model(state_size, action_size)
    target_model.set_weights(model.get_weights())
    
    # Initialize replay buffer
    memory = ExperienceReplay(capacity=10000)
    
    # Episode history for analysis
    history = {
        'rewards': [],
        'best_reward': float('-inf'),
        'best_schedule': None,
        'hard_violations': [],
        'soft_violations': [],
        'epsilon': []
    }
    
    # Training loop
    for episode in range(n_episodes):
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes}, Epsilon: {epsilon:.4f}", end='\r')
        
        # Initialize state as empty schedule
        state = [[] for _ in range(len(slots))]
        total_reward = 0
        
        for slot_idx in range(len(slots)):
            # Get available actions
            actions = get_available_actions(state, slot_idx, activities_dict, spaces_dict, groups_dict, slots)
            
            # If no valid actions, skip slot
            if not actions:
                continue
            
            # Encode state
            state_encoded = encoder.encode(state)
            
            # Epsilon-greedy policy
            if random.uniform(0, 1) < epsilon:
                # Exploration: choose random action
                action = random.choice(actions)
            else:
                # Exploitation: choose best action
                encoded_actions = [encoder.encode_action(a) for a in actions]
                q_values = model.predict(np.array([state_encoded]), verbose=0)[0]
                
                # Filter valid actions
                valid_q_values = [(q_values[a], a) for a in encoded_actions]
                _, action_idx = max(valid_q_values, key=lambda x: x[0])
                action = encoder.decode_action(action_idx)
            
            # Apply action to get next state
            next_state = apply_action(state, action, slot_idx)
            
            # Get intermediate reward for this step
            interim_schedule = sarsa_state_to_schedule(next_state, slots, spaces_ids)
            reward, hard_v, soft_v = evaluate_schedule(
                interim_schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots
            )
            
            # Check if episode is done
            done = (slot_idx == len(slots) - 1)
            
            # Encode action
            action_encoded = encoder.encode_action(action)
            
            # Encode next state
            next_state_encoded = encoder.encode(next_state)
            
            # Store transition in replay buffer
            memory.push(state_encoded, action_encoded, reward, next_state_encoded, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Train the model
            if len(memory) > batch_size:
                # Sample batch from replay buffer
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                
                # Convert to numpy arrays
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)
                
                # Calculate target Q-values
                targets = model.predict(states, verbose=0)
                next_q_values = target_model.predict(next_states, verbose=0)
                
                # Update target Q-values
                for i in range(batch_size):
                    if dones[i]:
                        targets[i, actions[i]] = rewards[i]
                    else:
                        targets[i, actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])
                
                # Train the model
                model.fit(states, targets, epochs=1, verbose=0)
        
        # Update target network
        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())
        
        # Store episode results
        history['rewards'].append(total_reward)
        history['epsilon'].append(epsilon)
        
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
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Report progress
        if episode % max(1, n_episodes // 10) == 0 or episode == n_episodes - 1:
            print(f"Episode {episode}: Reward = {total_reward}, Hard Violations = {hard_violations}, Soft Violations = {soft_violations}")
    
    print(f"DQN completed. Best reward: {history['best_reward']}")
    return history, model
