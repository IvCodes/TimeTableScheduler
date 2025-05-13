"""
Minimal test script to verify both GA and RL implementations work with shared components.
Runs with minimal iterations to quickly identify any issues.
"""

import os
import sys
import time

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import common components
from common.data_loader import load_data, Activity, Group, Space, Lecturer
from common.evaluator import evaluate_schedule, multi_objective_evaluator

dataset_path = "Dataset/sliit_computing_dataset_balanced.json"  # Use the balanced dataset for testing

# Add __hash__ method to Activity class and fix attribute compatibility
def make_dataclasses_hashable():
    """Add __hash__ method to dataclasses to make them hashable."""
    def hash_method(self):
        return hash(self.id)
    
    # Add hash method to make classes hashable
    Activity.__hash__ = hash_method
    Group.__hash__ = hash_method
    Space.__hash__ = hash_method
    Lecturer.__hash__ = hash_method
    
    # Add compatibility for teacher_id / lecturer_id differences
    def activity_teacher_id_getter(self):
        return self.lecturer_id
    
    def activity_teacher_id_setter(self, value):
        self.lecturer_id = value
    
    # Add teacher_id property to Activity class for GA compatibility
    Activity.teacher_id = property(activity_teacher_id_getter, activity_teacher_id_setter)

# Functions to run minimal GA test
def run_minimal_ga_test():
    print("\n===== Running Minimal GA Test =====")
    
    # Import GA modules only when needed
    sys.path.append(os.path.join(os.path.dirname(__file__), "1. Genetic_EAs"))
    from Scheduler_fix_Ga import nsga2, spea2, moead
    
    # Load data
    try:
        print("Loading data for GA test...")
        # Get the correct order from load_data
        data = load_data()
        activities_dict = data[0]  # First element is activities_dict
        groups_dict = data[1]      # Second element is groups_dict
        spaces_dict = data[2]      # Third element is spaces_dict
        lecturers_dict = data[3]   # Fourth element is lecturers_dict
        slots = data[12]           # Last element (13th) is slots
        
        print(f"Loaded: {len(activities_dict)} activities, {len(spaces_dict)} spaces, {len(groups_dict)} groups, {len(lecturers_dict)} lecturers")
    except Exception as e:
        print(f"Error loading data for GA: {str(e)}")
        return False
    
    # Minimal parameters for quick testing
    pop_size = 100
    generations = 50
    
    # Try running NSGA-II with minimal iterations
    try:
        print("\nTesting NSGA-II with minimal parameters...")
        start_time = time.time()
        population, fitness = nsga2(
            pop_size=pop_size, 
            generations=generations, 
            activities_dict=activities_dict,
            groups_dict=groups_dict, 
            lecturers_dict=lecturers_dict, 
            spaces_dict=spaces_dict, 
            slots=slots
        )
        duration = time.time() - start_time
        
        # Verify results
        if population and fitness:
            print(f"NSGA-II test successful! Duration: {duration:.2f}s")
            print(f"Population size: {len(population)}")
            
            # Find the best individual based on various metrics
            best_idx_soft = min(range(len(fitness)), key=lambda i: fitness[i][4])
            best_idx_unassigned = min(range(len(fitness)), key=lambda i: fitness[i][3])
            best_idx_conflicts = min(range(len(fitness)), key=lambda i: fitness[i][0] + fitness[i][1] + fitness[i][2])
            
            print(f"\nBest fitness by soft score (index 4):")
            best_fitness_soft = fitness[best_idx_soft]
            print(f"  - Professor conflicts: {best_fitness_soft[0]}")
            print(f"  - Group conflicts: {best_fitness_soft[1]}")
            print(f"  - Room size conflicts: {best_fitness_soft[2]}")
            print(f"  - Unassigned activities: {best_fitness_soft[3]}")
            print(f"  - Soft score (inverted, lower is better): {best_fitness_soft[4]:.4f}")
            
            print(f"\nBest fitness by minimizing unassigned activities (index 3):")
            best_fitness_unassigned = fitness[best_idx_unassigned]
            print(f"  - Professor conflicts: {best_fitness_unassigned[0]}")
            print(f"  - Group conflicts: {best_fitness_unassigned[1]}")
            print(f"  - Room size conflicts: {best_fitness_unassigned[2]}")
            print(f"  - Unassigned activities: {best_fitness_unassigned[3]}")
            print(f"  - Soft score (inverted, lower is better): {best_fitness_unassigned[4]:.4f}")
            
            print(f"\nBest fitness by minimizing conflicts (index 0+1+2):")
            best_fitness_conflicts = fitness[best_idx_conflicts]
            print(f"  - Professor conflicts: {best_fitness_conflicts[0]}")
            print(f"  - Group conflicts: {best_fitness_conflicts[1]}")
            print(f"  - Room size conflicts: {best_fitness_conflicts[2]}")
            print(f"  - Unassigned activities: {best_fitness_conflicts[3]}")
            print(f"  - Soft score (inverted, lower is better): {best_fitness_conflicts[4]:.4f}")
            
            # Analyze activity scheduling
            best_schedule = population[best_idx_soft]
            total_activities = len(activities_dict)
            scheduled_count = 0
            for slot in best_schedule.values():
                for room, activity in slot.items():
                    if activity is not None:
                        scheduled_count += 1
            
            print(f"\nActivity scheduling analysis:")
            print(f"  - Activities in dataset: {total_activities}")
            print(f"  - Activities scheduled: {scheduled_count}")
            print(f"  - Scheduling rate: {scheduled_count/total_activities*100:.1f}%")
            
            # Check if we're getting perfect soft score but poor scheduling
            if best_fitness_soft[4] == 0.0 and best_fitness_soft[3] > 0:
                print("\nPOSSIBLE ISSUE DETECTED: Perfect soft score (0.0000) with unassigned activities.")
                print("This suggests the algorithm may be prioritizing soft constraints over scheduling activities.")
                print("Consider adjusting the fitness function to better balance these objectives.")
            
            return True
        else:
            print("NSGA-II test failed - empty results.")
            return False
            
    except Exception as e:
        print(f"Error running NSGA-II: {str(e)}")
        return False

# Functions to run minimal RL test
def run_minimal_rl_test():
    print("\n===== Running Minimal RL Test =====")
    
    # Import RL modules
    sys.path.append(os.path.join(os.path.dirname(__file__), "3. RL", "RL_Script"))
    
    # Use the new unified script
    try:
        from RL_Script_Unified import q_learning, sarsa, dqn_scheduling
    except ImportError:
        print("Unable to import from RL_Script_Unified.py. Falling back to original RL_Scirpt.py...")
        try:
            from RL_Scirpt import q_learning, sarsa, dqn_scheduling
        except ImportError:
            print("Error importing RL algorithms. Please check that RL scripts exist.")
            return False
    
    # Load data
    try:
        print("Loading data for RL test...")
        data_tuple = load_data()
        (
            activities_dict, groups_dict, spaces_dict, lecturers_dict,
            activities_list, groups_list, spaces_list, lecturers_list,
            activity_types, timeslots_list, days_list, periods_list, slots
        ) = data_tuple
        print(f"Loaded: {len(activities_dict)} activities, {len(spaces_dict)} spaces")
    except Exception as e:
        print(f"Error loading data for RL: {str(e)}")
        return False
    
    # Minimal parameters for quick testing
    n_episodes = 50
    alpha = 0.5
    gamma = 0.6
    epsilon = 0.5
    
    # Try running Q-Learning with minimal iterations
    try:
        print("\nTesting Q-Learning with minimal parameters...")
        start_time = time.time()
        history_q, _ = q_learning(
            activities_dict=activities_dict,
            groups_dict=groups_dict,
            spaces_dict=spaces_dict,
            lecturers_dict=lecturers_dict,
            slots=slots,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            n_episodes=n_episodes
        )
        duration = time.time() - start_time
        
        # Verify results
        if history_q and 'best_reward' in history_q:
            print(f"Q-Learning test successful! Duration: {duration:.2f}s")
            print(f"Best reward: {history_q['best_reward']:.4f}")
            
            # Print the best schedule summary
            if history_q['best_schedule']:
                _, hard_v, soft_v = evaluate_schedule(
                    history_q['best_schedule'], 
                    activities_dict, 
                    groups_dict, 
                    spaces_dict, 
                    lecturers_dict, 
                    slots
                )
                print(f"Best schedule - Hard violations: {hard_v}, Soft violations: {soft_v:.4f}")
            return True
        else:
            print("Q-Learning test failed - invalid results.")
            return False
            
    except Exception as e:
        print(f"Error running Q-Learning: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Main function
def main():
    print("===== Starting Minimal Test =====")
    
    # Make dataclasses hashable
    make_dataclasses_hashable()
    print("This script runs both GA and RL implementations with minimal iterations")
    print("to verify they work correctly with shared components.")
    
    # Test GA implementation
    ga_success = run_minimal_ga_test()
    
    # Test RL implementation
    rl_success = run_minimal_rl_test()
    
    # Report overall results
    print("\n===== Test Summary =====")
    print(f"GA implementation: {'Success' if ga_success else 'Failed'}")
    print(f"RL implementation: {'Success' if rl_success else 'Failed'}")
    
    if ga_success and rl_success:
        print("\nBoth implementations passed the minimal test!")
        print("You can now proceed with more comprehensive tests or move to a Fabric environment.")
    else:
        print("\nOne or both implementations failed. Please check the errors above.")

if __name__ == "__main__":
    main()
