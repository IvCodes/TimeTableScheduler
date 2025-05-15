import json
import os
import random
import copy
import matplotlib.pyplot as plt
from collections import Counter

def balance_dataset(input_path, output_path, target_distribution=None):
    """
    Balance the dataset by adjusting the activity types to match a more realistic distribution.
    
    Args:
        input_path: Path to the original dataset
        output_path: Path to save the balanced dataset
        target_distribution: Dictionary with target percentages for each activity type
                            e.g., {'Lecture': 40, 'Tutorial': 40, 'Lab': 20}
    """
    # Default target distribution if none provided
    if target_distribution is None:
        target_distribution = {
            'Lecture': 40,  # 40% lectures
            'Tutorial': 40, # 40% tutorials
            'Lab': 20       # 20% labs
        }
    
    # Load the original dataset
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    activities = data.get('activities', [])
    total_activities = len(activities)
    
    # Classify existing activities
    activity_types = []
    for activity in activities:
        name = activity.get('name', '').lower()
        description = activity.get('description', '').lower()
        duration = activity.get('duration', 0)
        
        if 'tutorial' in name or 'tutorial' in description:
            activity_types.append('Tutorial')
        elif 'lab' in name or 'lab' in description or 'practical' in name:
            activity_types.append('Lab')
        elif 'lecture' in name or 'lecture' in description:
            activity_types.append('Lecture')
        elif duration <= 60:
            activity_types.append('Tutorial')
        elif duration >= 120:
            activity_types.append('Lab')
        else:
            activity_types.append('Lecture')
    
    # Count current distribution
    current_counts = Counter(activity_types)
    print(f"Current distribution:")
    for activity_type, count in current_counts.items():
        percentage = (count / total_activities) * 100
        print(f"  {activity_type}: {count} ({percentage:.1f}%)")
    
    # Calculate target counts
    target_counts = {}
    for activity_type, percentage in target_distribution.items():
        target_counts[activity_type] = int(round(total_activities * percentage / 100))
    
    # Adjust for rounding errors to maintain the same total
    adjustment = total_activities - sum(target_counts.values())
    if adjustment != 0:
        # Add or remove from the largest category
        largest_type = max(target_counts, key=target_counts.get)
        target_counts[largest_type] += adjustment
    
    print(f"\nTarget distribution:")
    for activity_type, count in target_counts.items():
        percentage = (count / total_activities) * 100
        print(f"  {activity_type}: {count} ({percentage:.1f}%)")
    
    # Create a deep copy of the dataset to modify
    balanced_data = copy.deepcopy(data)
    balanced_activities = balanced_data['activities']
    
    # Identify activities to convert for each type
    conversions = {}
    for activity_type in target_counts:
        current = current_counts.get(activity_type, 0)
        target = target_counts.get(activity_type, 0)
        conversions[activity_type] = target - current
    
    print("\nRequired conversions:")
    for activity_type, change in conversions.items():
        print(f"  {activity_type}: {'+' if change > 0 else ''}{change}")
    
    # Group activities by their current type
    activities_by_type = {t: [] for t in current_counts}
    for i, activity_type in enumerate(activity_types):
        activities_by_type[activity_type].append(i)
    
    # Perform conversions
    # First, handle conversions from Tutorial to other types (since we have excess tutorials)
    from_type = 'Tutorial'  # We know we need to convert from tutorials
    
    # Convert tutorials to lectures
    if conversions['Lecture'] > 0:
        convert_count = min(abs(conversions[from_type]), conversions['Lecture'])
        indices_to_convert = random.sample(activities_by_type[from_type], convert_count)
        
        for idx in indices_to_convert:
            # Update the activity
            activity = balanced_activities[idx]
            
            # Update to lecture
            prefix = "Lecture: "
            duration = random.randint(90, 110)  # Typical lecture duration
            
            # Update the activity name
            old_name = activity['name']
            if ': ' in old_name:
                activity['name'] = prefix + old_name.split(': ')[-1]
            else:
                activity['name'] = prefix + old_name
            
            # Update duration
            activity['duration'] = duration
            
            # Update activity_types for verification later
            activity_types[idx] = 'Lecture'
            
            # Remove from tutorial list
            activities_by_type[from_type].remove(idx)
            # Add to lecture list
            activities_by_type.setdefault('Lecture', []).append(idx)
        
        # Update conversion counts
        conversions[from_type] += convert_count
        conversions['Lecture'] -= convert_count
    
    # Convert tutorials to labs
    if conversions['Lab'] > 0:
        # Recalculate how many tutorials are left
        remaining_tutorials = len(activities_by_type[from_type])
        convert_count = min(remaining_tutorials, conversions['Lab'])
        
        if convert_count > 0:
            indices_to_convert = random.sample(activities_by_type[from_type], convert_count)
            
            for idx in indices_to_convert:
                # Update the activity
                activity = balanced_activities[idx]
                
                # Update to lab
                prefix = "Lab: "
                duration = random.randint(120, 180)  # Typical lab duration
                
                # Update the activity name
                old_name = activity['name']
                if ': ' in old_name:
                    activity['name'] = prefix + old_name.split(': ')[-1]
                else:
                    activity['name'] = prefix + old_name
                
                # Update duration
                activity['duration'] = duration
                
                # Update activity_types for verification later
                activity_types[idx] = 'Lab'
                
                # Remove from tutorial list
                activities_by_type[from_type].remove(idx)
                # Add to lab list
                activities_by_type.setdefault('Lab', []).append(idx)
            
            # Update conversion counts
            conversions[from_type] += convert_count
            conversions['Lab'] -= convert_count
    
    # Save the balanced dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(balanced_data, f, indent=2)
    
    print(f"\nBalanced dataset saved to {output_path}")
    
    # Verify the new distribution - use the activity_types list that we've been updating
    # during the conversion process rather than re-inferring the types
    new_activity_types = activity_types
    
    new_counts = Counter(new_activity_types)
    print(f"\nFinal distribution:")
    for activity_type, count in new_counts.items():
        percentage = (count / total_activities) * 100
        print(f"  {activity_type}: {count} ({percentage:.1f}%)")
    
    # Visualize the before and after distribution
    visualize_distribution_change(current_counts, new_counts, 
                                 os.path.basename(input_path), 
                                 os.path.basename(output_path))
    
    return balanced_data

def visualize_distribution_change(before_counts, after_counts, input_name, output_name):
    """Visualize the change in distribution before and after balancing."""
    activity_types = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))
    
    before_values = [before_counts.get(t, 0) for t in activity_types]
    after_values = [after_counts.get(t, 0) for t in activity_types]
    
    x = range(len(activity_types))
    width = 0.35
    
    plt.figure(figsize=(12, 7))
    
    # Plot raw counts
    plt.subplot(1, 2, 1)
    before_bars = plt.bar([i - width/2 for i in x], before_values, width, label=f'Original ({input_name})', color='#3498db')
    after_bars = plt.bar([i + width/2 for i in x], after_values, width, label=f'Balanced ({output_name})', color='#2ecc71')
    
    plt.xlabel('Activity Type')
    plt.ylabel('Count')
    plt.title('Activity Distribution Change (Counts)')
    plt.xticks(x, activity_types)
    plt.legend()
    
    # Add count labels
    for i, count in enumerate(before_values):
        plt.text(i - width/2, count + 0.5, str(count), ha='center', va='bottom')
    
    for i, count in enumerate(after_values):
        plt.text(i + width/2, count + 0.5, str(count), ha='center', va='bottom')
    
    # Plot percentages
    plt.subplot(1, 2, 2)
    total_before = sum(before_values)
    total_after = sum(after_values)
    
    before_pct = [count/total_before*100 for count in before_values]
    after_pct = [count/total_after*100 for count in after_values]
    
    plt.bar([i - width/2 for i in x], before_pct, width, label=f'Original ({input_name})', color='#3498db')
    plt.bar([i + width/2 for i in x], after_pct, width, label=f'Balanced ({output_name})', color='#2ecc71')
    
    plt.xlabel('Activity Type')
    plt.ylabel('Percentage (%)')
    plt.title('Activity Distribution Change (Percentage)')
    plt.xticks(x, activity_types)
    plt.legend()
    
    # Add percentage labels
    for i, pct in enumerate(before_pct):
        plt.text(i - width/2, pct + 0.5, f"{pct:.1f}%", ha='center', va='bottom')
    
    for i, pct in enumerate(after_pct):
        plt.text(i + width/2, pct + 0.5, f"{pct:.1f}%", ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the visualization
    output_dir = "Plots/Dataset_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/distribution_change.png")
    plt.close()

def balance_all_datasets():
    """Balance all datasets in the Dataset directory."""
    # Define the target distribution
    target_distribution = {
        'Lecture': 40,  # 40% lectures
        'Tutorial': 40, # 40% tutorials
        'Lab': 20       # 20% labs
    }
    
    # Balance the 4-room dataset
    input_path_4room = "Dataset/sliit_computing_dataset_4.json"
    output_path_4room = "Dataset/sliit_computing_dataset_4_balanced.json"
    
    print(f"Balancing 4-room dataset...")
    balance_dataset(input_path_4room, output_path_4room, target_distribution)
    
    print("\n" + "="*50 + "\n")
    
    # Balance the 7-room dataset
    input_path_7room = "Dataset/sliit_computing_dataset_7.json"
    output_path_7room = "Dataset/sliit_computing_dataset_7_balanced.json"
    
    print(f"Balancing 7-room dataset...")
    balance_dataset(input_path_7room, output_path_7room, target_distribution)

if __name__ == "__main__":
    balance_all_datasets()
