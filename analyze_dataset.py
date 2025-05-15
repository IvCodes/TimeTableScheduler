import json
import os
import matplotlib.pyplot as plt
from collections import Counter

def analyze_dataset(dataset_path):
    """Analyze the distribution of activity types in the dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    activities = data.get('activities', [])
    total_activities = len(activities)
    
    # Infer activity types based on activity name/description
    activity_types = []
    for activity in activities:
        name = activity.get('name', '').lower()
        description = activity.get('description', '').lower()
        duration = activity.get('duration', 0)
        
        # Infer type based on keywords in name or description
        if 'tutorial' in name or 'tutorial' in description:
            activity_types.append('Tutorial')
        elif 'lab' in name or 'lab' in description or 'practical' in name or 'practical' in description:
            activity_types.append('Lab')
        elif 'lecture' in name or 'lecture' in description:
            activity_types.append('Lecture')
        elif duration <= 60:  # Shorter activities are likely tutorials
            activity_types.append('Tutorial')
        elif duration >= 120:  # Longer activities are likely labs
            activity_types.append('Lab')
        else:
            activity_types.append('Lecture')  # Default to lecture
    type_counts = Counter(activity_types)
    
    print(f"Dataset: {os.path.basename(dataset_path)}")
    print(f"Total activities: {total_activities}")
    print("Distribution:")
    
    for activity_type, count in type_counts.items():
        percentage = (count / total_activities) * 100
        print(f"  {activity_type}: {count} ({percentage:.1f}%)")
    
    # Create visualization
    labels = list(type_counts.keys())
    counts = list(type_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.title(f'Activity Type Distribution in {os.path.basename(dataset_path)}')
    plt.xlabel('Activity Type')
    plt.ylabel('Count')
    
    # Add count and percentage labels on bars
    for i, count in enumerate(counts):
        percentage = (count / total_activities) * 100
        plt.text(i, count + 0.5, f'{count} ({percentage:.1f}%)', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_dir = "Plots/Dataset_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{os.path.basename(dataset_path).replace('.json', '_distribution.png')}")
    plt.close()
    
    return type_counts

def compare_datasets():
    """Compare the distribution of activity types between 4-room and 7-room datasets."""
    dataset_4room = "Dataset/sliit_computing_dataset_4.json"
    dataset_7room = "Dataset/sliit_computing_dataset_7.json"
    
    counts_4room = analyze_dataset(dataset_4room)
    print("\n" + "="*50 + "\n")
    counts_7room = analyze_dataset(dataset_7room)
    
    # Create comparison visualization
    activity_types = sorted(set(list(counts_4room.keys()) + list(counts_7room.keys())))
    counts_4 = [counts_4room.get(t, 0) for t in activity_types]
    counts_7 = [counts_7room.get(t, 0) for t in activity_types]
    
    x = range(len(activity_types))
    width = 0.35
    
    plt.figure(figsize=(12, 7))
    plt.bar([i - width/2 for i in x], counts_4, width, label='4-Room Dataset', color='#3498db')
    plt.bar([i + width/2 for i in x], counts_7, width, label='7-Room Dataset', color='#e74c3c')
    
    plt.xlabel('Activity Type')
    plt.ylabel('Count')
    plt.title('Activity Type Distribution Comparison Between Datasets')
    plt.xticks(x, activity_types)
    plt.legend()
    
    # Add count labels
    for i, count in enumerate(counts_4):
        if count > 0:
            plt.text(i - width/2, count + 0.5, str(count), ha='center', va='bottom')
    
    for i, count in enumerate(counts_7):
        if count > 0:
            plt.text(i + width/2, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    output_dir = "Plots/Dataset_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/dataset_comparison.png")
    
    # Create a normalized comparison (percentage-based)
    plt.figure(figsize=(12, 7))
    
    total_4 = sum(counts_4)
    total_7 = sum(counts_7)
    
    percentages_4 = [count/total_4*100 for count in counts_4]
    percentages_7 = [count/total_7*100 for count in counts_7]
    
    plt.bar([i - width/2 for i in x], percentages_4, width, label='4-Room Dataset', color='#3498db')
    plt.bar([i + width/2 for i in x], percentages_7, width, label='7-Room Dataset', color='#e74c3c')
    
    plt.xlabel('Activity Type')
    plt.ylabel('Percentage (%)')
    plt.title('Activity Type Distribution Comparison (Percentage)')
    plt.xticks(x, activity_types)
    plt.legend()
    
    # Add percentage labels
    for i, percentage in enumerate(percentages_4):
        if percentage > 0:
            plt.text(i - width/2, percentage + 0.5, f"{percentage:.1f}%", ha='center', va='bottom')
    
    for i, percentage in enumerate(percentages_7):
        if percentage > 0:
            plt.text(i + width/2, percentage + 0.5, f"{percentage:.1f}%", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset_comparison_percentage.png")

if __name__ == "__main__":
    compare_datasets()
