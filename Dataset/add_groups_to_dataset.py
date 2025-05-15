import json
from collections import defaultdict

# Load the balanced dataset
with open('sliit_computing_dataset_balanced.json', 'r') as f:
    data = json.load(f)

# Collect all unique group IDs from activities
unique_group_ids = set()
for activity in data.get("activities", []):
    for gid in activity.get("subgroup_ids", []) + activity.get("group_ids", []):
        unique_group_ids.add(gid)

# Try to infer group sizes from the original dataset if available
sizes_by_id = {}
try:
    with open('sliit_computing_dataset.json', 'r') as f2:
        orig_data = json.load(f2)
        for group in orig_data.get("groups", []):
            sizes_by_id[group["id"]] = group.get("size", 40)
except Exception:
    pass  # If original dataset or group sizes not found, use default

# Build group objects
new_groups = []
for gid in sorted(unique_group_ids):
    size = sizes_by_id.get(gid, 40)  # Default size 40
    group = {"id": gid, "size": size}
    new_groups.append(group)

# Insert or replace the groups array in the balanced dataset
data["groups"] = new_groups

# Save the updated dataset
with open('sliit_computing_dataset_balanced.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Added {len(new_groups)} groups to sliit_computing_dataset_balanced.json")
