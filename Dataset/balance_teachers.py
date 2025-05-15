import json

# Load the dataset
with open('sliit_computing_dataset.json', 'r') as f:
    data = json.load(f)

# Only use FA... IDs as teachers (Faculty/Academic)
fa_teacher_ids = set()
for a in data["activities"]:
    for tid in a.get("teacher_ids", []):
        if tid.startswith("FA"):
            fa_teacher_ids.add(tid)
teacher_ids = sorted(fa_teacher_ids)
num_teachers = len(teacher_ids)
if num_teachers == 0:
    raise ValueError("No FA... teacher IDs found in dataset. Cannot balance assignments.")

# Remove any IT... IDs from teacher_ids in activities
def clean_teacher_ids(activity):
    return [tid for tid in activity.get("teacher_ids", []) if tid.startswith("FA")]

for activity in data["activities"]:
    activity["teacher_ids"] = clean_teacher_ids(activity)

# Assign activities round-robin to teachers
for i, activity in enumerate(data["activities"]):
    # Replace teacher_ids with just one teacher, assigned evenly
    activity["teacher_ids"] = [teacher_ids[i % num_teachers]]

# Save the balanced dataset
with open('sliit_computing_dataset_balanced.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Balanced dataset saved as 'sliit_computing_dataset_balanced.json'")
