yeimport json
from collections import defaultdict

# Load balanced dataset
with open('sliit_computing_dataset_balanced.json', 'r') as f:
    data = json.load(f)

# Collect all subgroups per year/semester
subgroups_by_year_sem = defaultdict(list)
for group in data.get('groups', []):
    # Expecting IDs like Y4S2.3 (Y=Year, S=Semester, .GroupNum)
    parts = group['id'].split('.')
    if len(parts) == 2 and parts[0].startswith('Y') and 'S' in parts[0]:
        year_sem = parts[0]  # e.g., Y4S2
        subgroups_by_year_sem[year_sem].append(group['id'])

# Fix tutorial activities
tutorials_fixed = 0
for activity in data.get('activities', []):
    if activity.get('type', '').lower() == 'tutorial':
        # Find year/semester from first subgroup
        if activity.get('subgroup_ids'):
            first_subgroup = activity['subgroup_ids'][0]
            year_sem = first_subgroup.split('.')[0]  # e.g., Y4S2
            # Set all 5 subgroups for this year/semester
            all_subgroups = sorted(subgroups_by_year_sem.get(year_sem, []))
            if set(activity['subgroup_ids']) != set(all_subgroups):
                activity['subgroup_ids'] = all_subgroups
                tutorials_fixed += 1

# Save the updated dataset
with open('sliit_computing_dataset_balanced.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Updated {tutorials_fixed} tutorial activities to include all subgroups per year/semester.")
