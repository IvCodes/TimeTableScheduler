import json
from collections import Counter, defaultdict

with open('sliit_computing_dataset_balanced.json', 'r') as f:
    data = json.load(f)

# 1. Teacher assignment balance
teacher_counts = Counter()
for a in data["activities"]:
    for tid in a.get("teacher_ids", []):
        teacher_counts[tid] += 1

print("Teacher assignment counts (should be even):")
for tid, count in sorted(teacher_counts.items(), key=lambda x: -x[1]):
    print(f"  {tid}: {count}")

# 2. Group overload
group_counts = Counter()
for a in data["activities"]:
    for gid in a.get("subgroup_ids", []) + a.get("group_ids", []):
        group_counts[gid] += 1

most_loaded_groups = group_counts.most_common(5)
print("\nMost loaded student groups:")
for gid, count in most_loaded_groups:
    print(f"  {gid}: {count} activities")

# 3. Space overload (space requirements)
space_counts = Counter()
for a in data["activities"]:
    for s in a.get("space_requirements", []):
        space_counts[s] += 1

print("\nRoom/space requirement counts:")
for s, count in space_counts.items():
    print(f"  {s}: {count} activities need this")

# 4. Activity duration
durations = Counter(a.get("duration", 1) for a in data["activities"])
print("\nActivity durations:")
for d, count in sorted(durations.items()):
    print(f"  Duration {d}: {count} activities")

# 5. Unassigned Entities
unassigned = [a["code"] for a in data["activities"] if not a.get("teacher_ids") or not (a.get("subgroup_ids") or a.get("group_ids"))]
print(f"\nActivities missing teacher or group assignment: {len(unassigned)}")
if unassigned:
    print("  Codes:", unassigned)

# 6. Duplicate Activities
codes = [a["code"] for a in data["activities"]]
dupes = [item for item, count in Counter(codes).items() if count > 1]
print(f"\nDuplicate activity codes: {dupes if dupes else 'None'}")
