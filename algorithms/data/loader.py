class Space:
    def __init__(self, *args):
        self.code = args[0]
        self.size = args[1]

    def __repr__(self):
        return f"Space(code={self.code}, size={self.size})"


class Group:
    def __init__(self, *args):
        self.id = args[0]
        self.size = args[1]

    def __repr__(self):
        return f"Group(id={self.id}, size={self.size})"


class Activity:
    def __init__(self, id, *args):
        self.id = id
        self.subject = args[0]
        self.teacher_id = args[1]
        self.group_ids = args[2]
        self.duration = args[3]

    def __repr__(self):
        return f"Activity(id={self.id}, subject={self.subject}, teacher_id={self.teacher_id}, group_ids={self.group_ids}, duration={self.duration})"


class Period:
    def __init__(self, *args):
        self.space = args[0]
        self.slot = args[1]
        self.activity = args[2]

    def __repr__(self):
        return f"Period(space={self.space}, group={self.group}, activity={self.activity})"

class Lecturer:
    def __init__(self, id, first_name, last_name, username, department):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.username = username
        self.department = department

    def __repr__(self):
        return f"Lecturer(id={self.id}, name={self.first_name} {self.last_name}, department={self.department})"


import json
import os

def load_data(dataset_path=None):
    """
    Load timetable data from a JSON file.
    
    Args:
        dataset_path (str): Path to the dataset JSON file. If None, will use default.
        
    Returns:
        tuple: Tuple containing (activities_dict, groups_dict, spaces_dict, lecturers_dict, slots)
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    if dataset_path is None:
        # Use environment variable to select dataset, default to standard one
        dataset_filename = os.environ.get('TIMETABLE_DATASET', 'sliit_computing_dataset.json')
        
        # Try several potential locations for the dataset
        potential_paths = [
            os.path.join(project_root, 'data', dataset_filename),            # New data directory
            os.path.join(project_root, 'Dataset', dataset_filename),        # Old Dataset directory
            os.path.join(current_dir, '..', 'Dataset', dataset_filename),   # Relative from loader.py
            os.path.join(current_dir, dataset_filename)                     # In the same directory
        ]
        
        # Use the first path that exists
        for path in potential_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        else:
            raise FileNotFoundError(f"Dataset file {dataset_filename} not found in any of the expected locations")
    
    print(f"--- Loading Timetable Dataset from: {dataset_path} ---")
    
    # Load the data from the JSON file
    with open(dataset_path, 'r') as file:
        data = json.load(file)
    
    # Create dictionaries to store instances
    spaces_dict = {}
    groups_dict = {}
    activities_dict = {}
    lecturers_dict = {}
    slots = []

    # Populate the dictionaries with data from the JSON file
    for space in data['spaces']:
        spaces_dict[space['code']] = Space(space['code'], space['capacity'])

    for group in data['years']:
        groups_dict[group['id']] = Group(group['id'], group['size'])

    for activity in data['activities']:
        activities_dict[activity['code']] = Activity(
            activity['code'], activity['subject'], activity['teacher_ids'][0], 
            activity['subgroup_ids'], activity['duration'])

    for user in data["users"]:
        if user["role"] == "lecturer":
            lecturers_dict[user["id"]] = Lecturer(
                user["id"], user["first_name"], user["last_name"], 
                user["username"], user["department"]
            )

    for day in ["MON", "TUE", "WED", "THU", "FRI"]:
        for id in range(1, 9):
            slots.append(day+str(id))
            
    # Print summary of loaded data
    print(f"Loaded {len(spaces_dict)} spaces, {len(groups_dict)} groups, {len(activities_dict)} activities,"
          f" {len(lecturers_dict)} lecturers, and {len(slots)} time slots")
    
    return activities_dict, groups_dict, spaces_dict, lecturers_dict, slots

# Load the default dataset if this module is imported directly
if __name__ != "__main__":
    activities_dict, groups_dict, spaces_dict, lecturers_dict, slots = load_data()

# Add this if you need to test the loader directly
if __name__ == "__main__":
    # Test the loader
    activities_dict, groups_dict, spaces_dict, lecturers_dict, slots = load_data()
    
    # Print sample data for verification
    print("\nSample spaces:", list(spaces_dict.items())[:2])
    print("\nSample groups:", list(groups_dict.items())[:2])
    print("\nSample activities:", list(activities_dict.items())[:2])
    print("\nSample lecturers:", list(lecturers_dict.items())[:2])
    print("\nSlots:", slots[:5], "...")

class Period:
    def __init__(self, space, slot, activity=None):
        self.space = space
        self.slot = slot
        self.activity = activity

    def __repr__(self):
        return f"Period(space={self.space}, slot={self.slot}, activity={self.activity})"

def create_empty_schedule(spaces_list=None, slots_list=None):
    """
    Create an empty schedule structure.
    
    Args:
        spaces_list (list): List of space codes to use. If None, uses all spaces.
        slots_list (list): List of time slots. If None, uses all slots.
        
    Returns:
        dict: Empty schedule dictionary
    """
    if spaces_list is None:
        spaces_list = list(spaces_dict.keys())
        
    if slots_list is None:
        slots_list = slots
    
    # Build a schedule dictionary: for each slot, create a sub-dictionary for each space (initially None)
    schedule = {slot: {space: None for space in spaces_list} for slot in slots_list}
    
    return schedule

# If this module is imported directly, create a test schedule
if __name__ == "__main__":
    # Test schedule creation
    test_schedule = create_empty_schedule()
    print("\nSample schedule structure:")
    for slot in list(test_schedule.keys())[:2]:  # Just show first two slots
        print(f"{slot}: {list(test_schedule[slot].keys())[:2]} ...")