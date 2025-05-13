"""
Common data loading module for timetable scheduling algorithms.
This module contains data classes and functions used by both GA and RL algorithms.
"""

import json
import os
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional


@dataclass
class Activity:
    """Activity (class/lecture) data class."""
    id: str
    group_ids: List[str]
    lecturer_id: str
    subject: str
    duration: int = 1
    activity_type: str = "Lecture"
    constraints: List[Dict] = None
    size: int = 0  # Added for GA compatibility
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []


@dataclass
class Group:
    """Student group data class."""
    id: str
    size: int
    name: str = ""
    constraints: List[Dict] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if not self.name:
            self.name = f"Group {self.id}"


@dataclass
class Space:
    """Space (classroom/lab) data class."""
    id: str
    capacity: int
    name: str = ""
    constraints: List[Dict] = None
    duration: int = 1  # Added for GA compatibility
    teacher_id: Optional[str] = None  # Added for GA compatibility
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if not self.name:
            self.name = f"Space {self.id}"
    
    @property
    def size(self):
        """Alias for capacity to maintain compatibility with both codebases."""
        return self.capacity


@dataclass
class Lecturer:
    """Lecturer/teacher data class."""
    id: str
    name: str = ""
    constraints: List[Dict] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if not self.name:
            self.name = f"Lecturer {self.id}"


def load_data(dataset_path=None):
    """Load timetable data from a JSON file.
    
    Args:
        dataset_path: Path to the dataset file. If None, will try to locate the file.
        
    Returns:
        Tuple of (activities_dict, groups_dict, spaces_dict, lecturers_dict,
                activities_list, groups_list, spaces_list, lecturers_list,
                activity_types, timeslots_list, days_list, periods_list, slots)
    """
    # Try to find the dataset file if path not provided
    if dataset_path is None:
        # Try to use environment variable if set
        env_path = os.environ.get('TIMETABLE_DATASET')
        if env_path and os.path.exists(env_path):
            dataset_path = env_path
        else:
            # Try various common locations
            possible_paths = [
                os.path.join(os.getcwd(), 'sliit_computing_dataset.json'),
                os.path.join(os.getcwd(), 'sliit_computing_dataset_7.json'),
                os.path.join(os.getcwd(), 'data', 'sliit_computing_dataset.json'),
                os.path.join(os.getcwd(), 'data', 'sliit_computing_dataset_7.json'),
                os.path.join(os.getcwd(), 'Dataset', 'sliit_computing_dataset.json'),
                os.path.join(os.getcwd(), 'Dataset', 'sliit_computing_dataset_7.json'),
                os.path.join(os.getcwd(), '..', 'data', 'sliit_computing_dataset.json'),
                os.path.join(os.getcwd(), '..', 'data', 'sliit_computing_dataset_7.json'),
                os.path.join(os.getcwd(), '..', 'Dataset', 'sliit_computing_dataset.json'),
                os.path.join(os.getcwd(), '..', 'Dataset', 'sliit_computing_dataset_7.json'),
                os.path.join(os.getcwd(), '..', '..', 'data', 'sliit_computing_dataset.json'),
                os.path.join(os.getcwd(), '..', '..', 'data', 'sliit_computing_dataset_7.json'),
                os.path.join(os.getcwd(), '..', '..', 'Dataset', 'sliit_computing_dataset.json'),
                os.path.join(os.getcwd(), '..', '..', 'Dataset', 'sliit_computing_dataset_7.json'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_path = path
                    print(f"Found dataset at: {path}")
                    break
            
            if dataset_path is None:
                raise FileNotFoundError("Could not find the dataset file. Please specify the path.")
    
    # Load data from the dataset file
    try:
        with open(dataset_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")
    
    # Initialize dictionaries and lists
    activities_dict = {}
    groups_dict = {}
    spaces_dict = {}
    lecturers_dict = {}
    
    activities_list = []
    groups_list = []
    spaces_list = []
    lecturers_list = []
    
    # Process spaces (classrooms)
    for space_data in data.get('spaces', []):
        # Handle capacity/size field variations
        capacity = space_data.get('capacity', space_data.get('size', 0))
        space_id = space_data.get('id', space_data.get('code', ''))
        name = space_data.get('name', f"Space {space_id}")
        
        space = Space(
            id=space_id,
            name=name,
            capacity=capacity,
            constraints=space_data.get('constraints', [])
        )
        spaces_dict[space.id] = space
        spaces_list.append(space)
    
    # Process groups/years
    for group_data in data.get('groups', data.get('years', [])):
        group_id = group_data.get('id')
        if not group_id:
            continue
            
        group = Group(
            id=group_id,
            name=group_data.get('name', f"Group {group_id}"),
            size=group_data.get('size', 0),
            constraints=group_data.get('constraints', [])
        )
        groups_dict[group.id] = group
        groups_list.append(group)
    
    # Process lecturers/teachers/users
    lecturer_sources = [
        # Check dedicated lecturer list
        data.get('lecturers', []),
        # Check users with role=lecturer
        [u for u in data.get('users', []) if u.get('role') == 'lecturer']
    ]
    
    for source in lecturer_sources:
        for lecturer_data in source:
            lecturer_id = lecturer_data.get('id')
            if not lecturer_id or lecturer_id in lecturers_dict:
                continue
                
            # Handle various name formats
            name = lecturer_data.get('name')
            if not name and lecturer_data.get('first_name'):
                name = f"{lecturer_data.get('first_name', '')} {lecturer_data.get('last_name', '')}".strip()
            if not name:
                name = f"Lecturer {lecturer_id}"
                
            lecturer = Lecturer(
                id=lecturer_id,
                name=name,
                constraints=lecturer_data.get('constraints', [])
            )
            lecturers_dict[lecturer.id] = lecturer
            lecturers_list.append(lecturer)
    
    # Process activities/courses
    for activity_data in data.get('activities', []):
        # Handle different ID/code fields
        activity_id = activity_data.get('id', activity_data.get('code', ''))
        if not activity_id:
            continue
            
        # Handle different subject/name fields
        subject = activity_data.get('subject', activity_data.get('name', f"Activity {activity_id}"))
        
        # Handle different group ID formats
        group_ids = []
        if activity_data.get('group_id'):
            group_ids = [activity_data.get('group_id')]
        elif activity_data.get('group_ids'):
            group_ids = activity_data.get('group_ids')
        elif activity_data.get('subgroup_ids'):
            group_ids = activity_data.get('subgroup_ids')
            
        # Handle different teacher/lecturer ID formats
        lecturer_id = None
        if activity_data.get('lecturer_id'):
            lecturer_id = activity_data.get('lecturer_id')
        elif activity_data.get('teacher_id'):
            lecturer_id = activity_data.get('teacher_id')
        elif activity_data.get('teacher_ids') and len(activity_data.get('teacher_ids')) > 0:
            lecturer_id = activity_data.get('teacher_ids')[0]
            
        # Handle activity type
        activity_type = activity_data.get('activity_type', activity_data.get('type', 'Lecture'))
        
        activity = Activity(
            id=activity_id,
            group_ids=group_ids,
            lecturer_id=lecturer_id,
            duration=activity_data.get('duration', 1),
            activity_type=activity_type,
            subject=subject,
            constraints=activity_data.get('constraints', [])
        )
        activities_dict[activity.id] = activity
        activities_list.append(activity)
    
    # Extract activity types
    activity_types = list(set(activity.activity_type for activity in activities_list))
    
    # Define time slots
    days_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    # Use abbreviated day codes to match the dataset
    day_codes = ["MON", "TUE", "WED", "THU", "FRI"]
    periods_list = [f"Period {i}" for i in range(1, 9)]
    
    # Create slots as a list of day-period combinations
    slots = []
    timeslots_list = []
    for day_idx, day in enumerate(days_list):
        for period_idx, period in enumerate(periods_list):
            # Use the dataset format for slots (e.g., MON1, TUE2, etc.)
            slot = f"{day_codes[day_idx]}{period_idx + 1}"
            slots.append(slot)
            timeslots_list.append({"day": day, "period": period, "slot": slot})
    
    print(f"Successfully loaded: {len(activities_dict)} activities, {len(groups_dict)} groups, {len(spaces_dict)} spaces, {len(lecturers_dict)} lecturers")
    
    return (
        activities_dict, groups_dict, spaces_dict, lecturers_dict,
        activities_list, groups_list, spaces_list, lecturers_list,
        activity_types, timeslots_list, days_list, periods_list, slots
    )


# Helper function to initialize an empty schedule
def initialize_empty_schedule(slots, spaces_ids):
    """Initialize an empty schedule dictionary."""
    return {slot: {space_id: None for space_id in spaces_ids} for slot in slots}


# Helper function to check if a space is suitable for an activity
def is_space_suitable(space: Space, activity: Activity, groups_dict: Dict[str, Group]) -> bool:
    """Check if a space is suitable for an activity based on capacity."""
    # Handle empty or invalid group_ids
    if not activity.group_ids:
        return True
        
    # Calculate total size of all groups involved in the activity
    total_size = 0
    for group_id in activity.group_ids:
        group = groups_dict.get(group_id)
        if group:
            total_size += group.size
    
    return space.capacity >= total_size
