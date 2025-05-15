"""
TimeTable Scheduler - Azure Fabric Notebook Script (Part 1: Setup and Data Loading)
This script is formatted for easy conversion to a Jupyter notebook.
Each cell is marked with '# CELL: {description}' comments.
"""

# CELL: Import libraries and setup environment
import os
import sys
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import pickle

# For Azure Databricks/Fabric visualization
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, MapType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("Spark not available - some visualizations may be limited")

# Configure Matplotlib for better quality plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.style.use('ggplot')

print("Libraries imported successfully")

# CELL: Define data classes with compatibility for both GA and RL
@dataclass
class Activity:
    """Activity (class/lecture) data class."""
    id: str
    group_ids: List[str]
    lecturer_id: str
    subject: str
    duration: int = 1
    activity_type: str = "Lecture"
    constraints: Optional[List[Dict]] = None
    size: int = 0  # Added for GA compatibility
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class Group:
    """Student group data class."""
    id: str
    size: int
    name: str = ""
    constraints: Optional[List[Dict]] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class Space:
    """Space (classroom/lab) data class."""
    id: str
    capacity: int
    name: str = ""
    constraints: Optional[List[Dict]] = None
    duration: int = 1  # Added for GA compatibility
    teacher_id: Optional[str] = None  # Added for GA compatibility
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class Lecturer:
    """Lecturer data class."""
    id: str
    name: str = ""
    constraints: Optional[List[Dict]] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
    
    def __hash__(self):
        return hash(self.id)

# CELL: Function to load data from Azure Data Lake Storage
def load_data_from_adls(storage_account_name, container_name, file_path):
    """
    Load dataset from Azure Data Lake Storage Gen2
    
    Args:
        storage_account_name: Name of the storage account
        container_name: Name of the container
        file_path: Path to the JSON file in the container
        
    Returns:
        Tuple containing all the data structures
    """
    if SPARK_AVAILABLE:
        spark = SparkSession.builder.getOrCreate()
        
        # Create the storage path
        storage_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/{file_path}"
        
        # Read the JSON file
        df = spark.read.json(storage_path)
        
        # Convert to local Python dict
        data = json.loads(df.toJSON().first())
    else:
        # Fallback if Spark is not available (e.g., running locally)
        raise ValueError("Spark not available. Please configure access to Azure Data Lake Storage.")
    
    return process_data(data)

# CELL: Function to process data after loading
def process_data(data):
    """Process the loaded data and convert to appropriate data structures."""
    # Initialize collections
    activities_dict = {}
    groups_dict = {}
    spaces_dict = {}
    lecturers_dict = {}
    
    # Process spaces
    for space_data in data.get("spaces", []):
        space = Space(
            id=space_data["id"],
            capacity=space_data["capacity"],
            name=space_data.get("name", "")
        )
        spaces_dict[space.id] = space
    
    # Process groups
    for group_data in data.get("student_groups", []):
        group = Group(
            id=group_data["id"],
            size=group_data["size"],
            name=group_data.get("name", "")
        )
        groups_dict[group.id] = group
    
    # Process lecturers
    for lecturer_data in data.get("teachers", []):
        lecturer = Lecturer(
            id=lecturer_data["id"],
            name=lecturer_data.get("name", "")
        )
        lecturers_dict[lecturer.id] = lecturer
    
    # Process activities
    for activity_data in data.get("activities", []):
        activity = Activity(
            id=activity_data["id"],
            group_ids=activity_data["group_ids"],
            lecturer_id=activity_data["teacher_id"],
            subject=activity_data.get("subject", ""),
            activity_type=activity_data.get("type", "Lecture")
        )
        
        # Calculate size based on groups (for GA compatibility)
        activity.size = sum(groups_dict[g_id].size for g_id in activity.group_ids if g_id in groups_dict)
        
        activities_dict[activity.id] = activity
    
    # Extract timeslots
    days = data.get("days", [])
    periods = data.get("periods", [])
    
    # Create slots (day, period pairs)
    slots = [(day, period) for day in days for period in periods]
    
    # Create lists for easy iteration
    activities_list = list(activities_dict.values())
    groups_list = list(groups_dict.values())
    spaces_list = list(spaces_dict.values())
    lecturers_list = list(lecturers_dict.values())
    
    # Define activity types for filtering
    activity_types = list(set(a.activity_type for a in activities_list))
    
    # Return all data structures as a tuple
    return (
        activities_dict, groups_dict, spaces_dict, lecturers_dict,
        activities_list, groups_list, spaces_list, lecturers_list,
        activity_types, periods, days, periods, slots
    )

# CELL: Function to upload datasets to Azure Data Lake Storage
def upload_dataset_to_adls(local_file_path, storage_account_name, container_name, target_path):
    """
    Upload dataset from local machine to Azure Data Lake Storage
    
    This function only works in Azure Databricks environment
    """
    if not SPARK_AVAILABLE:
        print("Spark not available. This operation requires Azure Databricks.")
        return False
    
    spark = SparkSession.builder.getOrCreate()
    
    # Create the storage path
    storage_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/{target_path}"
    
    # Read the local file
    with open(local_file_path, 'r') as file:
        data = json.load(file)
    
    # Convert to Spark DataFrame
    df = spark.createDataFrame([{"data": json.dumps(data)}])
    
    # Write to ADLS
    try:
        # Converts single-row DataFrame with JSON string to a JSON file
        df.coalesce(1).write.mode("overwrite").json(storage_path)
        print(f"Successfully uploaded {local_file_path} to {storage_path}")
        return True
    except Exception as e:
        print(f"Error uploading to ADLS: {str(e)}")
        return False

# CELL: Sample code to upload datasets to ADLS
# Replace with your actual storage information
# storage_account = "yourstorageaccount"
# container = "datasets"
# 
# # Upload 4-room dataset
# upload_dataset_to_adls(
#     local_file_path="Dataset/sliit_computing_dataset.json",
#     storage_account_name=storage_account,
#     container_name=container,
#     target_path="timetable/sliit_computing_dataset.json"
# )
# 
# # Upload 7-room dataset
# upload_dataset_to_adls(
#     local_file_path="Dataset/sliit_computing_dataset_7.json",
#     storage_account_name=storage_account,
#     container_name=container,
#     target_path="timetable/sliit_computing_dataset_7.json"
# )

# CELL: Alternative - load dataset from local file in notebook
def load_data_from_local(dataset_path=None):
    """Load dataset from a local file path or environment variable"""
    dataset_paths = [
        dataset_path,
        os.environ.get("TIMETABLE_DATASET"),
        "Dataset/sliit_computing_dataset.json",
        "Dataset/sliit_computing_dataset_7.json",
        "data/sliit_computing_dataset.json",
        "data/sliit_computing_dataset_7.json",
        "../Dataset/sliit_computing_dataset.json",
        "../Dataset/sliit_computing_dataset_7.json",
        "../data/sliit_computing_dataset.json",
        "../data/sliit_computing_dataset_7.json",
    ]
    
    # Filter None values
    dataset_paths = [path for path in dataset_paths if path is not None]
    
    for path in dataset_paths:
        try:
            with open(path, 'r') as file:
                print(f"Found dataset at: {os.path.abspath(path)}")
                data = json.load(file)
                return process_data(data)
        except (FileNotFoundError, json.JSONDecodeError, IOError):
            continue
    
    raise FileNotFoundError("Could not find valid dataset file in any of the search paths")

# CELL: Display dataset info and statistics
def display_dataset_info(data_tuple):
    """Display information about the loaded dataset"""
    (
        activities_dict, groups_dict, spaces_dict, lecturers_dict,
        activities_list, groups_list, spaces_list, lecturers_list,
        activity_types, periods, days, _, slots
    ) = data_tuple
    
    print(f"Successfully loaded: {len(activities_dict)} activities, {len(groups_dict)} groups, {len(spaces_dict)} spaces, {len(lecturers_dict)} lecturers")
    print(f"\nActivity Types: {', '.join(activity_types)}")
    print(f"Days: {', '.join(days)}")
    print(f"Periods per day: {len(periods)}")
    print(f"Total slots: {len(slots)}")
    
    # Calculate max and min sizes
    if groups_list:
        max_group = max(groups_list, key=lambda g: g.size)
        min_group = min(groups_list, key=lambda g: g.size)
        print(f"\nStudent group sizes: Min={min_group.size} ({min_group.id}), Max={max_group.size} ({max_group.id})")
    
    if spaces_list:
        max_space = max(spaces_list, key=lambda s: s.capacity)
        min_space = min(spaces_list, key=lambda s: s.capacity)
        print(f"Room capacities: Min={min_space.capacity} ({min_space.id}), Max={max_space.capacity} ({max_space.id})")
    
    return data_tuple

# CELL: Demo loading dataset
# Uncomment and run this cell to load and display dataset info
# 
# # Choose one method to load data:
# 
# # Option 1: From local file
# data_tuple = load_data_from_local()
# display_dataset_info(data_tuple)
# 
# # Option 2: From ADLS (Azure Data Lake Storage)
# # storage_account = "yourstorageaccount"
# # container = "datasets"
# # data_tuple = load_data_from_adls(
# #     storage_account_name=storage_account,
# #     container_name=container,
# #     file_path="timetable/sliit_computing_dataset.json"
# # )
# # display_dataset_info(data_tuple)
