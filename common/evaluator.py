"""
Common evaluation module for timetable scheduling algorithms.
Contains constraint evaluation functions shared between GA and RL implementations.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List, Any

from common.data_loader import Activity, Group, Space, Lecturer


def hard_constraint_space_capacity(schedule, activities_dict, groups_dict, spaces_dict):
    """Evaluate if any activities are scheduled in spaces with insufficient capacity."""
    violations = 0
    
    for slot, assignments in schedule.items():
        for space_id, activity_id in assignments.items():
            if activity_id is None:
                continue
                
            activity = activities_dict.get(activity_id)
            space = spaces_dict.get(space_id)
            
            if activity and space:
                if not is_space_suitable(space, activity, groups_dict):
                    violations += 1
    
    return violations


def hard_constraint_lecturer_clash(schedule, activities_dict):
    """Evaluate if any lecturer is assigned to multiple activities in the same timeslot."""
    violations = 0
    
    for slot, assignments in schedule.items():
        lecturer_activities = defaultdict(int)
        
        for space_id, activity_id in assignments.items():
            if activity_id is None:
                continue
                
            activity = activities_dict.get(activity_id)
            if activity and activity.lecturer_id:
                lecturer_activities[activity.lecturer_id] += 1
        
        # Count violations where a lecturer has more than one activity in a timeslot
        for lecturer_id, count in lecturer_activities.items():
            if count > 1:
                violations += (count - 1)
    
    return violations


def hard_constraint_group_clash(schedule, activities_dict):
    """Evaluate if any group is assigned to multiple activities in the same timeslot."""
    violations = 0
    group_assignments = defaultdict(list)
    
    # Collect all group assignments
    for slot, space_assignments in schedule.items():
        for space_id, activity_id in space_assignments.items():
            if activity_id:  # If there is an activity assigned
                activity = activities_dict.get(activity_id)
                if activity:
                    # Handle multiple group_ids for a single activity
                    for group_id in activity.group_ids:
                        group_assignments[(slot, group_id)].append(activity_id)
    
    # Check for violations
    for (slot, group_id), activity_ids in group_assignments.items():
        if len(activity_ids) > 1:  # Group is assigned to multiple activities in the same slot
            violations += len(activity_ids) - 1
    
    return violations


def hard_constraint_space_clash(schedule):
    """Evaluate if any space is assigned multiple activities in the same timeslot."""
    violations = 0
    
    for slot, assignments in schedule.items():
        for space_id, activity_id in assignments.items():
            if activity_id is None:
                continue
            
            # This is a simplification, as our data structure should prevent this
            # But included for completeness
            if isinstance(activity_id, list) and len(activity_id) > 1:
                violations += (len(activity_id) - 1)
    
    return violations


def soft_constraint_consecutive_activities(schedule, activities_dict, groups_dict, slots):
    """Evaluate preference for consecutive activities for the same group."""
    penalties = 0
    max_penalties = 0
    
    # Create a mapping from slots to their index
    slot_indices = {slot: idx for idx, slot in enumerate(slots)}
    
    # For each group, find its activities throughout the day
    group_daily_activities = defaultdict(lambda: defaultdict(list))
    
    for slot, space_assignments in schedule.items():
        # Extract day from slot (assuming format like 'Monday_Period 1' or 'MON1')
        if '_' in slot:
            day = slot.split('_')[0]  # Format: 'Monday_Period 1'
        else:
            day = slot[:3]  # Format: 'MON1'
        
        for space_id, activity_id in space_assignments.items():
            if activity_id:
                activity = activities_dict.get(activity_id)
                if activity:
                    # Handle multiple group_ids for a single activity
                    for group_id in activity.group_ids:
                        group_daily_activities[group_id][day].append((slot, activity_id))
    
    # For each group and day, check if activities are consecutive
    for group_id, daily_acts in group_daily_activities.items():
        for day, activities in daily_acts.items():
            if len(activities) <= 1:
                continue  # Skip if only one activity
                
            # Sort activities by slot index
            activities.sort(key=lambda x: slot_indices.get(x[0], 0))
            
            # Check for gaps between activities
            for i in range(len(activities) - 1):
                slot1, _ = activities[i]
                slot2, _ = activities[i + 1]
                
                idx1 = slot_indices.get(slot1, 0)
                idx2 = slot_indices.get(slot2, 0)
                
                if idx2 - idx1 > 1:  # There is a gap
                    penalties += 1
            
            # Maximum possible penalties for this group on this day
            max_penalties += len(activities) - 1
    
    # Normalize penalties to a 0-1 scale (0 is best, 1 is worst)
    normalized_score = penalties / max_penalties if max_penalties > 0 else 0
    
    return normalized_score


def soft_constraint_student_fatigue(schedule, activities_dict, groups_dict, slots):
    """Evaluate student fatigue based on consecutive hours of activities.
    Penalizes schedules where students have many consecutive hours without breaks."""
    max_consecutive_without_penalty = 3  # More than 3 consecutive hours causes fatigue
    fatigue_score = 0.0
    total_groups = 0
    
    # Map slots to indices and days
    slot_indices = {slot: idx for idx, slot in enumerate(slots)}
    slot_to_day = {}
    for slot in slots:
        if '_' in slot:
            day = slot.split('_')[0]  # Format: 'Monday_Period 1'
        else:
            day = slot[:3]  # Format: 'MON1'
        slot_to_day[slot] = day
    
    # Collect group activities by day
    group_daily_activities = defaultdict(lambda: defaultdict(list))
    for slot, space_assignments in schedule.items():
        day = slot_to_day.get(slot, '')
        for space_id, activity_id in space_assignments.items():
            if activity_id:
                activity = activities_dict.get(activity_id)
                if activity:
                    for group_id in activity.group_ids:
                        group_daily_activities[group_id][day].append((slot, activity_id))
    
    # Calculate fatigue for each group
    for group_id, daily_acts in group_daily_activities.items():
        total_groups += 1
        group_fatigue = 0.0
        for day, activities in daily_acts.items():
            if len(activities) <= max_consecutive_without_penalty:
                continue
            
            # Sort activities by slot
            activities.sort(key=lambda x: slot_indices.get(x[0], 0))
            
            # Find consecutive blocks
            current_consecutive = 1
            for i in range(1, len(activities)):
                slot1, _ = activities[i-1]
                slot2, _ = activities[i]
                
                idx1 = slot_indices.get(slot1, 0)
                idx2 = slot_indices.get(slot2, 0)
                
                if idx2 - idx1 == 1:  # Consecutive
                    current_consecutive += 1
                else:
                    # If we had a block exceeding threshold, add penalty
                    if current_consecutive > max_consecutive_without_penalty:
                        group_fatigue += (current_consecutive - max_consecutive_without_penalty) / max_consecutive_without_penalty
                    current_consecutive = 1
            
            # Check the last block
            if current_consecutive > max_consecutive_without_penalty:
                group_fatigue += (current_consecutive - max_consecutive_without_penalty) / max_consecutive_without_penalty
        
        # Average daily fatigue and add to total
        days_count = len(daily_acts)
        if days_count > 0:
            fatigue_score += group_fatigue / days_count
    
    # Normalize to 0-1 scale (higher means more fatigue)
    if total_groups == 0:
        return 0.0
        
    normalized_fatigue = min(1.0, fatigue_score / total_groups)
    return normalized_fatigue


def soft_constraint_student_idle_time(schedule, activities_dict, groups_dict, slots):
    """Evaluate student idle time (gaps between activities in the same day).
    Penalizes schedules where students have long gaps between classes."""
    idle_score = 0.0
    total_groups = 0
    max_acceptable_gap = 1  # One period gap is acceptable
    
    # Map slots to indices and days
    slot_indices = {slot: idx for idx, slot in enumerate(slots)}
    slot_to_day = {}
    for slot in slots:
        if '_' in slot:
            day = slot.split('_')[0]  # Format: 'Monday_Period 1'
        else:
            day = slot[:3]  # Format: 'MON1'
        slot_to_day[slot] = day
    
    # Group activities by day and group
    group_daily_activities = defaultdict(lambda: defaultdict(list))
    for slot, space_assignments in schedule.items():
        day = slot_to_day.get(slot, '')
        for space_id, activity_id in space_assignments.items():
            if activity_id:
                activity = activities_dict.get(activity_id)
                if activity:
                    for group_id in activity.group_ids:
                        group_daily_activities[group_id][day].append((slot, activity_id))
    
    # Calculate idle time for each group
    for group_id, daily_acts in group_daily_activities.items():
        total_groups += 1
        group_idle = 0.0
        for day, activities in daily_acts.items():
            if len(activities) <= 1:
                continue  # No idle time with 0 or 1 activity
            
            # Sort activities by slot
            activities.sort(key=lambda x: slot_indices.get(x[0], 0))
            
            # Calculate gaps
            total_gaps = 0
            excessive_gaps = 0
            for i in range(1, len(activities)):
                slot1, _ = activities[i-1]
                slot2, _ = activities[i]
                
                idx1 = slot_indices.get(slot1, 0)
                idx2 = slot_indices.get(slot2, 0)
                
                gap = idx2 - idx1 - 1  # -1 because consecutive slots have gap=0
                if gap > 0:
                    total_gaps += 1
                    if gap > max_acceptable_gap:
                        excessive_gaps += (gap - max_acceptable_gap)
            
            # If there are gaps, calculate score
            if total_gaps > 0:
                group_idle += excessive_gaps / (len(activities) - 1)  # Normalize by possible gaps
        
        # Average daily idle time and add to total
        days_count = len(daily_acts)
        if days_count > 0:
            idle_score += group_idle / days_count
    
    # Normalize to 0-1 scale
    if total_groups == 0:
        return 0.0
        
    normalized_idle = min(1.0, idle_score / total_groups)
    return normalized_idle


def soft_constraint_student_lecture_spread(schedule, activities_dict, groups_dict, slots):
    """Evaluate how well activities are spread across the week for each student group.
    Penalizes schedules where activities are concentrated on few days."""
    spread_score = 0.0
    total_groups = 0
    days_in_week = 5  # Assuming a 5-day week (Monday to Friday)
    
    # Identify the days in the schedule
    schedule_days = set()
    for slot in slots:
        if '_' in slot:
            day = slot.split('_')[0]  # Format: 'Monday_Period 1'
        else:
            day = slot[:3]  # Format: 'MON1'
        schedule_days.add(day)
    
    # If no valid days, return 0
    if not schedule_days:
        return 0.0
    
    # Count activities per group per day
    group_daily_activity_count = defaultdict(lambda: defaultdict(int))
    for slot, space_assignments in schedule.items():
        if '_' in slot:
            day = slot.split('_')[0]  # Format: 'Monday_Period 1'
        else:
            day = slot[:3]  # Format: 'MON1'
            
        for space_id, activity_id in space_assignments.items():
            if activity_id:
                activity = activities_dict.get(activity_id)
                if activity:
                    for group_id in activity.group_ids:
                        group_daily_activity_count[group_id][day] += 1
    
    # Calculate spread for each group
    for group_id, daily_counts in group_daily_activity_count.items():
        total_groups += 1
        days_with_activities = len(daily_counts)
        total_activities = sum(daily_counts.values())
        
        if total_activities == 0:
            continue
        
        # Ideal distribution: equal number of activities each day
        ideal_per_day = total_activities / min(days_in_week, len(schedule_days))
        
        # Calculate deviation from ideal
        deviation = 0
        for day, count in daily_counts.items():
            deviation += abs(count - ideal_per_day) / ideal_per_day
        
        # Normalize by number of days
        if days_with_activities > 0:
            spread_score += deviation / days_with_activities
    
    # Normalize to 0-1 scale (higher means worse spread)
    if total_groups == 0:
        return 0.0
        
    normalized_spread = min(1.0, spread_score / total_groups)
    return normalized_spread


def soft_constraint_preferred_times(schedule, activities_dict, lecturers_dict):
    """Evaluate lecturer preferences for specific timeslots."""
    violations = 0
    
    for slot, assignments in schedule.items():
        for space_id, activity_id in assignments.items():
            if activity_id is None:
                continue
                
            activity = activities_dict.get(activity_id)
            if not activity:
                continue
                
            lecturer = lecturers_dict.get(activity.lecturer_id)
            if not lecturer:
                continue
            
            # Check lecturer constraints for preferred times
            for constraint in lecturer.constraints:
                if constraint.get('type') == 'preferred_times':
                    preferred_times = constraint.get('times', [])
                    if slot not in preferred_times:
                        violations += 1
    
    return violations


def soft_constraint_lecturer_fatigue(schedule, activities_dict, lecturers_dict, slots):
    """Evaluate lecturer fatigue based on consecutive hours of teaching.
    Penalizes schedules where lecturers have many consecutive teaching hours without breaks."""
    max_consecutive_without_penalty = 3  # More than 3 consecutive hours causes fatigue
    fatigue_score = 0.0
    total_lecturers = 0
    
    # Map slots to indices and days
    slot_indices = {slot: idx for idx, slot in enumerate(slots)}
    slot_to_day = {}
    for slot in slots:
        if '_' in slot:
            day = slot.split('_')[0]  # Format: 'Monday_Period 1'
        else:
            day = slot[:3]  # Format: 'MON1'
        slot_to_day[slot] = day
    
    # Collect lecturer activities by day
    lecturer_daily_activities = defaultdict(lambda: defaultdict(list))
    for slot, space_assignments in schedule.items():
        day = slot_to_day.get(slot, '')
        for space_id, activity_id in space_assignments.items():
            if activity_id:
                activity = activities_dict.get(activity_id)
                if activity and activity.lecturer_id:
                    lecturer_daily_activities[activity.lecturer_id][day].append((slot, activity_id))
    
    # Calculate fatigue for each lecturer
    for lecturer_id, daily_acts in lecturer_daily_activities.items():
        total_lecturers += 1
        lecturer_fatigue = 0.0
        for day, activities in daily_acts.items():
            if len(activities) <= max_consecutive_without_penalty:
                continue
            
            # Sort activities by slot
            activities.sort(key=lambda x: slot_indices.get(x[0], 0))
            
            # Find consecutive blocks
            current_consecutive = 1
            for i in range(1, len(activities)):
                slot1, _ = activities[i-1]
                slot2, _ = activities[i]
                
                idx1 = slot_indices.get(slot1, 0)
                idx2 = slot_indices.get(slot2, 0)
                
                if idx2 - idx1 == 1:  # Consecutive
                    current_consecutive += 1
                else:
                    # If we had a block exceeding threshold, add penalty
                    if current_consecutive > max_consecutive_without_penalty:
                        lecturer_fatigue += (current_consecutive - max_consecutive_without_penalty) / max_consecutive_without_penalty
                    current_consecutive = 1
            
            # Check the last block
            if current_consecutive > max_consecutive_without_penalty:
                lecturer_fatigue += (current_consecutive - max_consecutive_without_penalty) / max_consecutive_without_penalty
        
        # Average daily fatigue and add to total
        days_count = len(daily_acts)
        if days_count > 0:
            fatigue_score += lecturer_fatigue / days_count
    
    # Normalize to 0-1 scale (higher means more fatigue)
    if total_lecturers == 0:
        return 0.0
        
    normalized_fatigue = min(1.0, fatigue_score / total_lecturers)
    return normalized_fatigue


def soft_constraint_lecturer_idle_time(schedule, activities_dict, lecturers_dict, slots):
    """Evaluate lecturer idle time (gaps between teaching activities in the same day).
    Penalizes schedules where lecturers have long gaps between classes."""
    idle_score = 0.0
    total_lecturers = 0
    max_acceptable_gap = 1  # One period gap is acceptable
    
    # Map slots to indices and days
    slot_indices = {slot: idx for idx, slot in enumerate(slots)}
    slot_to_day = {}
    for slot in slots:
        if '_' in slot:
            day = slot.split('_')[0]  # Format: 'Monday_Period 1'
        else:
            day = slot[:3]  # Format: 'MON1'
        slot_to_day[slot] = day
    
    # Group activities by day and lecturer
    lecturer_daily_activities = defaultdict(lambda: defaultdict(list))
    for slot, space_assignments in schedule.items():
        day = slot_to_day.get(slot, '')
        for space_id, activity_id in space_assignments.items():
            if activity_id:
                activity = activities_dict.get(activity_id)
                if activity and activity.lecturer_id:
                    lecturer_daily_activities[activity.lecturer_id][day].append((slot, activity_id))
    
    # Calculate idle time for each lecturer
    for lecturer_id, daily_acts in lecturer_daily_activities.items():
        total_lecturers += 1
        lecturer_idle = 0.0
        for day, activities in daily_acts.items():
            if len(activities) <= 1:
                continue  # No idle time with 0 or 1 activity
            
            # Sort activities by slot
            activities.sort(key=lambda x: slot_indices.get(x[0], 0))
            
            # Calculate gaps
            total_gaps = 0
            excessive_gaps = 0
            for i in range(1, len(activities)):
                slot1, _ = activities[i-1]
                slot2, _ = activities[i]
                
                idx1 = slot_indices.get(slot1, 0)
                idx2 = slot_indices.get(slot2, 0)
                
                gap = idx2 - idx1 - 1  # -1 because consecutive slots have gap=0
                if gap > 0:
                    total_gaps += 1
                    if gap > max_acceptable_gap:
                        excessive_gaps += (gap - max_acceptable_gap)
            
            # If there are gaps, calculate score
            if total_gaps > 0:
                lecturer_idle += excessive_gaps / (len(activities) - 1)  # Normalize by possible gaps
        
        # Average daily idle time and add to total
        days_count = len(daily_acts)
        if days_count > 0:
            idle_score += lecturer_idle / days_count
    
    # Normalize to 0-1 scale
    if total_lecturers == 0:
        return 0.0
        
    normalized_idle = min(1.0, idle_score / total_lecturers)
    return normalized_idle


def soft_constraint_lecturer_lecture_spread(schedule, activities_dict, lecturers_dict, slots):
    """Evaluate how well activities are spread across the week for each lecturer.
    Penalizes schedules where teaching is concentrated on few days."""
    spread_score = 0.0
    total_lecturers = 0
    days_in_week = 5  # Assuming a 5-day week (Monday to Friday)
    
    # Identify the days in the schedule
    schedule_days = set()
    for slot in slots:
        if '_' in slot:
            day = slot.split('_')[0]  # Format: 'Monday_Period 1'
        else:
            day = slot[:3]  # Format: 'MON1'
        schedule_days.add(day)
    
    # If no valid days, return 0
    if not schedule_days:
        return 0.0
    
    # Count activities per lecturer per day
    lecturer_daily_activity_count = defaultdict(lambda: defaultdict(int))
    for slot, space_assignments in schedule.items():
        if '_' in slot:
            day = slot.split('_')[0]  # Format: 'Monday_Period 1'
        else:
            day = slot[:3]  # Format: 'MON1'
            
        for space_id, activity_id in space_assignments.items():
            if activity_id:
                activity = activities_dict.get(activity_id)
                if activity and activity.lecturer_id:
                    lecturer_daily_activity_count[activity.lecturer_id][day] += 1
    
    # Calculate spread for each lecturer
    for lecturer_id, daily_counts in lecturer_daily_activity_count.items():
        total_lecturers += 1
        days_with_activities = len(daily_counts)
        total_activities = sum(daily_counts.values())
        
        if total_activities == 0:
            continue
        
        # Ideal distribution: equal number of activities each day
        ideal_per_day = total_activities / min(days_in_week, len(schedule_days))
        
        # Calculate deviation from ideal
        deviation = 0
        for day, count in daily_counts.items():
            deviation += abs(count - ideal_per_day) / ideal_per_day
        
        # Normalize by number of days
        if days_with_activities > 0:
            spread_score += deviation / days_with_activities
    
    # Normalize to 0-1 scale (higher means worse spread)
    if total_lecturers == 0:
        return 0.0
        
    normalized_spread = min(1.0, spread_score / total_lecturers)
    return normalized_spread


def soft_constraint_lecturer_workload_balance(schedule, activities_dict, lecturers_dict):
    """Evaluate how balanced the workload is across all lecturers.
    Penalizes schedules where some lecturers have much more work than others."""
    if not lecturers_dict or len(lecturers_dict) <= 1:
        return 0.0  # No imbalance with 0 or 1 lecturer
    
    # Count assigned activities per lecturer
    lecturer_activity_count = defaultdict(int)
    for slot, space_assignments in schedule.items():
        for space_id, activity_id in space_assignments.items():
            if activity_id:
                activity = activities_dict.get(activity_id)
                if activity and activity.lecturer_id:
                    lecturer_activity_count[activity.lecturer_id] += 1
    
    # Get activity counts as list
    activity_counts = list(lecturer_activity_count.values())
    if not activity_counts:
        return 0.0
    
    # Calculate statistics
    avg_activities = sum(activity_counts) / len(activity_counts)
    if avg_activities == 0:
        return 0.0
    
    # Calculate standard deviation
    variance = sum((count - avg_activities) ** 2 for count in activity_counts) / len(activity_counts)
    std_dev = variance ** 0.5
    
    # Coefficient of variation (normalized std dev) as measure of imbalance
    # CV = std_dev / mean (lower is better balanced)
    coefficient_of_variation = std_dev / avg_activities if avg_activities > 0 else 0
    
    # Normalize to 0-1 scale (higher means more imbalance)
    # Usually CV > 0.5 indicates significant imbalance
    normalized_imbalance = min(1.0, coefficient_of_variation * 2)
    return normalized_imbalance


def calculate_room_utilization(schedule, activities_dict, spaces_dict):
    """Calculate the room utilization percentage."""
    total_slots = len(schedule) * len(spaces_dict)
    used_slots = 0
    
    for slot, assignments in schedule.items():
        for space_id, activity_id in assignments.items():
            if activity_id is not None:
                used_slots += 1
    
    if total_slots == 0:
        return 0
    
    return used_slots / total_slots


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


def evaluate_schedule(schedule, activities_dict, groups_dict, spaces_dict, lecturers_dict, slots):
    """Evaluate a schedule based on hard and soft constraints.
    
    Returns:
        Tuple of (fitness_score, hard_violations, soft_violations)
    """
    # Hard constraints (must be satisfied)
    hard_violations = (
        hard_constraint_space_capacity(schedule, activities_dict, groups_dict, spaces_dict) +
        hard_constraint_lecturer_clash(schedule, activities_dict) +
        hard_constraint_group_clash(schedule, activities_dict) +
        hard_constraint_space_clash(schedule)
    )
    
    # Count unassigned activities
    scheduled_activities = set()
    for slot, assignments in schedule.items():
        for space_id, activity_id in assignments.items():
            if activity_id is not None:
                scheduled_activities.add(activity_id)
    
    unassigned = len(activities_dict) - len(scheduled_activities)
    hard_violations += unassigned
    
    # Soft constraints (preferences)
    # Student-related soft constraints
    student_consecutive = soft_constraint_consecutive_activities(schedule, activities_dict, groups_dict, slots)
    student_fatigue = soft_constraint_student_fatigue(schedule, activities_dict, groups_dict, slots)
    student_idle = soft_constraint_student_idle_time(schedule, activities_dict, groups_dict, slots)
    student_spread = soft_constraint_student_lecture_spread(schedule, activities_dict, groups_dict, slots)
    
    # Lecturer-related soft constraints
    lecturer_preferred = soft_constraint_preferred_times(schedule, activities_dict, lecturers_dict)
    lecturer_fatigue = soft_constraint_lecturer_fatigue(schedule, activities_dict, lecturers_dict, slots)
    lecturer_idle = soft_constraint_lecturer_idle_time(schedule, activities_dict, lecturers_dict, slots)
    lecturer_spread = soft_constraint_lecturer_lecture_spread(schedule, activities_dict, lecturers_dict, slots)
    lecturer_balance = soft_constraint_lecturer_workload_balance(schedule, activities_dict, lecturers_dict)
    
    # Combine all soft constraints
    soft_violations = (
        student_consecutive + 
        student_fatigue + 
        student_idle + 
        student_spread + 
        lecturer_preferred + 
        lecturer_fatigue + 
        lecturer_idle + 
        lecturer_spread + 
        lecturer_balance
    )
    
    # Calculate fitness (higher is better)
    # We prioritize hard constraints with a larger weight
    hard_weight = 100
    soft_weight = 1
    
    fitness = -1 * (hard_weight * hard_violations + soft_weight * soft_violations)
    
    return fitness, hard_violations, soft_violations


def multi_objective_evaluator(timetable, activities_dict, groups_dict, lecturers_dict, spaces_dict, slots):
    """Evaluate a timetable and return a tuple of objectives to be minimized.
    Used for GA approaches."""
    # Count hard constraint violations
    prof_conflicts = hard_constraint_lecturer_clash(timetable, activities_dict)
    
    sub_group_conflicts = hard_constraint_group_clash(timetable, activities_dict)
    
    room_size_conflicts = hard_constraint_space_capacity(
        timetable, activities_dict, groups_dict, spaces_dict)
    
    # Count unassigned activities
    scheduled_activities = set()
    for slot, assignments in timetable.items():
        for space_id, activity_id in assignments.items():
            if activity_id is not None:
                scheduled_activities.add(activity_id)
    
    unassigned = len(activities_dict) - len(scheduled_activities)
    
    # Calculate all soft constraint values (higher is worse)
    student_fatigue = soft_constraint_student_fatigue(timetable, activities_dict, groups_dict, slots)
    student_idle = soft_constraint_student_idle_time(timetable, activities_dict, groups_dict, slots)
    student_spread = soft_constraint_student_lecture_spread(timetable, activities_dict, groups_dict, slots)
    lecturer_fatigue = soft_constraint_lecturer_fatigue(timetable, activities_dict, lecturers_dict, slots)
    lecturer_idle = soft_constraint_lecturer_idle_time(timetable, activities_dict, lecturers_dict, slots)
    lecturer_spread = soft_constraint_lecturer_lecture_spread(timetable, activities_dict, lecturers_dict, slots)
    lecturer_workload = soft_constraint_lecturer_workload_balance(timetable, activities_dict, lecturers_dict)
    consecutive = soft_constraint_consecutive_activities(timetable, activities_dict, groups_dict, slots)
    preferred_times = soft_constraint_preferred_times(timetable, activities_dict, lecturers_dict)
    
    # Normalize the soft constraint score (0 to 1, where 1 is best)
    soft_score = 1.0 - (
        student_fatigue + 
        student_idle + 
        student_spread + 
        lecturer_fatigue + 
        lecturer_idle + 
        lecturer_spread + 
        lecturer_workload + 
        consecutive + 
        preferred_times
    ) / 9.0  # Divide by number of constraints
    
    # Return objectives to minimize
    return (
        prof_conflicts,          # 1. Professor Conflicts
        sub_group_conflicts,      # 2. Sub-group Conflicts
        room_size_conflicts,      # 3. Room Size Conflicts
        unassigned,               # 4. Unassigned Activities
        1.0 - soft_score,         # 5. Inverted Combined Soft Score
        student_fatigue,          # 6. Student Fatigue 
        student_idle,             # 7. Student Idle Time
        lecturer_fatigue,         # 8. Lecturer Fatigue
        lecturer_workload         # 9. Workload Balance
    )
