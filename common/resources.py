"""
Common resource tracking module for timetable scheduling algorithms.
Provides utilities for monitoring computational resources like execution time and memory usage.
"""

import time
import os
import psutil
import functools

# Global dictionary to store resource metrics for each algorithm run
ALGORITHM_RESOURCE_METRICS = {}

def track_computational_resources(algorithm_func):
    """
    Decorator to track computational resources used by algorithms.
    
    Unlike traditional decorators that might modify the return value,
    this one stores metrics in a global dictionary to avoid changing
    the function's return signature.
    
    Args:
        algorithm_func: Function implementing an evolutionary algorithm
        
    Returns:
        Wrapped function that tracks execution time and memory usage
    """
    @functools.wraps(algorithm_func)
    def wrapper(*args, **kwargs):
        # Start tracking
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run algorithm
        result = algorithm_func(*args, **kwargs)
        
        # End tracking
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        algorithm_name = algorithm_func.__name__.replace('run_', '').upper()
        
        # Store metrics in global dictionary instead of modifying the return value
        global ALGORITHM_RESOURCE_METRICS
        ALGORITHM_RESOURCE_METRICS[algorithm_name] = {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'algorithm': algorithm_name
        }
                
        print(f"\n{algorithm_name} Performance Metrics:")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Memory Usage: {memory_usage:.2f} MB")
                
        # Return original result without modification
        return result
        
    return wrapper

def get_resource_metrics():
    """
    Get the stored computational resource metrics.
    
    Returns:
        dict: Dictionary of algorithm resource metrics
    """
    return ALGORITHM_RESOURCE_METRICS
