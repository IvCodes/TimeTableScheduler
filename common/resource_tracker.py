"""
Resource tracking module for timetable scheduling algorithms.
Provides decorators and utilities for monitoring computational resource usage.
"""

import time
import psutil
import functools
import numpy as np
from typing import Dict, Any, Callable, List, Tuple
import tracemalloc

# Global dictionary to store resource metrics for each algorithm run
ALGORITHM_RESOURCE_METRICS = {}


def track_computational_resources(algorithm_func):
    """
    Decorator to track computational resources used by algorithms.
    
    Unlike traditional decorators that might modify the return value,
    this one stores metrics in a global dictionary to avoid changing
    the function's return signature.
    
    Args:
        algorithm_func: Function implementing an optimization algorithm
        
    Returns:
        Wrapped function that tracks execution time and memory usage
    """
    @functools.wraps(algorithm_func)
    def wrapper(*args, **kwargs):
        # Get algorithm name from function name or keyword arg
        algorithm_name = kwargs.get('algorithm_name', algorithm_func.__name__)
        
        # Start resource tracking
        start_time = time.time()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Start with 0 peak memory usage
        peak_memory_usage = 0
        
        try:
            # Execute the original function
            result = algorithm_func(*args, **kwargs)
            
            # Track execution time
            execution_time = time.time() - start_time
            
            # Get peak memory info
            current, peak = tracemalloc.get_traced_memory()
            peak_memory_usage = peak / (1024 * 1024)  # Convert to MB
            
            # Store metrics
            ALGORITHM_RESOURCE_METRICS[algorithm_name] = {
                'execution_time': execution_time,
                'peak_memory_usage': peak_memory_usage,
                'timestamp': time.time()
            }
            
            print(f"{algorithm_name} execution completed in {execution_time:.2f} seconds")
            print(f"Peak memory usage: {peak_memory_usage:.2f} MB")
            
            return result
            
        except Exception as e:
            # Record failure
            ALGORITHM_RESOURCE_METRICS[algorithm_name] = {
                'execution_time': time.time() - start_time,
                'peak_memory_usage': peak_memory_usage,
                'error': str(e),
                'timestamp': time.time()
            }
            
            # Re-raise the exception
            raise
            
        finally:
            # Stop memory tracking
            tracemalloc.stop()
    
    return wrapper


class ResourceTracker:
    """Class for monitoring resource usage during algorithm execution."""
    
    def __init__(self, algorithm_name):
        """
        Initialize resource tracker.
        
        Args:
            algorithm_name: Name of the algorithm being tracked
        """
        self.algorithm_name = algorithm_name
        self.start_time = None
        self.execution_time = None
        self.peak_memory_usage = 0
        self.checkpoint_times = []
        self.memory_snapshots = []
        self.iteration_times = []
        self.is_tracking = False
    
    def start(self):
        """Start resource tracking."""
        self.start_time = time.time()
        tracemalloc.start()
        self.is_tracking = True
        self.checkpoint_times = []
        self.memory_snapshots = []
        self.iteration_times = []
        return self
    
    def stop(self):
        """Stop resource tracking and record final metrics."""
        if not self.is_tracking:
            return self
            
        self.execution_time = time.time() - self.start_time
        
        # Get peak memory info
        current, peak = tracemalloc.get_traced_memory()
        self.peak_memory_usage = peak / (1024 * 1024)  # Convert to MB
        
        # Store metrics in global dictionary
        ALGORITHM_RESOURCE_METRICS[self.algorithm_name] = {
            'execution_time': self.execution_time,
            'peak_memory_usage': self.peak_memory_usage,
            'checkpoint_times': self.checkpoint_times.copy() if self.checkpoint_times else [],
            'memory_snapshots': self.memory_snapshots.copy() if self.memory_snapshots else [],
            'iteration_times': self.iteration_times.copy() if self.iteration_times else [],
            'timestamp': time.time()
        }
        
        tracemalloc.stop()
        self.is_tracking = False
        
        print(f"{self.algorithm_name} execution completed in {self.execution_time:.2f} seconds")
        print(f"Peak memory usage: {self.peak_memory_usage:.2f} MB")
        
        return self
    
    def checkpoint(self, label=None):
        """
        Record a checkpoint during algorithm execution.
        
        Args:
            label: Optional label for the checkpoint
        """
        if not self.is_tracking:
            return self
            
        checkpoint_time = time.time() - self.start_time
        self.checkpoint_times.append((checkpoint_time, label))
        
        # Take memory snapshot
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / (1024 * 1024)  # Convert to MB
        self.memory_snapshots.append((checkpoint_time, current_mb, label))
        
        return self
    
    def record_iteration(self, iteration_time):
        """
        Record the time taken for a single iteration.
        
        Args:
            iteration_time: Time taken for the iteration in seconds
        """
        if not self.is_tracking:
            return self
            
        self.iteration_times.append(iteration_time)
        return self
    
    def __enter__(self):
        """Support for context manager (with statement)."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager (with statement)."""
        self.stop()
        return False  # re-raise any exceptions
    
    @property
    def metrics(self):
        """Get the current metrics as a dictionary."""
        return {
            'execution_time': self.execution_time,
            'peak_memory_usage': self.peak_memory_usage,
            'checkpoint_times': self.checkpoint_times.copy() if self.checkpoint_times else [],
            'memory_snapshots': self.memory_snapshots.copy() if self.memory_snapshots else [],
            'iteration_times': self.iteration_times.copy() if self.iteration_times else []
        }


def get_resource_metrics(algorithm_name=None):
    """
    Get resource metrics for a specific algorithm or all algorithms.
    
    Args:
        algorithm_name: Name of algorithm (optional, if None returns all)
        
    Returns:
        Dict: Resource metrics
    """
    if algorithm_name:
        return ALGORITHM_RESOURCE_METRICS.get(algorithm_name, {})
    return ALGORITHM_RESOURCE_METRICS


def clear_resource_metrics(algorithm_name=None):
    """
    Clear resource metrics for a specific algorithm or all algorithms.
    
    Args:
        algorithm_name: Name of algorithm (optional, if None clears all)
    """
    global ALGORITHM_RESOURCE_METRICS
    if algorithm_name:
        if algorithm_name in ALGORITHM_RESOURCE_METRICS:
            del ALGORITHM_RESOURCE_METRICS[algorithm_name]
    else:
        ALGORITHM_RESOURCE_METRICS = {}


def compare_resource_usage(algorithm_names, metrics=None):
    """
    Compare resource usage across algorithms.
    
    Args:
        algorithm_names: List of algorithm names to compare
        metrics: Optional dictionary of metrics (if None, uses global)
        
    Returns:
        Dict: Comparison metrics
    """
    if metrics is None:
        metrics = ALGORITHM_RESOURCE_METRICS
    
    comparison = {}
    
    # Execution time comparison
    execution_times = {}
    for algo in algorithm_names:
        if algo in metrics:
            execution_times[algo] = metrics[algo].get('execution_time', 0)
    
    if execution_times:
        fastest_algo = min(execution_times.items(), key=lambda x: x[1])[0]
        comparison['fastest_algorithm'] = fastest_algo
        comparison['execution_time_ratio'] = {
            algo: execution_times[algo] / execution_times[fastest_algo]
            for algo in execution_times
        }
    
    # Memory usage comparison
    memory_usages = {}
    for algo in algorithm_names:
        if algo in metrics:
            memory_usages[algo] = metrics[algo].get('peak_memory_usage', 0)
    
    if memory_usages:
        most_efficient_algo = min(memory_usages.items(), key=lambda x: x[1])[0]
        comparison['most_memory_efficient'] = most_efficient_algo
        comparison['memory_usage_ratio'] = {
            algo: memory_usages[algo] / memory_usages[most_efficient_algo]
            for algo in memory_usages
        }
    
    return comparison
