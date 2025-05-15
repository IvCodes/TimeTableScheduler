"""
Common statistical analysis module for timetable scheduling algorithms.
Provides functions for statistical testing and comparison of algorithm performance.
"""

import numpy as np
import itertools
from scipy import stats
from typing import Dict, List, Tuple, Any


def perform_statistical_tests(algorithm_results, metrics=['hypervolume', 'igd', 'execution_time']):
    """
    Perform statistical tests to compare algorithm performance.
    
    Args:
        algorithm_results: Dictionary mapping algorithm names to their results
        metrics: List of metrics to compare
        
    Returns:
        dict: Statistical test results
    """
    results = {}
    
    for metric in metrics:
        metric_values = {alg: res[metric] for alg, res in algorithm_results.items() 
                       if metric in res}
        
        # Skip metrics with insufficient data
        if not _has_sufficient_data(metric_values):
            continue
            
        # Perform Kruskal-Wallis test and add to results
        kruskal_result = _perform_kruskal_test(metric, metric_values)
        if kruskal_result:
            results.update(kruskal_result)
            
            # Only do pairwise tests if Kruskal test was significant
            if kruskal_result[f"{metric}_kruskal"]['significant']:
                pairwise_results = _perform_pairwise_tests(metric, metric_values)
                results.update(pairwise_results)
    
    return results


def _has_sufficient_data(metric_values):
    """Check if there is sufficient data for statistical tests."""
    if not metric_values:
        return False
        
    algorithms = list(metric_values.keys())
    values = [metric_values[alg] for alg in algorithms]
    
    # Need at least 2 samples per algorithm
    return all(len(v) > 1 for v in values)


def _perform_kruskal_test(metric, metric_values):
    """Perform Kruskal-Wallis test for a metric."""
    algorithms = list(metric_values.keys())
    values = [metric_values[alg] for alg in algorithms]
    
    h_stat, p_value = stats.kruskal(*values)
    return {
        f"{metric}_kruskal": {
            'statistic': h_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    }


def _perform_pairwise_tests(metric, metric_values):
    """Perform pairwise Mann-Whitney U tests between algorithms."""
    results = {}
    algorithms = list(metric_values.keys())
    
    for alg1, alg2 in itertools.combinations(algorithms, 2):
        u_stat, p_value = stats.mannwhitneyu(
            metric_values[alg1], metric_values[alg2], alternative='two-sided')
        
        # Determine which algorithm is better
        better = _determine_better_algorithm(metric, metric_values, alg1, alg2)
        
        results[f"{metric}_{alg1}_vs_{alg2}"] = {
            'statistic': u_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'better': better
        }
    
    return results


def _determine_better_algorithm(metric, metric_values, alg1, alg2):
    """Determine which algorithm performed better for a metric."""
    # For hypervolume, higher is better; for others, lower is better
    better_is_higher = metric == 'hypervolume'
    
    median1 = np.median(metric_values[alg1])
    median2 = np.median(metric_values[alg2])
    
    if better_is_higher:
        return alg1 if median1 > median2 else alg2
    else:
        return alg1 if median1 < median2 else alg2


def create_comparison_table(algorithm_results, metrics):
    """
    Create a comparison table of algorithm performance metrics.
    
    Args:
        algorithm_results: Dictionary mapping algorithm names to their results
        metrics: List of metrics to include in the table
        
    Returns:
        tuple: (Table header, table rows)
    """
    # Create header row
    alg_names = sorted(algorithm_results.keys())
    header = ["Metric"] + alg_names
    
    # Create data rows
    rows = []
    for metric in metrics:
        row = [metric]
        for alg in alg_names:
            if metric in algorithm_results[alg]:
                value = algorithm_results[alg][metric]
                if isinstance(value, (int, float)):
                    # Format based on metric type
                    if metric in ['igd', 'gd', 'spread', 'hypervolume', 'convergence_speed']:
                        row.append(f"{value:.4f}")
                    elif metric in ['execution_time', 'memory_usage']:
                        row.append(f"{value:.2f}")
                    else:
                        row.append(str(value))
                else:
                    row.append(str(value))
            else:
                row.append("-")
        rows.append(row)
    
    return header, rows


def identify_best_algorithm(algorithm_results, primary_metrics=['hypervolume', 'igd'], 
                           secondary_metrics=['execution_time', 'memory_usage']):
    """
    Identify the best algorithm based on performance metrics.
    
    Args:
        algorithm_results: Dictionary mapping algorithm names to their results
        primary_metrics: List of primary metrics for comparison
        secondary_metrics: List of secondary metrics for comparison
        
    Returns:
        dict: Results of best algorithm analysis
    """
    algorithms = list(algorithm_results.keys())
    if not algorithms:
        return {}
        
    scores = {alg: 0 for alg in algorithms}
    
    # Score based on primary metrics
    for metric in primary_metrics:
        _score_algorithms_by_metric(algorithm_results, algorithms, scores, metric, weight=2)
    
    # Score based on secondary metrics
    for metric in secondary_metrics:
        _score_algorithms_by_metric(algorithm_results, algorithms, scores, metric, weight=1)
    
    # Determine the best algorithm
    if scores:
        best_alg = max(scores.keys(), key=lambda alg: scores[alg])
        
        return {
            'best_algorithm': best_alg,
            'scores': scores,
            'reasons': {
                'primary_metrics': _get_metrics_for_alg(algorithm_results, best_alg, primary_metrics),
                'secondary_metrics': _get_metrics_for_alg(algorithm_results, best_alg, secondary_metrics)
            }
        }
    return {}


def _score_algorithms_by_metric(algorithm_results, algorithms, scores, metric, weight=1):
    """Helper function to score algorithms based on a metric."""
    metric_values = {alg: algorithm_results[alg].get(metric) 
                    for alg in algorithms 
                    if metric in algorithm_results[alg]}
    
    if not metric_values:
        return
        
    # For hypervolume, higher is better; for others, lower is better
    better_is_higher = metric == 'hypervolume'
    
    if better_is_higher:
        # Use explicit parameter m_values to avoid linting error with metric_values
        best_alg = max(metric_values.keys(), 
                     key=lambda alg, m_values=metric_values: m_values[alg])
    else:
        best_alg = min(metric_values.keys(), 
                     key=lambda alg, m_values=metric_values: m_values[alg])
        
    scores[best_alg] += weight


def _get_metrics_for_alg(algorithm_results, algorithm, metrics_list):
    """Helper function to get metrics for a specific algorithm."""
    return {m: algorithm_results[algorithm].get(m) for m in metrics_list 
           if m in algorithm_results[algorithm]}
