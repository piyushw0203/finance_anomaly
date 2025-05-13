import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def validate_time_column(series: pd.Series) -> Dict:
    """
    Validates if a column is suitable for time series analysis.
    """
    try:
        # Try to convert to datetime
        pd.to_datetime(series)
        is_time = True
    except:
        is_time = False
    
    # Check for missing values
    missing_count = series.isnull().sum()
    
    # Check if sorted
    is_sorted = series.is_monotonic_increasing
    
    # Check for duplicate timestamps
    duplicates = series.duplicated().sum()
    
    return {
        'is_valid_time': is_time,
        'missing_values': missing_count,
        'is_sorted': is_sorted,
        'duplicate_timestamps': duplicates
    }

def validate_value_column(series: pd.Series) -> Dict:
    """
    Validates if a column is suitable for value analysis.
    """
    # Check if numeric
    is_numeric = pd.api.types.is_numeric_dtype(series)
    
    # Check for missing values
    missing_count = series.isnull().sum()
    
    # Check for constant values
    is_constant = series.nunique() == 1
    
    # Check for infinite values
    infinite_count = np.isinf(series).sum()
    
    # Basic statistics
    stats = {
        'mean': series.mean() if is_numeric else None,
        'std': series.std() if is_numeric else None,
        'min': series.min() if is_numeric else None,
        'max': series.max() if is_numeric else None
    }
    
    return {
        'is_numeric': is_numeric,
        'missing_values': missing_count,
        'is_constant': is_constant,
        'infinite_values': infinite_count,
        'statistics': stats
    }

def validate_column_relationship(data: pd.DataFrame, time_column: str, value_column: str) -> Dict:
    """
    Validates the relationship between time and value columns.
    """
    time_validation = validate_time_column(data[time_column])
    value_validation = validate_value_column(data[value_column])
    
    # Check if time intervals are consistent
    if time_validation['is_valid_time']:
        time_diff = pd.to_datetime(data[time_column]).diff()
        time_diff_std = time_diff.std()
        time_diff_mean = time_diff.mean()
        is_consistent_interval = time_diff_std / time_diff_mean < 0.1  # 10% threshold
    else:
        is_consistent_interval = False
    
    return {
        'time_column': time_validation,
        'value_column': value_validation,
        'is_consistent_interval': is_consistent_interval,
        'warnings': generate_warnings(time_validation, value_validation, is_consistent_interval)
    }

def generate_warnings(time_validation: Dict, value_validation: Dict, is_consistent_interval: bool) -> List[str]:
    """
    Generates warning messages based on validation results.
    """
    warnings = []
    
    # Time column warnings
    if not time_validation['is_valid_time']:
        warnings.append("Time column is not in a valid time format")
    if time_validation['missing_values'] > 0:
        warnings.append(f"Time column has {time_validation['missing_values']} missing values")
    if not time_validation['is_sorted']:
        warnings.append("Time column is not sorted")
    if time_validation['duplicate_timestamps'] > 0:
        warnings.append(f"Time column has {time_validation['duplicate_timestamps']} duplicate timestamps")
    if not is_consistent_interval:
        warnings.append("Time intervals are not consistent")
    
    # Value column warnings
    if not value_validation['is_numeric']:
        warnings.append("Value column is not numeric")
    if value_validation['missing_values'] > 0:
        warnings.append(f"Value column has {value_validation['missing_values']} missing values")
    if value_validation['is_constant']:
        warnings.append("Value column contains constant values")
    if value_validation['infinite_values'] > 0:
        warnings.append(f"Value column has {value_validation['infinite_values']} infinite values")
    
    return warnings 