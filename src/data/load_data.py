"""Data loading utilities for Communities and Crime dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_project_root


def parse_communities_names(names_file_path: Path) -> list:
    """
    Parse the communities.names file to extract column names.
    
    Args:
        names_file_path: Path to the communities.names file
        
    Returns:
        List of column names
    """
    column_names = []
    
    with open(names_file_path, 'r') as f:
        lines = f.readlines()
    
    # Look for @attribute lines after @relation line
    in_attributes = False
    
    for line in lines:
        line = line.strip()
        
        # Start collecting after @relation line
        if line.startswith('@relation'):
            in_attributes = True
            continue
        
        # Stop at @data marker
        if line == '@data':
            break
        
        # Extract attribute names
        if in_attributes and line.startswith('@attribute'):
            # Parse: @attribute name type
            parts = line.split()
            if len(parts) >= 2:
                attr_name = parts[1]
                column_names.append(attr_name)
    
    return column_names


def load_raw_data(
    data_file: Optional[Path] = None,
    names_file: Optional[Path] = None,
    config_path: str = "config/config.yaml"
) -> pd.DataFrame:
    """
    Load and parse the Communities and Crime dataset.
    
    Args:
        data_file: Path to communities.data file (optional, uses config if None)
        names_file: Path to communities.names file (optional, uses config if None)
        config_path: Path to configuration file
        
    Returns:
        DataFrame with loaded data
    """
    # Load configuration
    config = load_config(config_path)
    project_root = get_project_root()
    
    # Set default paths from config if not provided
    if data_file is None:
        data_file = project_root / config['data']['raw_data_file']
    else:
        data_file = Path(data_file)
    
    if names_file is None:
        names_file = project_root / config['data']['raw_names_file']
    else:
        names_file = Path(names_file)
    
    # Check if files exist
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not names_file.exists():
        raise FileNotFoundError(f"Names file not found: {names_file}")
    
    # Parse column names
    column_names = parse_communities_names(names_file)
    
    # Load data file (using '?' as missing value indicator)
    df = pd.read_csv(
        data_file,
        header=None,
        names=column_names,
        na_values='?',
        low_memory=False
    )
    
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def get_target_column(config_path: str = "config/config.yaml") -> str:
    """Get the target column name from config."""
    config = load_config(config_path)
    return config['model']['target_column']


def get_non_predictive_columns(config_path: str = "config/config.yaml") -> list:
    """Get non-predictive column names from config."""
    config = load_config(config_path)
    return config['model']['non_predictive_columns']


if __name__ == "__main__":
    # Test loading
    print("Testing data loading...")
    df = load_raw_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few columns: {df.columns[:10].tolist()}")
    print(f"\nTarget column: {get_target_column()}")
    print(f"\nNon-predictive columns: {get_non_predictive_columns()}")
