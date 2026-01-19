"""Configuration loading utilities"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
