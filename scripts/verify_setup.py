#!/usr/bin/env python3
"""Verify that the project setup is correct."""

import sys
from pathlib import Path


def check_file_exists(filepath: Path, description: str) -> bool:
    """Check if a file exists."""
    exists = filepath.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists


def check_directory_exists(dirpath: Path, description: str) -> bool:
    """Check if a directory exists."""
    exists = dirpath.exists() and dirpath.is_dir()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {dirpath}")
    return exists


def main():
    """Main verification function."""
    project_root = Path(__file__).parent.parent
    
    print("Verifying project setup...\n")
    
    # Check essential files
    files_to_check = [
        (project_root / "requirements.txt", "Requirements file"),
        (project_root / "setup.py", "Setup.py file"),
        (project_root / ".gitignore", ".gitignore file"),
        (project_root / "README.md", "README.md file"),
        (project_root / "config/config.yaml", "Configuration file"),
    ]
    
    # Check essential directories
    dirs_to_check = [
        (project_root / "src", "Source directory"),
        (project_root / "src/data", "Data module"),
        (project_root / "src/models", "Models module"),
        (project_root / "src/api", "API module"),
        (project_root / "tests", "Tests directory"),
        (project_root / "data/raw", "Raw data directory"),
        (project_root / "data/processed", "Processed data directory"),
    ]
    
    all_good = True
    
    print("Files:")
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_good = False
    
    print("\nDirectories:")
    for dirpath, description in dirs_to_check:
        if not check_directory_exists(dirpath, description):
            all_good = False
    
    # Check data files
    print("\nData files:")
    data_files = [
        (project_root / "data/raw/communities.data", "Communities data file"),
        (project_root / "data/raw/communities.names", "Communities names file"),
    ]
    for filepath, description in data_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Try importing key modules (optional - will fail until package is installed)
    print("\nPython imports (optional - requires package installation):")
    try:
        sys.path.insert(0, str(project_root / "src"))
        import src.utils.config
        print("✓ Configuration utility module")
    except ImportError as e:
        print(f"⚠ Configuration utility module not importable: {e}")
        print("  (This is expected until you run 'pip install -e .')")
        # Don't fail the check for this - it's expected before installation
    
    print("\n" + "="*50)
    if all_good:
        print("✓ All checks passed! Project setup is correct.")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
