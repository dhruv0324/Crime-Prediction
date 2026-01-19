#!/bin/bash
# Setup script for DVC initialization

set -e

echo "Setting up DVC for data versioning..."

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init --no-scm
fi

# Configure DVC remote storage (local for now, can be changed to cloud later)
if [ ! -f ".dvc/config" ]; then
    echo "Configuring DVC remote storage..."
    mkdir -p dvc-storage
    dvc remote add -d local-storage ./dvc-storage
fi

# Add data to DVC tracking
echo "Adding raw data to DVC tracking..."
dvc add data/raw/communities.data
dvc add data/raw/communities.names

# Stage DVC files
git add data/raw/*.dvc .dvcignore .dvc/config

echo "DVC setup complete!"
echo "Data files are now tracked by DVC."
echo ""
echo "To add data to git tracking, run:"
echo "  git add data/raw/*.dvc .dvcignore .dvc/config"
echo ""
echo "To push data to remote storage, run:"
echo "  dvc push"
