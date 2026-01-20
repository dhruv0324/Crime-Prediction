#!/bin/bash
# Setup script for DVC initialization and data tracking

set -e

echo "=========================================="
echo "DVC Setup for Crime Prediction Project"
echo "=========================================="
echo ""

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "[1/4] Initializing DVC..."
    dvc init --no-scm
else
    echo "[1/4] DVC already initialized ✓"
fi

# Configure DVC remote storage (local for now, can be changed to cloud later)
if [ ! -f ".dvc/config" ] || ! grep -q "\[remote" .dvc/config 2>/dev/null; then
    echo "[2/4] Configuring DVC remote storage..."
    mkdir -p dvc-storage
    dvc remote add -d local-storage ./dvc-storage 2>/dev/null || \
        dvc remote modify local-storage url ./dvc-storage
    echo "  Remote storage configured: ./dvc-storage"
else
    echo "[2/4] DVC remote storage already configured ✓"
fi

# Add raw data to DVC tracking
echo ""
echo "[3/4] Adding raw data to DVC tracking..."
if [ -f "data/raw/communities.data" ] && [ ! -f "data/raw/communities.data.dvc" ]; then
    echo "  Adding communities.data..."
    dvc add data/raw/communities.data
else
    echo "  communities.data already tracked or not found ✓"
fi

if [ -f "data/raw/communities.names" ] && [ ! -f "data/raw/communities.names.dvc" ]; then
    echo "  Adding communities.names..."
    dvc add data/raw/communities.names
else
    echo "  communities.names already tracked or not found ✓"
fi

# Add processed data to DVC tracking
echo ""
echo "[4/4] Adding processed data to DVC tracking..."
processed_files=(
    "data/processed/train.csv"
    "data/processed/val.csv"
    "data/processed/test.csv"
    "data/processed/train.parquet"
    "data/processed/val.parquet"
    "data/processed/test.parquet"
)

processed_count=0
for file in "${processed_files[@]}"; do
    if [ -f "$file" ]; then
        dvc_file="${file}.dvc"
        if [ ! -f "$dvc_file" ]; then
            echo "  Adding $(basename $file)..."
            dvc add "$file"
            processed_count=$((processed_count + 1))
        else
            echo "  $(basename $file) already tracked ✓"
        fi
    else
        echo "  $(basename $file) not found (skipping - run preprocessing first)"
    fi
done

if [ $processed_count -eq 0 ] && [ ! -f "data/processed/train.csv" ]; then
    echo "  No processed data found. Run preprocessing first:"
    echo "    python -m src.data.preprocess"
fi

echo ""
echo "=========================================="
echo "DVC Setup Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Raw data: Tracked by DVC"
if [ $processed_count -gt 0 ]; then
    echo "  - Processed data: $processed_count files added to DVC"
elif [ -f "data/processed/train.csv" ]; then
    echo "  - Processed data: Already tracked"
else
    echo "  - Processed data: Not found (run preprocessing first)"
fi
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Stage DVC files: git add data/**/*.dvc .dvcignore .dvc/config"
echo "  3. Commit: git commit -m 'Add data to DVC tracking'"
echo "  4. Push data (optional): dvc push"
echo ""