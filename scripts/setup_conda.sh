#!/bin/bash
# Setup using conda or mamba

set -e

PROJECT_NAME="zero-prototype"
PYTHON_VERSION="3.10"

# Detect if mamba is available (faster than conda)
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba (faster conda alternative)..."
else
    CONDA_CMD="conda"
    echo "Using conda..."
fi

echo ""

# Create environment
echo "Creating conda environment: $PROJECT_NAME with Python $PYTHON_VERSION"
$CONDA_CMD create -n $PROJECT_NAME python=$PYTHON_VERSION -y

# Activate environment
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $PROJECT_NAME

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "✓ Conda environment setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  conda activate $PROJECT_NAME"
echo ""
echo "Or if using mamba:"
echo "  mamba activate $PROJECT_NAME"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo ""
echo "To run the pipeline:"
echo "  python src/main.py"
echo ""
echo "To launch the UI:"
echo "  python ui/app.py"
echo ""
