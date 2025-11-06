#!/bin/bash
# Setup using standard Python venv

set -e

ENV_DIR="venv"
PYTHON_CMD="python3"

echo "Setting up with Python venv..."
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment: $ENV_DIR"
$PYTHON_CMD -m venv $ENV_DIR

# Activate environment
echo ""
echo "Activating environment..."
source $ENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "✓ Python venv setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source $ENV_DIR/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "To run the pipeline:"
echo "  python src/main.py"
echo ""
echo "To launch the UI:"
echo "  python ui/app.py"
echo ""
