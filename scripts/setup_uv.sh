#!/bin/bash
# Setup using uv (recommended - fastest)

set -e

PROJECT_NAME="zero-prototype-env"
PYTHON_VERSION="3.10"

echo "Setting up with uv..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# Create virtual environment with uv
echo "Creating virtual environment: $PROJECT_NAME"
uv venv $PROJECT_NAME --python $PYTHON_VERSION

# Activate the environment
echo ""
echo "Activating environment..."
source $PROJECT_NAME/bin/activate

# Upgrade pip, setuptools, wheel
echo "Upgrading pip and build tools..."
uv pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo ""
echo "============================================================"
echo "✓ uv virtual environment setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source $PROJECT_NAME/bin/activate"
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
