#!/bin/bash
# Virtual environment setup for zero-shot medical image captioning

# This script helps set up a virtual environment using either uv or conda

set -e

echo "============================================================"
echo "Virtual Environment Setup"
echo "============================================================"
echo ""

# Check what's available
echo "Checking available tools..."
echo ""

UV_AVAILABLE=false
CONDA_AVAILABLE=false

if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
    echo "✓ uv is available: $UV_VERSION"
    UV_AVAILABLE=true
else
    echo "✗ uv not found"
fi

if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version 2>/dev/null || echo "unknown")
    echo "✓ conda is available: $CONDA_VERSION"
    CONDA_AVAILABLE=true
else
    echo "✗ conda not found"
fi

if command -v mamba &> /dev/null; then
    MAMBA_VERSION=$(mamba --version 2>/dev/null || echo "unknown")
    echo "✓ mamba is available: $MAMBA_VERSION (faster conda alternative)"
    CONDA_AVAILABLE=true
    USE_MAMBA=true
else
    USE_MAMBA=false
fi

echo ""
echo "============================================================"
echo ""

if [ "$UV_AVAILABLE" = false ] && [ "$CONDA_AVAILABLE" = false ]; then
    echo "⚠ Neither uv nor conda found!"
    echo ""
    echo "Install one of them:"
    echo "  uv:    pip install uv"
    echo "  conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html"
    exit 1
fi

# Ask user preference
echo "Which virtual environment would you like to use?"
echo ""

if [ "$UV_AVAILABLE" = true ]; then
    echo "  1) uv (recommended - fast & simple)"
fi

if [ "$CONDA_AVAILABLE" = true ]; then
    if [ "$USE_MAMBA" = true ]; then
        echo "  2) mamba (fast conda)"
    else
        echo "  2) conda (traditional)"
    fi
fi

echo "  3) Python venv (no external tools)"
echo ""

read -p "Choose option (1-3): " choice

case $choice in
    1)
        if [ "$UV_AVAILABLE" = false ]; then
            echo "uv not available!"
            exit 1
        fi
        echo ""
        echo "Setting up with uv..."
        bash setup_uv.sh
        ;;
    2)
        if [ "$CONDA_AVAILABLE" = false ]; then
            echo "conda/mamba not available!"
            exit 1
        fi
        echo ""
        echo "Setting up with conda..."
        bash setup_conda.sh
        ;;
    3)
        echo ""
        echo "Setting up with Python venv..."
        bash setup_venv.sh
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✓ Setup complete!"
