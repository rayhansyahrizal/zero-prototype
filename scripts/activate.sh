#!/bin/bash
# Quick environment setup and activation script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_help() {
    cat << EOF
Zero-Shot Medical Image Captioning - Environment Manager

Usage: source activate.sh [command]

Commands:
  setup     Create virtual environment (interactive menu)
  uv        Setup with uv
  conda     Setup with conda/mamba
  venv      Setup with Python venv
  check     Verify current environment
  info      Show environment info
  clean     Remove virtual environments

Examples:
  source activate.sh setup     # Interactive setup
  source activate.sh uv        # Quick uv setup
  source activate.sh check     # Verify environment

Note: This script must be sourced with 'source' to activate the environment!
EOF
}

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

COMMAND=${1:-""}

case $COMMAND in
    setup)
        echo -e "${BLUE}Starting environment setup...${NC}"
        bash setup_env.sh
        ;;
    
    uv)
        echo -e "${BLUE}Setting up with uv...${NC}"
        bash setup_uv.sh
        # Auto-activate if successful
        if [ -f "zero-prototype-env/bin/activate" ]; then
            echo -e "${GREEN}Activating environment...${NC}"
            source zero-prototype-env/bin/activate
            echo -e "${GREEN}✓ Environment activated!${NC}"
            python --version
        fi
        ;;
    
    conda)
        echo -e "${BLUE}Setting up with conda...${NC}"
        bash setup_conda.sh
        ;;
    
    venv)
        echo -e "${BLUE}Setting up with Python venv...${NC}"
        bash setup_venv.sh
        # Auto-activate if successful
        if [ -f "venv/bin/activate" ]; then
            echo -e "${GREEN}Activating environment...${NC}"
            source venv/bin/activate
            echo -e "${GREEN}✓ Environment activated!${NC}"
            python --version
        fi
        ;;
    
    check)
        echo -e "${BLUE}Checking current environment...${NC}"
        echo ""
        
        if [ -z "$VIRTUAL_ENV" ]; then
            echo -e "${YELLOW}⚠ No virtual environment activated${NC}"
            echo ""
            echo "To activate an environment, run:"
            echo "  source activate.sh uv      # For uv environment"
            echo "  source activate.sh venv    # For venv environment"
            echo "  conda activate zero-prototype  # For conda"
        else
            echo -e "${GREEN}✓ Virtual environment active${NC}"
            echo "  Environment: $(basename $VIRTUAL_ENV)"
            echo "  Location: $VIRTUAL_ENV"
            echo ""
            echo "Python info:"
            python --version
            echo "  Location: $(which python)"
            echo ""
            echo "Installed packages:"
            pip list | head -5
            echo "  ... (and more)"
        fi
        ;;
    
    info)
        echo -e "${BLUE}Environment Information${NC}"
        echo ""
        
        # Check available tools
        echo "Available environment managers:"
        echo ""
        
        if command -v uv &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} uv $(uv --version 2>&1 | awk '{print $NF}')"
        else
            echo -e "  ${YELLOW}✗${NC} uv not installed"
        fi
        
        if command -v conda &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} conda $(conda --version 2>&1 | awk '{print $NF}')"
        else
            echo -e "  ${YELLOW}✗${NC} conda not installed"
        fi
        
        if command -v mamba &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} mamba $(mamba --version 2>&1 | awk '{print $NF}')"
        else
            echo -e "  ${YELLOW}✗${NC} mamba not installed"
        fi
        
        # Check existing environments
        echo ""
        echo "Existing environments:"
        echo ""
        
        if [ -d "zero-prototype-env" ]; then
            echo -e "  ${GREEN}✓${NC} uv environment: zero-prototype-env"
            echo "    Activate: source zero-prototype-env/bin/activate"
        fi
        
        if [ -d "venv" ]; then
            echo -e "  ${GREEN}✓${NC} venv environment: venv"
            echo "    Activate: source venv/bin/activate"
        fi
        
        if command -v conda &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} conda environments:"
            conda info --envs | tail -n +2 | grep -v "^#" | while read line; do
                echo "    $line"
            done
        fi
        ;;
    
    clean)
        echo -e "${YELLOW}⚠ Remove virtual environments?${NC}"
        echo ""
        echo "This will delete:"
        
        [ -d "zero-prototype-env" ] && echo "  - zero-prototype-env"
        [ -d "venv" ] && echo "  - venv"
        
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            [ -d "zero-prototype-env" ] && rm -rf zero-prototype-env && echo "  Deleted zero-prototype-env"
            [ -d "venv" ] && rm -rf venv && echo "  Deleted venv"
            echo -e "${GREEN}✓ Cleanup complete${NC}"
        else
            echo "Cancelled"
        fi
        ;;
    
    help|--help|-h)
        show_help
        ;;
    
    *)
        if [ -z "$COMMAND" ]; then
            echo -e "${BLUE}Zero-Shot Medical Image Captioning - Environment Manager${NC}"
            echo ""
            echo "Usage: source activate.sh [command]"
            echo ""
            echo "Most common commands:"
            echo "  ${GREEN}source activate.sh setup${NC}   - Interactive setup menu"
            echo "  ${GREEN}source activate.sh uv${NC}      - Quick uv setup"
            echo "  ${GREEN}source activate.sh check${NC}   - Check current environment"
            echo ""
            echo "See 'source activate.sh help' for all options"
        else
            echo -e "${YELLOW}Unknown command: $COMMAND${NC}"
            echo "Run: source activate.sh help"
        fi
        ;;
esac
