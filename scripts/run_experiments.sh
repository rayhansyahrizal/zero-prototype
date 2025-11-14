#!/bin/bash
# Hyperparameter Sweep Experiments for TTA Improvements
# Run this script to systematically test all improvement configurations

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "  TTA IMPROVEMENT EXPERIMENTS"
echo "=========================================="
echo ""
echo "This script will run 4 experiments:"
echo "  1. Conservative TTA (small LR, more steps)"
echo "  2. Moderate TTA (balanced approach)"
echo "  3. Prototype WITHOUT TTA (ablation)"
echo "  4. More Prototypes (300 instead of 100)"
echo ""
echo "Each experiment takes ~30-60 minutes depending on GPU."
echo ""

# Function to run a single experiment
run_experiment() {
    local config_file=$1
    local experiment_name=$2

    echo ""
    echo "=========================================="
    echo "  EXPERIMENT: $experiment_name"
    echo "=========================================="
    echo "Config: $config_file"
    echo "Start time: $(date)"
    echo ""

    python src/main.py \
        --config "$config_file" \
        --force-regenerate

    echo ""
    echo "Completed: $experiment_name"
    echo "End time: $(date)"
    echo ""
}

# Ask for confirmation
read -p "Start experiments? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Experiments cancelled."
    exit 0
fi

# Record start time
START_TIME=$(date +%s)

# Run experiments
run_experiment "config_tta_conservative.yaml" "Conservative TTA"
run_experiment "config_tta_moderate.yaml" "Moderate TTA"
run_experiment "config_prototype_no_tta.yaml" "Prototype WITHOUT TTA"
run_experiment "config_more_prototypes.yaml" "More Prototypes (300)"

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo "  ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved in:"
echo "  - results/conservative/"
echo "  - results/moderate/"
echo "  - results/no_tta/"
echo "  - results/more_prototypes/"
echo ""
echo "Next step: Run analysis script to compare results"
echo "  python scripts/compare_experiments.py"
echo ""
