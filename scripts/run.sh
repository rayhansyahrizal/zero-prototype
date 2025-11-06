#!/bin/bash
# Quick run script for zero-shot medical image captioning pipeline

set -e

echo "================================================"
echo "Zero-Shot Medical Image Captioning Pipeline"
echo "================================================"
echo ""

# Function to show usage
usage() {
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  install        Install dependencies"
    echo "  check          Check environment and resources"
    echo "  pipeline       Run complete pipeline"
    echo "  ui             Launch Gradio UI"
    echo "  test           Run tests for individual modules"
    echo "  clean          Clean cached embeddings and results"
    echo ""
    echo "Examples:"
    echo "  ./run.sh install       # Install all dependencies"
    echo "  ./run.sh check         # Verify everything is ready"
    echo "  ./run.sh pipeline      # Run full pipeline"
    echo "  ./run.sh ui            # Launch interactive UI"
    exit 1
}

# Check if command provided
if [ $# -eq 0 ]; then
    usage
fi

COMMAND=$1

case $COMMAND in
    install)
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
        echo ""
        echo "✓ Installation complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Run './run.sh check' to verify environment"
        echo "  2. Run './run.sh pipeline' to execute pipeline"
        ;;
    
    check)
        echo "🔍 Checking environment..."
        echo ""
        python src/main.py --check-only
        ;;
    
    pipeline)
        echo "🚀 Running complete pipeline..."
        echo ""
        python src/main.py "$@"
        ;;
    
    ui)
        echo "🌐 Launching Gradio UI..."
        echo ""
        echo "The UI will open at http://localhost:7860"
        echo "Press Ctrl+C to stop"
        echo ""
        python ui/app.py
        ;;
    
    test)
        echo "🧪 Running module tests..."
        echo ""
        
        echo "Testing data loader..."
        python src/data_loader.py
        echo ""
        
        read -p "Test embedding generation? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Testing embeddings (10 samples)..."
            python src/embedding.py
        fi
        
        echo ""
        echo "✓ Tests complete"
        ;;
    
    clean)
        echo "🧹 Cleaning cached data..."
        read -p "This will remove embeddings and results. Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf data/embeddings/*
            rm -rf data/prototypes.npy
            rm -rf results/*
            echo "✓ Cleaned successfully"
        else
            echo "Cancelled"
        fi
        ;;
    
    help|--help|-h)
        usage
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        usage
        ;;
esac
