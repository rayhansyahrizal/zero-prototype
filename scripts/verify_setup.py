#!/usr/bin/env python3
"""
Quick setup verification script.
Checks if all components are ready to run.
"""

import sys
from pathlib import Path
import yaml

def check_item(name, condition, details=""):
    """Print check result."""
    status = "✓" if condition else "✗"
    print(f"{status} {name}")
    if details:
        print(f"  {details}")
    return condition

def main():
    print("=" * 60)
    print("Zero-Shot Medical Image Captioning - Setup Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print("1. Python Environment")
    python_ok = sys.version_info >= (3, 8)
    all_ok &= check_item(
        "Python version >= 3.8",
        python_ok,
        f"Current: {sys.version.split()[0]}"
    )
    print()
    
    # Check dependencies
    print("2. Core Dependencies")
    deps_ok = True
    
    try:
        import torch
        torch_ok = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        deps_ok &= check_item(
            "PyTorch",
            torch_ok,
            f"Version: {torch_version}, CUDA: {cuda_available}"
        )
    except ImportError:
        deps_ok &= check_item("PyTorch", False, "Not installed - run: pip install torch")
    
    try:
        import transformers
        deps_ok &= check_item(
            "Transformers",
            True,
            f"Version: {transformers.__version__}"
        )
    except ImportError:
        deps_ok &= check_item("Transformers", False, "Not installed")
    
    try:
        import gradio
        deps_ok &= check_item("Gradio", True, f"Version: {gradio.__version__}")
    except ImportError:
        deps_ok &= check_item("Gradio", False, "Not installed")
    
    try:
        import nltk
        deps_ok &= check_item("NLTK", True, "Installed")
    except ImportError:
        deps_ok &= check_item("NLTK", False, "Not installed")
    
    all_ok &= deps_ok
    print()
    
    # Check config
    print("3. Configuration")
    config_path = Path("config.yaml")
    config_ok = config_path.exists()
    all_ok &= check_item("config.yaml", config_ok)
    
    if config_ok:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    print()
    
    # Check dataset
    print("4. Dataset (ROCO)")
    if config_ok:
        dataset_root = Path(config['dataset']['root_dir'])
        split = config['dataset']['split']
        modality = config['dataset']['modality']
        
        captions_file = dataset_root / "data" / split / modality / "captions.txt"
        images_dir = dataset_root / "data" / split / modality / "images"
        
        captions_ok = captions_file.exists()
        all_ok &= check_item(
            "Captions file",
            captions_ok,
            str(captions_file) if captions_ok else "Not found"
        )
        
        images_ok = images_dir.exists()
        if images_ok:
            num_images = len(list(images_dir.glob("*.*")))
            all_ok &= check_item(
                "Images directory",
                num_images > 0,
                f"{num_images} images found"
            )
        else:
            all_ok &= check_item("Images directory", False, "Not found")
    print()
    
    # Check MedImageInsight
    print("5. MedImageInsight Model")
    if config_ok:
        model_dir = Path(config['medimageinsight']['model_dir'])
        vision_model = model_dir / "vision_model" / config['medimageinsight']['vision_model_name']
        tokenizer = model_dir / "language_model" / "clip_tokenizer_4.16.2"
        
        all_ok &= check_item(
            "Vision model",
            vision_model.exists(),
            str(vision_model)
        )
        all_ok &= check_item(
            "Tokenizer",
            tokenizer.exists(),
            str(tokenizer)
        )
    print()
    
    # Check project structure
    print("6. Project Structure")
    all_ok &= check_item("src/ directory", Path("src").exists())
    all_ok &= check_item("ui/ directory", Path("ui").exists())
    all_ok &= check_item("data/ directory", Path("data").exists())
    
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
        check_item("results/ directory", True, "Created")
    else:
        check_item("results/ directory", True)
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ All checks passed! Ready to run pipeline.")
        print()
        print("Next steps:")
        print("  1. Run pipeline: python src/main.py")
        print("  2. Or launch UI: python ui/app.py")
    else:
        print("✗ Some checks failed. Please fix issues above.")
        print()
        print("Common fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Wait for dataset download to complete")
        print("  - Check paths in config.yaml")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
