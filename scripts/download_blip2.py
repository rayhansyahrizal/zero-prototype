"""
Pre-download and cache BLIP2 model to avoid downloading on first use.
Run this script once before running the main pipeline.
"""

import logging
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_blip2_model(model_name: str = "Salesforce/blip2-opt-2.7b"):
    """
    Pre-download BLIP2 model to cache.
    
    Args:
        model_name: HuggingFace model identifier
    """
    logger.info("=" * 60)
    logger.info("BLIP2 MODEL DOWNLOAD")
    logger.info("=" * 60)
    
    try:
        logger.info(f"\nDownloading BLIP2 Processor: {model_name}")
        logger.info("This may take 1-2 minutes...")
        
        # Download processor
        processor = Blip2Processor.from_pretrained(model_name)
        logger.info("✓ Processor downloaded successfully")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        logger.info(f"\nDownloading BLIP2 Model: {model_name}")
        logger.info(f"Device: {device} | Data type: {torch_dtype}")
        logger.info("This may take 5-10 minutes (model is ~8GB)...")
        
        # Download model
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        )
        logger.info("✓ Model downloaded successfully")
        
        # Verify by moving to device
        model.to(device)
        logger.info(f"✓ Model loaded to {device}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ BLIP2 model successfully cached!")
        logger.info("You can now run the pipeline without internet connection.")
        logger.info("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Failed to download BLIP2 model: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check your internet connection")
        logger.error("2. Ensure you have enough disk space (~15GB)")
        logger.error("3. Try again or run the pipeline (it will download on first use)")
        return False


if __name__ == "__main__":
    success = download_blip2_model()
    sys.exit(0 if success else 1)
