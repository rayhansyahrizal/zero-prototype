"""
Caption generation using BLIP2 model.
Supports three modes: baseline, retrieval-augmented, and prototype-augmented.
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import numpy as np
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Generate captions using BLIP2 model."""
    
    def __init__(self, config: dict):
        """
        Initialize caption generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.blip2_config = config['blip2']
        
        self.model = None
        self.processor = None
        self.device = torch.device(
            self.blip2_config['device'] 
            if torch.cuda.is_available() and self.blip2_config['device'] == 'cuda'
            else 'cpu'
        )
        
        logger.info(f"CaptionGenerator initialized on device: {self.device}")
    
    def load_model(self):
        """Load BLIP2 model and processor."""
        try:
            model_name = self.blip2_config['model_name']
            
            logger.info(f"Loading BLIP2 model: {model_name}")
            logger.info("This may take a while if model is not cached...")
            
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("BLIP2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP2 model: {e}")
            logger.error(
                f"Make sure the model '{model_name}' is available or cached. "
                "If not cached, it will be downloaded automatically (requires internet)."
            )
            raise
    
    def generate_caption(
        self,
        image: Image.Image,
        context: Optional[str] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None
    ) -> str:
        """
        Generate caption for a single image.
        
        Args:
            image: PIL Image
            context: Optional context text to condition generation
            max_length: Maximum caption length (used for max_new_tokens)
            num_beams: Number of beams for beam search
            
        Returns:
            Generated caption string
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Set generation parameters
        max_new_tokens = max_length or self.blip2_config['max_length']
        num_beams = num_beams or self.blip2_config['num_beams']
        
        # Prepare inputs
        if context:
            # Retrieval-augmented generation
            # Truncate context to avoid excessive input length
            # Keep only ~150 tokens of context to leave room for image features
            max_context_length = 150
            context_words = context.split()
            if len(context_words) > max_context_length:
                context = ' '.join(context_words[:max_context_length]) + "..."
            
            prompt = f"Context: {context}\n\nBased on the context, describe this medical image:"
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
        else:
            # Baseline generation
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
        
        # Generate caption
        # Use max_new_tokens instead of max_length to only limit generated tokens
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode caption
        caption = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        
        return caption
    
    def generate_baseline(
        self,
        images: List[Image.Image],
        image_ids: List[str]
    ) -> List[Dict[str, str]]:
        """
        Generate captions using baseline BLIP2 (no retrieval).
        
        Args:
            images: List of PIL Images
            image_ids: List of image IDs
            
        Returns:
            List of dictionaries with image_id and caption
        """
        logger.info("Generating baseline captions...")
        
        results = []
        for image_id, image in tqdm(
            zip(image_ids, images), 
            total=len(images),
            desc="Baseline generation"
        ):
            caption = self.generate_caption(image)
            results.append({
                'image_id': image_id,
                'caption': caption,
                'method': 'baseline'
            })
        
        return results
    
    def generate_with_retrieval(
        self,
        images: List[Image.Image],
        image_ids: List[str],
        retriever,
        use_prototypes: bool = False
    ) -> List[Dict[str, str]]:
        """
        Generate captions with retrieval-augmented context.
        
        Args:
            images: List of PIL Images
            image_ids: List of image IDs
            retriever: CaptionRetriever or PrototypeRetriever instance
            use_prototypes: Whether using prototype retriever
            
        Returns:
            List of dictionaries with image_id, caption, and retrieved context
        """
        method = "prototype" if use_prototypes else "retrieval"
        logger.info(f"Generating captions with {method}...")
        
        results = []
        
        # Retrieve for all images
        retrieved_contexts = retriever.batch_retrieve(use_image_similarity=True)
        
        for idx, (image_id, image) in enumerate(tqdm(
            zip(image_ids, images),
            total=len(images),
            desc=f"{method.capitalize()} generation"
        )):
            # Get retrieved captions
            if idx in retrieved_contexts:
                retrieved = retrieved_contexts[idx]
                context = retriever.format_retrieved_context(retrieved)
            else:
                context = ""
            
            # Generate caption with context
            caption = self.generate_caption(image, context=context)
            
            results.append({
                'image_id': image_id,
                'caption': caption,
                'method': method,
                'retrieved_context': context
            })
        
        return results
    
    def generate_all_modes(
        self,
        images: List[Image.Image],
        image_ids: List[str],
        retriever=None,
        prototype_retriever=None
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate captions using all three modes.
        
        Args:
            images: List of PIL Images
            image_ids: List of image IDs
            retriever: CaptionRetriever instance (optional)
            prototype_retriever: PrototypeRetriever instance (optional)
            
        Returns:
            Dictionary with results for each mode
        """
        if self.model is None:
            self.load_model()
        
        results = {}
        
        # 1. Baseline
        logger.info("\n=== Mode 1: Baseline BLIP2 ===")
        results['baseline'] = self.generate_baseline(images, image_ids)
        
        # 2. Retrieval-augmented
        if retriever is not None:
            logger.info("\n=== Mode 2: BLIP2 + Retrieval ===")
            results['retrieval'] = self.generate_with_retrieval(
                images, image_ids, retriever, use_prototypes=False
            )
        else:
            logger.warning("Retriever not provided, skipping retrieval mode")
        
        # 3. Prototype-augmented
        if prototype_retriever is not None:
            logger.info("\n=== Mode 3: BLIP2 + Prototype Sampling ===")
            results['prototype'] = self.generate_with_retrieval(
                images, image_ids, prototype_retriever, use_prototypes=True
            )
        else:
            logger.warning("Prototype retriever not provided, skipping prototype mode")
        
        return results


def check_blip2_availability(config: dict) -> tuple:
    """
    Check if BLIP2 model is available (cached).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_available, message)
    """
    model_name = config['blip2']['model_name']
    
    try:
        from transformers import Blip2Processor
        
        # Try to load processor (lightweight check)
        processor = Blip2Processor.from_pretrained(
            model_name,
            local_files_only=True  # Only use cached files
        )
        
        return True, f"BLIP2 model '{model_name}' is cached and ready"
        
    except Exception as e:
        return False, (
            f"BLIP2 model '{model_name}' not cached. "
            "It will be downloaded on first use (requires internet connection)."
        )


if __name__ == "__main__":
    # Test caption generation
    import yaml
    from pathlib import Path
    from data_loader import ROCODataLoader, check_dataset_availability
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check BLIP2
    available, message = check_blip2_availability(config)
    print(f"\nBLIP2 status: {message}\n")
    
    # Check dataset
    available, message = check_dataset_availability(config)
    if not available:
        print(f"Dataset not available: {message}")
        exit(1)
    
    # Load a small test set
    loader = ROCODataLoader(
        root_dir=config['dataset']['root_dir'],
        split=config['dataset']['split'],
        modality=config['dataset']['modality'],
        max_samples=3
    )
    
    if len(loader) == 0:
        print("No images available for testing")
        exit(1)
    
    # Load images
    images = []
    image_ids = []
    for sample in loader.get_all_samples():
        images.append(loader.load_image(sample['image_path']))
        image_ids.append(sample['image_id'])
    
    # Test baseline generation
    generator = CaptionGenerator(config)
    generator.load_model()
    
    print("\n=== Testing Baseline Generation ===")
    for i, (img_id, img) in enumerate(zip(image_ids, images)):
        print(f"\nImage {i+1}: {img_id}")
        print(f"Ground truth: {loader[i]['caption']}")
        
        caption = generator.generate_caption(img)
        print(f"Generated: {caption}")
