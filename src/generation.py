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
        retrieved_captions: Optional[List[Dict[str, any]]] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> str:
        """
        Generate caption for a single image with optional retrieval context.

        Args:
            image: PIL Image
            context: Optional raw context text (deprecated - use retrieved_captions)
            retrieved_captions: List of retrieved caption dicts with 'caption' and 'similarity'
            max_length: Maximum caption length (used for max_new_tokens)
            num_beams: Number of beams for beam search
            similarity_threshold: Minimum similarity to include in context (filter low-quality retrievals)

        Returns:
            Generated caption string
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Set generation parameters
        max_new_tokens = max_length or self.blip2_config['max_length']
        num_beams = num_beams or self.blip2_config['num_beams']

        # Prepare inputs with tokenizer-based truncation
        if retrieved_captions:
            # NEW: Filter by similarity threshold
            if similarity_threshold is not None:
                filtered_captions = [
                    c for c in retrieved_captions
                    if c['similarity'] >= similarity_threshold
                ]

                # If all captions filtered out, fall back to baseline (no context)
                if not filtered_captions:
                    logger.debug(
                        f"All retrieved captions below threshold {similarity_threshold:.2f}, "
                        f"using baseline generation"
                    )
                    retrieved_captions = None
                else:
                    retrieved_captions = filtered_captions
                    logger.debug(
                        f"Filtered to {len(filtered_captions)}/{len(retrieved_captions)} "
                        f"captions above threshold {similarity_threshold:.2f}"
                    )

        if retrieved_captions:
            # Construct prompt following thesis methodology
            context_lines = [f"- {c['caption']}" for c in retrieved_captions]
            prompt = (
                "Based on the context of similar medical images:\n" +
                "\n".join(context_lines) +
                "\nDescribe the current image in clinical terms:"
            )
            
            # Use BLIP2 tokenizer to truncate by tokens (not words)
            # Max 150 tokens for context to leave room for image features
            MAX_PROMPT_TOKENS = 150
            
            # Tokenize and truncate
            prompt_tokens = self.processor.tokenizer(
                prompt,
                truncation=True,
                max_length=MAX_PROMPT_TOKENS,
                return_tensors="pt"
            )
            
            # Decode back to get truncated prompt text
            truncated_prompt = self.processor.tokenizer.decode(
                prompt_tokens['input_ids'][0],
                skip_special_tokens=True
            )
            
            inputs = self.processor(
                images=image,
                text=truncated_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            logger.debug(f"Prompt length: {len(prompt_tokens['input_ids'][0])} tokens")
            
        elif context:
            # Backward compatibility: use raw context string
            # Still apply tokenizer-based truncation
            prompt = f"Context: {context}\n\nBased on the context, describe this medical image:"
            
            prompt_tokens = self.processor.tokenizer(
                prompt,
                truncation=True,
                max_length=150,
                return_tensors="pt"
            )
            
            truncated_prompt = self.processor.tokenizer.decode(
                prompt_tokens['input_ids'][0],
                skip_special_tokens=True
            )
            
            inputs = self.processor(
                images=image,
                text=truncated_prompt,
                return_tensors="pt"
            ).to(self.device)
        else:
            # Baseline generation (no context)
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
        # IMPORTANT: Only decode the newly generated tokens, not the input prompt
        # generated_ids contains [prompt tokens + new tokens], we only want new tokens
        if retrieved_captions or context:
            # When using context, skip the prompt tokens
            prompt_length = inputs.input_ids.shape[1]
            generated_tokens_only = generated_ids[:, prompt_length:]
            caption = self.processor.batch_decode(
                generated_tokens_only, skip_special_tokens=True
            )[0].strip()
        else:
            # Baseline mode (no prompt), decode everything
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
        use_prototypes: bool = False,
        use_tta: bool = False,
        tta_params: Optional[Dict] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, str]]:
        """
        Generate captions with retrieval-augmented context.

        Follows thesis methodology:
        1. (Optional) Apply TTA to adapt query embedding
        2. Retrieve top-k similar captions using adapted embedding
        3. Construct prompt with retrieved captions
        4. Generate caption with BLIP2

        Args:
            images: List of PIL Images
            image_ids: List of image IDs
            retriever: CaptionRetriever or PrototypeRetriever instance
            use_prototypes: Whether using prototype retriever
            use_tta: Whether to use Test-Time Adaptation (only for PrototypeRetriever)
            tta_params: TTA parameters (learning_rate, num_steps, etc.)
            similarity_threshold: Minimum similarity to include retrieved captions in context

        Returns:
            List of dictionaries with image_id, caption, and retrieved context
        """
        method = "prototype" if use_prototypes else "retrieval"
        if use_tta:
            method += "+TTA"
        
        logger.info(f"Generating captions with {method}...")
        
        results = []
        
        # Retrieve for all images
        # Note: TTA is applied inside retrieve_by_image_embedding if use_tta=True
        if use_tta:
            logger.info("⚙️  Menjalankan adaptasi 1D test-time (scaling vector)...")
            logger.info(f"   - Learning rate: {tta_params.get('learning_rate', 1e-3)}")
            logger.info(f"   - Iterasi: {tta_params.get('num_steps', 10)} steps")

        retrieved_contexts = retriever.batch_retrieve(use_image_similarity=True)
        
        for idx, (image_id, image) in enumerate(tqdm(
            zip(image_ids, images),
            total=len(images),
            desc=f"{method.capitalize()} generation"
        )):
            # Get retrieved captions
            if idx in retrieved_contexts:
                retrieved = retrieved_contexts[idx]
                # Format as plain text for logging
                context_str = retriever.format_retrieved_context(retrieved)

                # Log retrieval context (hanya 3 sample pertama biar ga spam)
                if idx < 3:
                    logger.info(f"\n📋 [Retrieval] Image: {image_id}")
                    for i, r in enumerate(retrieved[:3], 1):  # Log top-3 aja
                        logger.info(f"   {i}. (sim={r['similarity']:.3f}) {r['caption'][:80]}...")
            else:
                retrieved = []
                context_str = ""

            # Generate caption with structured retrieved captions and similarity filtering
            caption = self.generate_caption(
                image,
                retrieved_captions=retrieved if retrieved else None,
                similarity_threshold=similarity_threshold
            )

            results.append({
                'image_id': image_id,
                'caption': caption,
                'method': method,
                'retrieved_context': context_str,
                'retrieved_captions': retrieved  # Keep structured data for evaluation
            })
        
        return results
    
    def generate_all_modes(
        self,
        images: List[Image.Image],
        image_ids: List[str],
        retriever=None,
        prototype_retriever=None,
        config: Optional[Dict] = None
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate captions using all three modes following thesis methodology.

        Modes:
        1. Baseline BLIP2 (no retrieval)
        2. BLIP2 + Retrieval (visual embedding similarity)
        3. BLIP2 + Prototype Sampling (with optional TTA)

        Args:
            images: List of PIL Images
            image_ids: List of image IDs
            retriever: CaptionRetriever instance (optional)
            prototype_retriever: PrototypeRetriever instance (optional)
            config: Configuration dict with TTA settings

        Returns:
            Dictionary with results for each mode
        """
        if self.model is None:
            self.load_model()

        results = {}

        # Extract similarity threshold from config
        similarity_threshold = None
        if config and 'retrieval' in config:
            similarity_threshold = config['retrieval'].get('similarity_threshold')
            if similarity_threshold:
                logger.info(f"Using similarity threshold: {similarity_threshold}")

        # 1. Baseline
        logger.info("\n=== Mode 1: Baseline BLIP2 (no retrieval) ===")
        results['baseline'] = self.generate_baseline(images, image_ids)

        # 2. Retrieval-augmented (visual embedding similarity)
        if retriever is not None:
            logger.info("\n=== Mode 2: BLIP2 + Retrieval (visual embedding similarity) ===")
            results['retrieval'] = self.generate_with_retrieval(
                images, image_ids, retriever,
                use_prototypes=False,
                use_tta=False,
                similarity_threshold=similarity_threshold
            )
        else:
            logger.warning("Retriever not provided, skipping retrieval mode")
        
        # 3. Prototype-augmented (with optional TTA)
        if prototype_retriever is not None:
            # Check if TTA is enabled
            use_tta = False
            tta_params = None
            
            if config and 'tta' in config:
                tta_config = config['tta']
                use_tta = tta_config.get('enabled', False)
                
                if use_tta:
                    tta_params = {
                        'learning_rate': tta_config.get('learning_rate', 1e-3),
                        'num_steps': tta_config.get('num_steps', 10),
                        'top_k_for_loss': tta_config.get('top_k_for_loss', 2),
                        'weight_variance': tta_config.get('weight_variance', 0.1),
                        'weight_entropy': tta_config.get('weight_entropy', 0.01)
                    }
                    logger.info(f"TTA enabled with params: {tta_params}")
            
            mode_name = "Mode 3: BLIP2 + Prototype Sampling"
            if use_tta:
                mode_name += " + TTA"
            
            logger.info(f"\n=== {mode_name} ===")
            results['prototype'] = self.generate_with_retrieval(
                images, image_ids, prototype_retriever,
                use_prototypes=True,
                use_tta=use_tta,
                tta_params=tta_params,
                similarity_threshold=similarity_threshold
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
