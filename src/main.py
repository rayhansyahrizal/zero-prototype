"""
Main pipeline for zero-shot medical image captioning.

Orchestrates the entire workflow:
1. Load dataset
2. Generate/load embeddings
3. Sample prototypes
4. Generate captions (baseline, retrieval, prototype)
5. Evaluate results
"""

import sys
from pathlib import Path
import yaml
import logging
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ROCODataLoader, check_dataset_availability
from huggingface_loader import HuggingFaceDataLoader
from embedding import EmbeddingGenerator
from sampling import sample_and_save_prototypes
from retrieval import CaptionRetriever, PrototypeRetriever
from generation import CaptionGenerator, check_blip2_availability
from evaluation import CaptionEvaluator, save_results, compare_methods

# Configure logging
def setup_logging(config: dict, timestamp: str = None):
    """Setup logging configuration with timestamp.
    
    Args:
        config: Configuration dictionary
        timestamp: Optional timestamp string for log file naming
        
    Returns:
        timestamp: The timestamp used for this run
    """
    from datetime import datetime
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler if enabled
    if log_config.get('save_logs', False):
        log_dir = Path(log_config.get('log_file', 'results/pipeline.log')).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamped log file
        log_file = log_dir / f"pipeline_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")
    
    return timestamp


def check_environment(config: dict) -> bool:
    """
    Check if all required resources are available.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if all resources available, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("ENVIRONMENT CHECK")
    logger.info("=" * 60)
    
    all_available = True
    
    # Check dataset
    logger.info("\n1. Checking dataset...")
    dataset_source = config['dataset'].get('source', 'local')
    
    if dataset_source == 'huggingface':
        # HF dataset always available if internet connection works
        dataset_name = config['dataset'].get('name', 'eltorio/ROCOv2-radiology')
        logger.info(f"   Using Hugging Face dataset: {dataset_name}")
        logger.info(f"   Will download/cache on first use")
    else:
        # Check local dataset
        dataset_available, message = check_dataset_availability(config)
        logger.info(f"   {message}")
        if not dataset_available:
            logger.warning(f"   Local dataset not ready. Consider using 'source: huggingface' in config")
            all_available = False
    
    # Check MedImageInsight model
    logger.info("\n2. Checking MedImageInsight model...")
    model_dir = Path(config['medimageinsight']['model_dir'])
    vision_model = model_dir / "vision_model" / config['medimageinsight']['vision_model_name']
    
    if vision_model.exists():
        logger.info(f"   MedImageInsight model found at {model_dir}")
    else:
        logger.error(f"   MedImageInsight model NOT found at {vision_model}")
        all_available = False
    
    # Check BLIP2 model
    logger.info("\n3. Checking BLIP2 model...")
    blip2_available, message = check_blip2_availability(config)
    logger.info(f"   {message}")
    if not blip2_available:
        logger.warning("   BLIP2 will be downloaded on first use (requires internet)")
    
    logger.info("\n" + "=" * 60)
    
    return all_available


def run_pipeline(
    config: dict,
    force_regenerate: bool = False,
    skip_embedding: bool = False,
    skip_generation: bool = False,
    timestamp: str = None
):
    """
    Run the complete pipeline.
    
    Args:
        config: Configuration dictionary
        force_regenerate: Force regeneration of cached data
        skip_embedding: Skip embedding generation (use cached)
        skip_generation: Skip caption generation (use cached)
        timestamp: Timestamp for this run (passed from setup_logging)
    """
    logger = logging.getLogger(__name__)
    
    start_time = datetime.now()
    
    logger.info("\n" + "=" * 60)
    logger.info("ZERO-SHOT MEDICAL IMAGE CAPTIONING PIPELINE")
    logger.info("=" * 60)
    
    # ========== Step 1: Load Dataset ==========
    logger.info("\n[Step 1/6] Loading dataset...")
    
    dataset_source = config['dataset'].get('source', 'local')
    
    if dataset_source == 'huggingface':
        # Use Hugging Face dataset
        dataset_name = config['dataset'].get('name', 'eltorio/ROCOv2-radiology')
        logger.info(f"Using Hugging Face dataset: {dataset_name}")
        
        hf_loader = HuggingFaceDataLoader(
            dataset_name=dataset_name,
            split=config['dataset']['split'],
            max_samples=config['dataset'].get('max_samples'),
            stream=False
        )
        
        # Convert to standard format
        samples = []
        for sample in hf_loader.iterate(max_samples=config['dataset'].get('max_samples')):
            normalized = hf_loader.normalize_sample(sample)
            # Convert to format expected by pipeline
            samples.append({
                'image_id': normalized['image_id'],
                'image': hf_loader.load_image(normalized['image']),
                'caption': normalized['caption']
            })
        
        logger.info(f"Loaded {len(samples)} samples from Hugging Face")
        
        # Create a simple wrapper that matches ROCODataLoader interface
        class HFDataWrapper:
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                return {
                    'image_id': sample['image_id'],
                    'image_path': None,  # Image already loaded
                    'caption': sample['caption']
                }
            
            def get_all_samples(self):
                return [self[i] for i in range(len(self))]
            
            def load_image(self, image_path=None, sample_idx=None):
                if sample_idx is not None:
                    return self.samples[sample_idx]['image']
                # Fallback: return first image
                return self.samples[0]['image'] if self.samples else None
        
        data_loader = HFDataWrapper(samples)
    else:
        # Use local ROCO dataset
        logger.info(f"Using local dataset: {config['dataset']['root_dir']}")
        
        data_loader = ROCODataLoader(
            root_dir=config['dataset']['root_dir'],
            split=config['dataset']['split'],
            modality=config['dataset']['modality'],
            max_samples=config['dataset'].get('max_samples')
        )
    
    if len(data_loader) == 0:
        logger.error("No samples loaded. Cannot proceed.")
        return
    
    logger.info(f"Loaded {len(data_loader)} samples")
    
    # ========== Step 2: Generate Embeddings ==========
    if not skip_embedding:
        logger.info("\n[Step 2/6] Generating embeddings...")
        
        embedding_generator = EmbeddingGenerator(config)
        embeddings_data = embedding_generator.generate_and_save_embeddings(
            data_loader,
            force_regenerate=force_regenerate
        )
    else:
        logger.info("\n[Step 2/6] Loading cached embeddings...")
        import numpy as np
        
        embeddings_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
        if not embeddings_file.exists():
            logger.error(f"Embeddings file not found: {embeddings_file}")
            return
        
        data = np.load(embeddings_file, allow_pickle=True)
        embeddings_data = {
            'image_embeddings': data['image_embeddings'],
            'text_embeddings': data['text_embeddings'],
            'image_ids': data['image_ids'].tolist(),
            'captions': data['captions'].tolist()
        }
        logger.info(f"Loaded {len(embeddings_data['image_ids'])} embeddings")
    
    # ========== Step 3: Sample Prototypes ==========
    logger.info("\n[Step 3/6] Sampling prototypes...")
    
    prototype_indices = sample_and_save_prototypes(
        config,
        embeddings_data['image_embeddings'],
        force_regenerate=force_regenerate
    )
    
    # ========== Step 4: Setup Retrievers ==========
    logger.info("\n[Step 4/6] Setting up retrievers...")
    
    # Regular retriever
    retriever = CaptionRetriever(
        image_embeddings=embeddings_data['image_embeddings'],
        text_embeddings=embeddings_data['text_embeddings'],
        captions=embeddings_data['captions'],
        image_ids=embeddings_data['image_ids'],
        top_k=config['retrieval']['top_k']
    )
    
    # Prototype retriever
    prototype_retriever = PrototypeRetriever(
        image_embeddings=embeddings_data['image_embeddings'],
        text_embeddings=embeddings_data['text_embeddings'],
        captions=embeddings_data['captions'],
        image_ids=embeddings_data['image_ids'],
        prototype_indices=prototype_indices,
        top_k=config['retrieval']['top_k']
    )
    
    logger.info("Retrievers initialized")
    
    # ========== Step 5: Generate Captions ==========
    if not skip_generation:
        logger.info("\n[Step 5/6] Generating captions...")
        
        # Load images for generation
        logger.info("Loading images...")
        images = []
        image_ids = []
        
        for idx, sample in enumerate(data_loader.get_all_samples()):
            try:
                # Handle both local and HF datasets
                if dataset_source == 'huggingface':
                    # Image already loaded in HF wrapper
                    img = data_loader.load_image(sample_idx=idx)
                else:
                    # Load from path for local dataset
                    img = data_loader.load_image(sample['image_path'])
                
                images.append(img)
                image_ids.append(sample['image_id'])
            except Exception as e:
                logger.warning(f"Failed to load {sample['image_id']}: {e}")
        
        logger.info(f"Loaded {len(images)} images for generation")
        
        # Generate captions
        generator = CaptionGenerator(config)
        results = generator.generate_all_modes(
            images=images,
            image_ids=image_ids,
            retriever=retriever,
            prototype_retriever=prototype_retriever
        )
    else:
        logger.info("\n[Step 5/6] Loading cached generation results...")
        import json
        
        captions_file = Path(config['output']['captions_file'])
        if not captions_file.exists():
            logger.error(f"Captions file not found: {captions_file}")
            return
        
        with open(captions_file) as f:
            results = json.load(f)
        
        logger.info(f"Loaded results for {len(results)} methods")
    
    # ========== Step 6: Evaluate Results ==========
    logger.info("\n[Step 6/6] Evaluating results...")
    
    # Build ground truth dictionary
    ground_truths = {
        sample['image_id']: sample['caption']
        for sample in data_loader.get_all_samples()
    }
    
    # Evaluate
    evaluator = CaptionEvaluator()
    scores = evaluator.evaluate_results(results, ground_truths)
    
    # Save results with timestamp
    if config['evaluation'].get('save_results', True):
        save_results(results, scores, config, timestamp=timestamp)
    
    # Display comparison
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 60)
    
    comparison = compare_methods(scores)
    print("\n" + comparison.to_string())
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Pipeline completed in {duration:.1f} seconds")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-shot medical image captioning pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regeneration of cached embeddings and prototypes'
    )
    parser.add_argument(
        '--skip-embedding',
        action='store_true',
        help='Skip embedding generation (use cached)'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip caption generation (use cached)'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check environment, do not run pipeline'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['local', 'huggingface'],
        help='Dataset source (overrides config)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        help='Hugging Face dataset name (e.g., eltorio/ROCOv2-radiology)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override dataset settings from command line
    if args.dataset:
        config['dataset']['source'] = args.dataset
    if args.dataset_name:
        config['dataset']['name'] = args.dataset_name
    
    # Setup logging and get timestamp for this run
    timestamp = setup_logging(config)
    
    # Check environment
    if not check_environment(config):
        logging.error("\nEnvironment check failed. Please fix issues and try again.")
        if args.check_only:
            sys.exit(1)
        else:
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    if args.check_only:
        logging.info("\nEnvironment check complete. Exiting.")
        return
    
    # Run pipeline
    try:
        run_pipeline(
            config,
            force_regenerate=args.force_regenerate,
            skip_embedding=args.skip_embedding,
            skip_generation=args.skip_generation,
            timestamp=timestamp
        )
    except KeyboardInterrupt:
        logging.info("\n\nPipeline interrupted by user")
    except Exception as e:
        logging.error(f"\n\nPipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
