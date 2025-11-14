"""
Enhanced pipeline for TTA experimentation.

This pipeline extends main.py with:
1. Pre-TTA vs Post-TTA comparison
2. TTA convergence tracking
3. Delta performance metrics
4. Detailed ablation support

Following the experimental design from the thesis document.
"""

import sys
from pathlib import Path
import yaml
import logging
import argparse
from datetime import datetime
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ROCODataLoader, check_dataset_availability
from huggingface_loader import HuggingFaceDataLoader
from embedding import EmbeddingGenerator
from sampling import sample_and_save_prototypes
from retrieval import CaptionRetriever, PrototypeRetriever
from generation import CaptionGenerator, check_blip2_availability
from evaluation import CaptionEvaluator, save_results, compare_methods
from tta_analyzer import TTAAnalyzer, TTAMetrics

logger = logging.getLogger(__name__)


def setup_logging(config: dict, timestamp: str = None):
    """Setup logging configuration with timestamp."""
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    if log_config.get('save_logs', False):
        log_dir = Path(log_config.get('log_file', 'results/pipeline.log')).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"pipeline_tta_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Logging to file: {log_file}")

    return timestamp


def load_dataset(config: dict):
    """Load dataset from HuggingFace or local."""
    logger = logging.getLogger(__name__)

    dataset_source = config['dataset'].get('source', 'local')

    if dataset_source == 'huggingface':
        dataset_name = config['dataset'].get('name', 'eltorio/ROCOv2-radiology')
        logger.info(f"Using Hugging Face dataset: {dataset_name}")

        hf_loader = HuggingFaceDataLoader(
            dataset_name=dataset_name,
            split=config['dataset']['split'],
            max_samples=config['dataset'].get('max_samples'),
            stream=False
        )

        samples = []
        for sample in hf_loader.iterate(max_samples=config['dataset'].get('max_samples')):
            normalized = hf_loader.normalize_sample(sample)
            samples.append({
                'image_id': normalized['image_id'],
                'image': hf_loader.load_image(normalized['image']),
                'caption': normalized['caption'],
                'modality': normalized.get('modality', 'UNKNOWN')
            })

        logger.info(f"✅ Loaded {len(samples)} samples from HuggingFace")

        class HFDataWrapper:
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]
                return {
                    'image_id': sample['image_id'],
                    'image_path': None,
                    'caption': sample['caption'],
                    'modality': sample.get('modality', 'UNKNOWN')
                }

            def get_all_samples(self):
                return [self[i] for i in range(len(self))]

            def load_image(self, image_path=None, sample_idx=None):
                if sample_idx is not None:
                    return self.samples[sample_idx]['image']
                return self.samples[0]['image'] if self.samples else None

        data_loader = HFDataWrapper(samples)
    else:
        logger.info(f"Using local dataset: {config['dataset']['root_dir']}")
        data_loader = ROCODataLoader(
            root_dir=config['dataset']['root_dir'],
            split=config['dataset']['split'],
            modality=config['dataset']['modality'],
            max_samples=config['dataset'].get('max_samples')
        )

    return data_loader, dataset_source


def run_tta_experiment(
    config: dict,
    force_regenerate: bool = False,
    skip_embedding: bool = False,
    timestamp: str = None
):
    """
    Run TTA experiment with pre/post comparison.

    This function runs:
    1. Baseline BLIP2 (no retrieval)
    2. Retrieval (no TTA)
    3. Prototype without TTA (PRE-TTA)
    4. Prototype with TTA (POST-TTA)

    Then computes:
    - Delta performance (POST-TTA - PRE-TTA)
    - TTA convergence metrics
    - Per-sample improvements
    """
    logger.info("\n" + "=" * 60)
    logger.info("🧪 TTA EXPERIMENT PIPELINE")
    logger.info("=" * 60)

    start_time = datetime.now()

    # ========== Step 1: Load Dataset ==========
    logger.info("\n[Step 1/7] Loading dataset...")
    data_loader, dataset_source = load_dataset(config)

    if len(data_loader) == 0:
        logger.error("❌ No samples loaded. Cannot continue.")
        return

    logger.info(f"✅ Loaded {len(data_loader)} samples")

    # ========== Step 2: Generate Embeddings ==========
    if not skip_embedding:
        logger.info("\n[Step 2/7] Generating embeddings with MedImageInsight...")
        embedding_generator = EmbeddingGenerator(config)
        embeddings_data = embedding_generator.generate_and_save_embeddings(
            data_loader,
            force_regenerate=force_regenerate
        )
    else:
        logger.info("\n[Step 2/7] Loading embeddings from cache...")
        import numpy as np

        embeddings_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
        if not embeddings_file.exists():
            logger.error(f"❌ Embeddings file not found: {embeddings_file}")
            return

        data = np.load(embeddings_file, allow_pickle=True)
        embeddings_data = {
            'image_embeddings': data['image_embeddings'],
            'text_embeddings': data['text_embeddings'],
            'image_ids': data['image_ids'].tolist(),
            'captions': data['captions'].tolist()
        }

        if 'modalities' in data:
            embeddings_data['modalities'] = data['modalities'].tolist()
        else:
            from data_loader import extract_modality
            embeddings_data['modalities'] = [
                extract_modality(cap, img_id)
                for cap, img_id in zip(embeddings_data['captions'], embeddings_data['image_ids'])
            ]

        logger.info(f"✅ Loaded {len(embeddings_data['image_ids'])} embeddings")

    # ========== Step 3: Sample Prototypes ==========
    logger.info("\n[Step 3/7] Sampling prototypes with FPS...")
    prototype_indices = sample_and_save_prototypes(
        config,
        embeddings_data['image_embeddings'],
        force_regenerate=force_regenerate
    )

    # ========== Step 4: Setup Retrievers ==========
    logger.info("\n[Step 4/7] Setting up retrievers...")

    # Regular retriever
    retriever = CaptionRetriever(
        image_embeddings=embeddings_data['image_embeddings'],
        text_embeddings=embeddings_data['text_embeddings'],
        captions=embeddings_data['captions'],
        image_ids=embeddings_data['image_ids'],
        modalities=embeddings_data.get('modalities'),
        top_k=config['retrieval']['top_k']
    )

    # Prototype retriever
    prototype_retriever = PrototypeRetriever(
        image_embeddings=embeddings_data['image_embeddings'],
        text_embeddings=embeddings_data['text_embeddings'],
        captions=embeddings_data['captions'],
        image_ids=embeddings_data['image_ids'],
        modalities=embeddings_data.get('modalities'),
        prototype_indices=prototype_indices,
        top_k=config['retrieval']['top_k']
    )

    logger.info("✅ Retrievers ready")

    # ========== Step 5: Load Images ==========
    logger.info("\n[Step 5/7] Loading images for generation...")
    images = []
    image_ids = []

    for idx, sample in enumerate(data_loader.get_all_samples()):
        try:
            if dataset_source == 'huggingface':
                img = data_loader.load_image(sample_idx=idx)
            else:
                img = data_loader.load_image(sample['image_path'])

            images.append(img)
            image_ids.append(sample['image_id'])
        except Exception as e:
            logger.warning(f"Failed to load {sample['image_id']}: {e}")

    logger.info(f"✅ Loaded {len(images)} images")

    # ========== Step 6: Generate Captions (WITH TTA COMPARISON) ==========
    logger.info("\n[Step 6/7] Generating captions with TTA comparison...")

    generator = CaptionGenerator(config)
    generator.load_model()

    # Initialize TTA analyzer
    tta_analyzer = TTAAnalyzer(
        save_dir=Path(config['output']['results_dir']) / "tta_analysis"
    )

    results = {}
    tta_metrics_all = []

    # Mode 1: Baseline
    logger.info("\n=== Mode 1: Baseline BLIP2 ===")
    results['baseline'] = generator.generate_baseline(images, image_ids)

    # Mode 2: Retrieval (no prototype)
    logger.info("\n=== Mode 2: Retrieval (no TTA) ===")
    results['retrieval'] = generator.generate_with_retrieval(
        images, image_ids, retriever,
        use_prototypes=False,
        use_tta=False
    )

    # Mode 3: Prototype WITHOUT TTA (PRE-TTA baseline)
    logger.info("\n=== Mode 3: Prototype WITHOUT TTA ===")
    temp_tta_enabled = config['tta']['enabled']
    config['tta']['enabled'] = False  # Temporarily disable

    results['prototype_no_tta'] = generator.generate_with_retrieval(
        images, image_ids, prototype_retriever,
        use_prototypes=True,
        use_tta=False
    )

    config['tta']['enabled'] = temp_tta_enabled  # Restore

    # Mode 4: Prototype WITH TTA (POST-TTA)
    logger.info("\n=== Mode 4: Prototype WITH TTA ===")

    tta_config = config['tta']
    tta_params = {
        'learning_rate': tta_config.get('learning_rate', 1e-3),
        'num_steps': tta_config.get('num_steps', 10),
        'top_k_for_loss': tta_config.get('top_k_for_loss', 2),
        'weight_variance': tta_config.get('weight_variance', 0.1),
        'weight_entropy': tta_config.get('weight_entropy', 0.01)
    }

    logger.info(f"TTA parameters: {tta_params}")

    results['prototype_with_tta'] = generator.generate_with_retrieval(
        images, image_ids, prototype_retriever,
        use_prototypes=True,
        use_tta=True,
        tta_params=tta_params
    )

    # Collect TTA metrics (if available from retriever)
    # Note: We'll enhance the retriever to track these

    logger.info(f"✅ Generated captions for {len(results)} modes")

    # ========== Step 7: Evaluate with Delta Analysis ==========
    logger.info("\n[Step 7/7] Evaluating with delta analysis...")

    # Build ground truth
    ground_truths = {}
    modality_map = {}
    for sample in data_loader.get_all_samples():
        image_id = sample['image_id']
        ground_truths[image_id] = sample['caption']
        modality_map[image_id] = sample.get('modality', 'UNKNOWN')

    # Load text encoder for semantic similarity
    logger.info("Loading text encoder for semantic similarity...")
    text_embedding_model = None
    try:
        embedding_generator_for_eval = EmbeddingGenerator(config)
        embedding_generator_for_eval.load_model()
        text_embedding_model = embedding_generator_for_eval.model
        logger.info("✅ Text encoder ready")
    except Exception as e:
        logger.warning(f"Could not load text encoder: {e}")

    # Evaluate all modes
    evaluator = CaptionEvaluator(
        text_embedding_model=text_embedding_model,
        use_semantic_similarity=True
    )

    scores, detailed_scores = evaluator.evaluate_results(
        results,
        ground_truths,
        modalities=modality_map
    )

    # ========== Compute Delta Metrics ==========
    logger.info("\n" + "=" * 60)
    logger.info("📊 DELTA ANALYSIS (TTA Impact)")
    logger.info("=" * 60)

    if 'prototype_no_tta' in scores and 'prototype_with_tta' in scores:
        delta_metrics = {}

        pre_tta = scores['prototype_no_tta']
        post_tta = scores['prototype_with_tta']

        for metric in pre_tta.keys():
            delta = post_tta[metric] - pre_tta[metric]
            delta_pct = (delta / pre_tta[metric] * 100) if pre_tta[metric] != 0 else 0
            delta_metrics[metric] = {
                'pre_tta': pre_tta[metric],
                'post_tta': post_tta[metric],
                'delta': delta,
                'delta_pct': delta_pct
            }

            logger.info(f"\n{metric.upper()}:")
            logger.info(f"  PRE-TTA:  {pre_tta[metric]:.4f}")
            logger.info(f"  POST-TTA: {post_tta[metric]:.4f}")
            logger.info(f"  Δ:        {delta:+.4f} ({delta_pct:+.2f}%)")

        # Save delta metrics
        delta_file = Path(config['output']['results_dir']) / f"delta_metrics_{timestamp}.json"
        with open(delta_file, 'w') as f:
            json.dump(delta_metrics, f, indent=2)
        logger.info(f"\n💾 Delta metrics saved to: {delta_file}")

    # ========== Save Results ==========
    if config['evaluation'].get('save_results', True):
        save_results(
            results,
            scores,
            config,
            timestamp=timestamp,
            detailed_scores=detailed_scores
        )

    # ========== Display Comparison ==========
    logger.info("\n" + "=" * 60)
    logger.info("📊 METHOD COMPARISON")
    logger.info("=" * 60)

    comparison = compare_methods(scores)
    print("\n" + comparison.to_string())

    comparison_csv = Path(config['output']['results_dir']) / f"comparison_{timestamp}.csv"
    comparison.to_csv(comparison_csv, float_format='%.4f')
    logger.info(f"\n💾 Comparison table saved to: {comparison_csv}")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 60)
    logger.info(f"✅ TTA Experiment complete in {duration:.1f} seconds")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TTA experiment pipeline for medical image captioning"
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
        '--dataset',
        type=str,
        choices=['local', 'huggingface'],
        help='Dataset source (overrides config)'
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override dataset settings
    if args.dataset:
        config['dataset']['source'] = args.dataset

    # Setup logging
    timestamp = setup_logging(config)

    # Run experiment
    try:
        run_tta_experiment(
            config,
            force_regenerate=args.force_regenerate,
            skip_embedding=args.skip_embedding,
            timestamp=timestamp
        )
    except KeyboardInterrupt:
        logging.info("\n\nExperiment interrupted by user")
    except Exception as e:
        logging.error(f"\n\nExperiment failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
