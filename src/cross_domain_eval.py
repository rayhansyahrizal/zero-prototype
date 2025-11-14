"""
Cross-Domain Evaluation Pipeline.

Supports evaluation across different medical imaging domains:
- ROCO → ROCO (in-domain)
- ROCO → MIMIC-CXR (cross-domain)
- ROCO → IU X-Ray (cross-domain)
- Modality-specific evaluation (CT → X-ray, etc.)

Following Section 1 (Setup Dataset untuk Domain Shift) from the thesis document.
"""

import sys
from pathlib import Path
import yaml
import logging
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ROCODataLoader, extract_modality
from huggingface_loader import HuggingFaceDataLoader
from embedding import EmbeddingGenerator
from sampling import sample_and_save_prototypes
from retrieval import CaptionRetriever, PrototypeRetriever
from generation import CaptionGenerator
from evaluation import CaptionEvaluator, save_results, compare_methods

logger = logging.getLogger(__name__)


class CrossDomainEvaluator:
    """Evaluator for cross-domain performance."""

    def __init__(self, config: dict):
        """
        Initialize cross-domain evaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}

    def load_domain_dataset(
        self,
        domain_config: Dict,
        split: str = "test"
    ):
        """
        Load dataset for a specific domain.

        Args:
            domain_config: Configuration for the domain dataset
            split: Dataset split to load

        Returns:
            Data loader and samples
        """
        dataset_type = domain_config.get('type', 'huggingface')

        if dataset_type == 'huggingface':
            dataset_name = domain_config['name']
            logger.info(f"Loading HuggingFace dataset: {dataset_name}")

            hf_loader = HuggingFaceDataLoader(
                dataset_name=dataset_name,
                split=split,
                max_samples=domain_config.get('max_samples'),
                stream=False
            )

            samples = []
            for sample in hf_loader.iterate(max_samples=domain_config.get('max_samples')):
                normalized = hf_loader.normalize_sample(sample)
                samples.append({
                    'image_id': normalized['image_id'],
                    'image': hf_loader.load_image(normalized['image']),
                    'caption': normalized['caption'],
                    'modality': normalized.get('modality', 'UNKNOWN'),
                    'domain': domain_config.get('domain_name', dataset_name)
                })

            logger.info(f"✅ Loaded {len(samples)} samples from {dataset_name}")

            class DomainDataWrapper:
                def __init__(self, samples):
                    self.samples = samples

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    return self.samples[idx]

                def get_all_samples(self):
                    return self.samples

                def load_image(self, image_path=None, sample_idx=None):
                    if sample_idx is not None:
                        return self.samples[sample_idx]['image']
                    return self.samples[0]['image'] if self.samples else None

            return DomainDataWrapper(samples)

        elif dataset_type == 'local':
            logger.info(f"Loading local dataset: {domain_config['root_dir']}")

            loader = ROCODataLoader(
                root_dir=domain_config['root_dir'],
                split=split,
                modality=domain_config.get('modality', 'radiology'),
                max_samples=domain_config.get('max_samples')
            )

            # Add domain name to samples
            samples = loader.get_all_samples()
            for sample in samples:
                sample['domain'] = domain_config.get('domain_name', 'local')

            return loader

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def compute_domain_distance(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray
    ) -> float:
        """
        Compute domain distance using Maximum Mean Discrepancy (MMD).

        Args:
            source_embeddings: Source domain embeddings (N, D)
            target_embeddings: Target domain embeddings (M, D)

        Returns:
            Domain distance (MMD)
        """
        # Compute mean embeddings
        source_mean = np.mean(source_embeddings, axis=0)
        target_mean = np.mean(target_embeddings, axis=0)

        # L2 distance between means (simple MMD approximation)
        distance = np.linalg.norm(source_mean - target_mean)

        logger.info(f"Domain distance (MMD): {distance:.4f}")
        return float(distance)

    def evaluate_cross_domain(
        self,
        source_domain_config: Dict,
        target_domain_config: Dict,
        use_tta: bool = True,
        timestamp: str = None
    ) -> Dict:
        """
        Evaluate cross-domain performance.

        Workflow:
        1. Train on source domain (generate embeddings + prototypes)
        2. Test on target domain (evaluate with/without TTA)
        3. Compute domain distance and performance metrics

        Args:
            source_domain_config: Source domain configuration
            target_domain_config: Target domain configuration
            use_tta: Whether to use TTA for adaptation
            timestamp: Timestamp for saving results

        Returns:
            Dictionary with evaluation results
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("\n" + "=" * 70)
        logger.info("CROSS-DOMAIN EVALUATION")
        logger.info("=" * 70)
        logger.info(f"Source: {source_domain_config.get('domain_name', 'unknown')}")
        logger.info(f"Target: {target_domain_config.get('domain_name', 'unknown')}")
        logger.info(f"TTA: {'enabled' if use_tta else 'disabled'}")
        logger.info("=" * 70)

        # Step 1: Load source domain (training)
        logger.info("\n[1/6] Loading source domain (training)...")
        source_loader = self.load_domain_dataset(source_domain_config, split="train")
        logger.info(f"Source samples: {len(source_loader)}")

        # Step 2: Generate embeddings for source domain
        logger.info("\n[2/6] Generating embeddings for source domain...")
        embedding_generator = EmbeddingGenerator(self.config)

        source_embeddings_data = embedding_generator.generate_and_save_embeddings(
            source_loader,
            force_regenerate=False
        )

        # Step 3: Sample prototypes from source domain
        logger.info("\n[3/6] Sampling prototypes from source domain...")
        prototype_indices = sample_and_save_prototypes(
            self.config,
            source_embeddings_data['image_embeddings'],
            force_regenerate=False
        )

        # Create prototype retriever
        prototype_retriever = PrototypeRetriever(
            image_embeddings=source_embeddings_data['image_embeddings'],
            text_embeddings=source_embeddings_data['text_embeddings'],
            captions=source_embeddings_data['captions'],
            image_ids=source_embeddings_data['image_ids'],
            modalities=source_embeddings_data.get('modalities'),
            prototype_indices=prototype_indices,
            top_k=self.config['retrieval']['top_k']
        )

        # Step 4: Load target domain (testing)
        logger.info("\n[4/6] Loading target domain (testing)...")
        target_loader = self.load_domain_dataset(target_domain_config, split="test")
        logger.info(f"Target samples: {len(target_loader)}")

        # Load target images
        target_images = []
        target_image_ids = []
        target_ground_truths = {}

        for sample in target_loader.get_all_samples():
            target_images.append(sample['image'])
            target_image_ids.append(sample['image_id'])
            target_ground_truths[sample['image_id']] = sample['caption']

        # Step 5: Generate embeddings for target domain
        logger.info("\n[5/6] Generating embeddings for target domain...")
        target_embeddings_data = embedding_generator.generate_and_save_embeddings(
            target_loader,
            force_regenerate=False
        )

        # Compute domain distance
        domain_distance = self.compute_domain_distance(
            source_embeddings_data['image_embeddings'],
            target_embeddings_data['image_embeddings']
        )

        # Step 6: Evaluate on target domain
        logger.info("\n[6/6] Evaluating on target domain...")

        generator = CaptionGenerator(self.config)
        generator.load_model()

        # Generate captions with/without TTA
        results = {}

        # Baseline
        logger.info("\n=== Baseline (no retrieval) ===")
        results['baseline'] = generator.generate_baseline(target_images, target_image_ids)

        # Without TTA
        logger.info("\n=== Prototype without TTA ===")
        results['prototype_no_tta'] = generator.generate_with_retrieval(
            target_images,
            target_image_ids,
            prototype_retriever,
            use_prototypes=True,
            use_tta=False
        )

        # With TTA (if enabled)
        if use_tta:
            logger.info("\n=== Prototype with TTA ===")

            tta_params = {
                'learning_rate': self.config['tta'].get('learning_rate', 1e-3),
                'num_steps': self.config['tta'].get('num_steps', 10),
                'top_k_for_loss': self.config['tta'].get('top_k_for_loss', 2),
                'weight_variance': self.config['tta'].get('weight_variance', 0.1),
                'weight_entropy': self.config['tta'].get('weight_entropy', 0.01)
            }

            results['prototype_with_tta'] = generator.generate_with_retrieval(
                target_images,
                target_image_ids,
                prototype_retriever,
                use_prototypes=True,
                use_tta=True,
                tta_params=tta_params
            )

        # Evaluate
        logger.info("\n=== Evaluating results ===")

        # Load text encoder
        embedding_generator.load_model()
        text_embedding_model = embedding_generator.model

        evaluator = CaptionEvaluator(
            text_embedding_model=text_embedding_model,
            use_semantic_similarity=True
        )

        scores, detailed_scores = evaluator.evaluate_results(
            results,
            target_ground_truths
        )

        # Save results
        results_dir = Path(self.config['output']['results_dir']) / "cross_domain"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save cross-domain specific metrics
        cross_domain_results = {
            'source_domain': source_domain_config.get('domain_name'),
            'target_domain': target_domain_config.get('domain_name'),
            'domain_distance': domain_distance,
            'source_samples': len(source_loader),
            'target_samples': len(target_loader),
            'use_tta': use_tta,
            'scores': scores,
            'timestamp': timestamp
        }

        # Compute TTA improvement (if applicable)
        if use_tta and 'prototype_no_tta' in scores and 'prototype_with_tta' in scores:
            pre_tta = scores['prototype_no_tta']
            post_tta = scores['prototype_with_tta']

            cross_domain_results['tta_improvement'] = {}
            for metric in pre_tta.keys():
                delta = post_tta[metric] - pre_tta[metric]
                cross_domain_results['tta_improvement'][metric] = {
                    'pre': pre_tta[metric],
                    'post': post_tta[metric],
                    'delta': delta
                }

        # Save to JSON
        result_file = results_dir / f"cross_domain_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(cross_domain_results, f, indent=2)

        logger.info(f"\n💾 Cross-domain results saved to: {result_file}")

        # Display summary
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-DOMAIN EVALUATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Source → Target: {source_domain_config.get('domain_name')} → {target_domain_config.get('domain_name')}")
        logger.info(f"Domain distance: {domain_distance:.4f}")
        logger.info(f"\nPerformance on target domain:")

        comparison = compare_methods(scores)
        print("\n" + comparison.to_string())

        if use_tta:
            logger.info("\nTTA Improvement:")
            for metric, values in cross_domain_results.get('tta_improvement', {}).items():
                logger.info(f"  {metric}: {values['pre']:.4f} → {values['post']:.4f} (Δ={values['delta']:+.4f})")

        return cross_domain_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--source-domain',
        type=str,
        required=True,
        choices=['roco', 'mimic', 'iu_xray'],
        help='Source domain for training'
    )
    parser.add_argument(
        '--target-domain',
        type=str,
        required=True,
        choices=['roco', 'mimic', 'iu_xray'],
        help='Target domain for testing'
    )
    parser.add_argument(
        '--use-tta',
        action='store_true',
        default=True,
        help='Use TTA for adaptation'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Define domain configurations
    domain_configs = {
        'roco': {
            'type': 'huggingface',
            'name': 'eltorio/ROCOv2-radiology',
            'domain_name': 'ROCO',
            'max_samples': 1000
        },
        'mimic': {
            'type': 'huggingface',
            'name': 'flaviagiammarino/path-vqa',  # Placeholder - update with actual MIMIC-CXR dataset
            'domain_name': 'MIMIC-CXR',
            'max_samples': 1000
        },
        'iu_xray': {
            'type': 'huggingface',
            'name': 'alkzar90/NIH-Chest-X-ray-dataset',  # Placeholder - update with actual IU X-Ray dataset
            'domain_name': 'IU X-Ray',
            'max_samples': 1000
        }
    }

    # Get source and target configs
    source_config = domain_configs[args.source_domain]
    target_config = domain_configs[args.target_domain]

    # Run cross-domain evaluation
    evaluator = CrossDomainEvaluator(config)
    results = evaluator.evaluate_cross_domain(
        source_config,
        target_config,
        use_tta=args.use_tta
    )

    logger.info("\n✅ Cross-domain evaluation complete")


if __name__ == "__main__":
    main()
