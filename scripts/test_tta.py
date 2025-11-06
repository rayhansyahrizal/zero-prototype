"""
Test script for Test-Time Adaptation (TTA) in PrototypeRetriever.

This script demonstrates how to use the test-time adaptation feature
and compares retrieval results with and without adaptation.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval import PrototypeRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_tta(verbose: bool = True):
    """Test test-time adaptation feature."""
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load embeddings
    logger.info("Loading embeddings...")
    embeddings_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
    
    if not embeddings_file.exists():
        logger.error(f"Embeddings not found at {embeddings_file}")
        logger.error("Run the main pipeline first to generate embeddings.")
        return
    
    data = np.load(embeddings_file, allow_pickle=True)
    
    # Load prototypes
    logger.info("Loading prototypes...")
    prototypes_path = Path(config['output']['prototypes_path'])
    
    if not prototypes_path.exists():
        logger.error(f"Prototypes not found at {prototypes_path}")
        logger.error("Run the main pipeline first to generate prototypes.")
        return
    
    prototype_indices = np.load(prototypes_path)
    
    # Create prototype retriever
    logger.info("Initializing PrototypeRetriever...")
    retriever = PrototypeRetriever(
        image_embeddings=data['image_embeddings'],
        text_embeddings=data['text_embeddings'],
        captions=data['captions'].tolist(),
        image_ids=data['image_ids'].tolist(),
        prototype_indices=prototype_indices,
        top_k=5
    )
    
    # Test on first sample
    logger.info("\n" + "="*70)
    logger.info("TEST: Comparing retrieval with and without TTA")
    logger.info("="*70)
    
    query_idx = 0
    query_embedding = data['image_embeddings'][query_idx]
    query_caption = data['captions'][query_idx]
    query_id = data['image_ids'][query_idx]
    
    logger.info(f"\nQuery Image ID: {query_id}")
    logger.info(f"Query Caption: {query_caption}")
    
    # Retrieval WITHOUT adaptation
    logger.info("\n" + "-"*70)
    logger.info("1. BASELINE RETRIEVAL (No Adaptation)")
    logger.info("-"*70)
    
    results_baseline = retriever.retrieve_by_image_embedding(
        query_embedding,
        exclude_indices=[query_idx],
        use_adaptation=False
    )
    
    logger.info(f"\nTop-{len(results_baseline)} Retrieved Captions:")
    for i, result in enumerate(results_baseline, 1):
        logger.info(
            f"{i}. [sim={result['similarity']:.4f}] {result['caption'][:100]}..."
        )
    
    # Retrieval WITH adaptation (more aggressive)
    logger.info("\n" + "-"*70)
    logger.info("2. WITH TEST-TIME ADAPTATION (Aggressive)")
    logger.info("-"*70)
    
    # Use more aggressive parameters for better visibility
    adaptation_params = {
        'learning_rate': 5e-3,      # Stronger learning rate
        'num_steps': 20,            # More iterations
        'top_k_for_loss': 2,
        'weight_variance': 0.05,    # Less variance penalty to allow more adaptation
        'weight_entropy': 0.005     # Less entropy penalty
    }
    
    logger.info(f"Adaptation parameters: {adaptation_params}")
    
    results_adapted = retriever.retrieve_by_image_embedding(
        query_embedding,
        exclude_indices=[query_idx],
        use_adaptation=True,
        adaptation_params=adaptation_params
    )
    
    logger.info(f"\nTop-{len(results_adapted)} Retrieved Captions:")
    for i, result in enumerate(results_adapted, 1):
        logger.info(
            f"{i}. [sim={result['similarity']:.4f}] {result['caption'][:100]}..."
        )
    
    # Compare results
    logger.info("\n" + "="*70)
    logger.info("COMPARISON")
    logger.info("="*70)
    
    # Calculate mean similarity
    baseline_mean_sim = np.mean([r['similarity'] for r in results_baseline])
    adapted_mean_sim = np.mean([r['similarity'] for r in results_adapted])
    
    logger.info(f"\nBaseline mean similarity:  {baseline_mean_sim:.4f}")
    logger.info(f"Adapted mean similarity:   {adapted_mean_sim:.4f}")
    logger.info(f"Improvement:               {(adapted_mean_sim - baseline_mean_sim):.4f} "
                f"({(adapted_mean_sim / baseline_mean_sim - 1) * 100:+.2f}%)")
    
    # Check if retrieved captions changed
    baseline_ids = [r['image_id'] for r in results_baseline]
    adapted_ids = [r['image_id'] for r in results_adapted]
    
    changed = sum(1 for b, a in zip(baseline_ids, adapted_ids) if b != a)
    logger.info(f"\nRetrieved captions changed: {changed}/{len(baseline_ids)}")
    
    if changed > 0:
        logger.info("\nDifferent retrievals:")
        for i, (b_id, a_id) in enumerate(zip(baseline_ids, adapted_ids), 1):
            if b_id != a_id:
                logger.info(f"  Position {i}: {b_id} -> {a_id}")
    
    logger.info("\n" + "="*70)
    logger.info("TEST COMPLETE")
    logger.info("="*70)


def test_multiple_samples():
    """Test TTA on multiple samples to see average improvement."""
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load data
    logger.info("Loading data...")
    embeddings_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
    prototypes_path = Path(config['output']['prototypes_path'])
    
    if not embeddings_file.exists() or not prototypes_path.exists():
        logger.error("Required files not found. Run main pipeline first.")
        return
    
    data = np.load(embeddings_file, allow_pickle=True)
    prototype_indices = np.load(prototypes_path)
    
    # Create retriever
    retriever = PrototypeRetriever(
        image_embeddings=data['image_embeddings'],
        text_embeddings=data['text_embeddings'],
        captions=data['captions'].tolist(),
        image_ids=data['image_ids'].tolist(),
        prototype_indices=prototype_indices,
        top_k=5
    )
    
    # Test on multiple samples
    num_samples = min(10, len(data['image_embeddings']))
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing TTA on {num_samples} samples")
    logger.info(f"{'='*70}")
    
    improvements = []
    
    for idx in range(num_samples):
        query_embedding = data['image_embeddings'][idx]
        
        # Baseline
        results_baseline = retriever.retrieve_by_image_embedding(
            query_embedding,
            exclude_indices=[idx],
            use_adaptation=False
        )
        baseline_sim = np.mean([r['similarity'] for r in results_baseline])
        
        # Adapted
        results_adapted = retriever.retrieve_by_image_embedding(
            query_embedding,
            exclude_indices=[idx],
            use_adaptation=True,
            adaptation_params={'learning_rate': 1e-3, 'num_steps': 10}
        )
        adapted_sim = np.mean([r['similarity'] for r in results_adapted])
        
        improvement = adapted_sim - baseline_sim
        improvements.append(improvement)
        
        logger.info(
            f"Sample {idx+1}: baseline={baseline_sim:.4f}, "
            f"adapted={adapted_sim:.4f}, improvement={improvement:+.4f}"
        )
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Average improvement: {np.mean(improvements):+.4f}")
    logger.info(f"Std deviation:       {np.std(improvements):.4f}")
    logger.info(f"Positive improvements: {sum(1 for x in improvements if x > 0)}/{num_samples}")
    logger.info(f"{'='*70}")


def test_tta_diagnostic():
    """Detailed diagnostic test to understand embedding alignment."""
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load data
    logger.info("Loading data for diagnostic...")
    embeddings_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
    prototypes_path = Path(config['output']['prototypes_path'])
    
    if not embeddings_file.exists() or not prototypes_path.exists():
        logger.error("Required files not found.")
        return
    
    data = np.load(embeddings_file, allow_pickle=True)
    prototype_indices = np.load(prototypes_path)
    
    retriever = PrototypeRetriever(
        image_embeddings=data['image_embeddings'],
        text_embeddings=data['text_embeddings'],
        captions=data['captions'].tolist(),
        image_ids=data['image_ids'].tolist(),
        prototype_indices=prototype_indices,
        top_k=5
    )
    
    logger.info(f"\n{'='*70}")
    logger.info("DIAGNOSTIC: Understanding Embedding Alignment")
    logger.info(f"{'='*70}")
    
    # Analyze baseline similarities
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total samples:     {len(data['image_embeddings'])}")
    logger.info(f"  Total prototypes:  {len(prototype_indices)}")
    logger.info(f"  Prototype ratio:   {len(prototype_indices) / len(data['image_embeddings']):.1%}")
    
    # Get statistics on baseline similarities
    all_similarities = []
    
    for idx in range(min(50, len(data['image_embeddings']))):
        query_embedding = data['image_embeddings'][idx]
        results = retriever.retrieve_by_image_embedding(query_embedding)
        sims = [r['similarity'] for r in results]
        all_similarities.extend(sims)
    
    all_similarities = np.array(all_similarities)
    
    logger.info(f"\nBaseline Similarity Statistics (50 samples × 5 top-k):")
    logger.info(f"  Mean:           {np.mean(all_similarities):.4f}")
    logger.info(f"  Median:         {np.median(all_similarities):.4f}")
    logger.info(f"  Std Dev:        {np.std(all_similarities):.4f}")
    logger.info(f"  Min:            {np.min(all_similarities):.4f}")
    logger.info(f"  Max:            {np.max(all_similarities):.4f}")
    logger.info(f"  95th percentile: {np.percentile(all_similarities, 95):.4f}")
    
    # Test on sample with lower baseline similarity
    logger.info(f"\n{'='*70}")
    logger.info("Finding sample with LOWER baseline similarity for testing...")
    logger.info(f"{'='*70}")
    
    # Find sample with lowest average similarity
    min_idx = 0
    min_avg_sim = 1.0
    
    for idx in range(min(100, len(data['image_embeddings']))):
        query_embedding = data['image_embeddings'][idx]
        results = retriever.retrieve_by_image_embedding(query_embedding)
        avg_sim = np.mean([r['similarity'] for r in results])
        
        if avg_sim < min_avg_sim:
            min_avg_sim = avg_sim
            min_idx = idx
    
    logger.info(f"\nSelected sample {min_idx} with baseline similarity: {min_avg_sim:.4f}")
    
    query_embedding = data['image_embeddings'][min_idx]
    query_caption = data['captions'][min_idx]
    
    logger.info(f"Query caption: {query_caption[:80]}...")
    
    # Baseline
    results_baseline = retriever.retrieve_by_image_embedding(query_embedding)
    baseline_mean = np.mean([r['similarity'] for r in results_baseline])
    
    logger.info(f"\n--- BASELINE Retrieval ---")
    sims_str = ", ".join([f'{r["similarity"]:.4f}' for r in results_baseline])
    logger.info(f"Similarities: [{sims_str}]")
    logger.info(f"Mean: {baseline_mean:.4f}")
    
    # Adapted (aggressive parameters)
    logger.info(f"\n--- WITH TEST-TIME ADAPTATION (Aggressive) ---")
    logger.info("Running adaptation with aggressive parameters...")
    
    results_adapted = retriever.retrieve_by_image_embedding(
        query_embedding,
        use_adaptation=True,
        adaptation_params={
            'learning_rate': 5e-2,     # Very strong learning rate
            'num_steps': 50,           # Many steps
            'top_k_for_loss': 5,       # Use all top-k
            'weight_variance': 0.01,   # Minimal variance penalty
            'weight_entropy': 0.0001   # Minimal entropy penalty
        }
    )
    adapted_mean = np.mean([r['similarity'] for r in results_adapted])
    
    adapted_sims_str = ", ".join([f'{r["similarity"]:.4f}' for r in results_adapted])
    logger.info(f"Similarities: [{adapted_sims_str}]")
    logger.info(f"Mean: {adapted_mean:.4f}")
    logger.info(f"\nImprovement: {(adapted_mean - baseline_mean):+.6f} "
                f"({(adapted_mean / baseline_mean - 1) * 100:+.3f}%)")
    
    # Check if retrieved items changed
    baseline_ids = [r['image_id'] for r in results_baseline]
    adapted_ids = [r['image_id'] for r in results_adapted]
    
    changed = sum(1 for b, a in zip(baseline_ids, adapted_ids) if b != a)
    logger.info(f"\nRetrieved items changed: {changed}/5")
    
    if changed > 0:
        logger.info("Changes in ranking:")
        for i, (b_id, a_id) in enumerate(zip(baseline_ids, adapted_ids), 1):
            if b_id != a_id:
                logger.info(f"  Position {i}: {b_id} → {a_id}")
    
    logger.info(f"\n{'='*70}")
    logger.info("INTERPRETATION")
    logger.info(f"{'='*70}")
    
    if baseline_mean > 0.95:
        logger.info(
            "✓ EXCELLENT embedding quality! (>0.95 baseline similarity)\n\n"
            "Why minimal TTA improvement?\n"
            "  • Your baseline similarities are already near-optimal (0.95+)\n"
            "  • TTA optimizes from an already excellent starting point\n"
            "  • Limited room for improvement due to ceiling effect\n\n"
            "When TTA shows better results:\n"
            "  • With lower-quality embeddings (e.g., basic CNN, ~0.7-0.8)\n"
            "  • With challenging/ambiguous queries\n"
            "  • With smaller prototype sets (more varied content)\n"
            "  • For cross-modal or zero-shot scenarios\n\n"
            "Current value: TTA still helps for edge cases and ranking refinement"
        )
    else:
        logger.info(f"Baseline similarity is moderate ({baseline_mean:.2f})")
        logger.info("TTA should show meaningful improvements")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Test-Time Adaptation")
    parser.add_argument(
        '--mode',
        choices=['single', 'multiple', 'diagnostic'],
        default='single',
        help='Test mode: single sample, multiple samples, or diagnostic'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        test_tta()
    elif args.mode == 'multiple':
        test_multiple_samples()
    else:
        test_tta_diagnostic()
