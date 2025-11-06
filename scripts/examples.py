"""
Simple example script demonstrating how to use the zero-shot captioning pipeline.
This shows the basic usage without running the full pipeline.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def example_load_and_retrieve():
    """Example: Load embeddings and retrieve similar captions."""
    print("=" * 60)
    print("Example 1: Loading Embeddings and Retrieval")
    print("=" * 60)
    
    from retrieval import CaptionRetriever
    
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load embeddings
    embeddings_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
    
    if not embeddings_file.exists():
        print(f"\n⚠ Embeddings not found at {embeddings_file}")
        print("Run the pipeline first: python src/main.py")
        return
    
    print(f"\nLoading embeddings from {embeddings_file}...")
    data = np.load(embeddings_file, allow_pickle=True)
    
    # Create retriever
    retriever = CaptionRetriever(
        image_embeddings=data['image_embeddings'],
        text_embeddings=data['text_embeddings'],
        captions=data['captions'].tolist(),
        image_ids=data['image_ids'].tolist(),
        top_k=5
    )
    
    print(f"Loaded {len(retriever.captions)} captions")
    
    # Retrieve for first image
    print(f"\nQuery caption: {retriever.captions[0]}")
    print("\nTop 5 similar captions:")
    
    results = retriever.retrieve_for_index(0)
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['similarity']:.3f}] {result['caption']}")
    
    # Format as context
    print("\nFormatted context for generation:")
    context = retriever.format_retrieved_context(results)
    print(context)


def example_generate_caption_from_file(image_path: str):
    """Example: Generate caption for a custom image."""
    print("\n" + "=" * 60)
    print("Example 2: Generate Caption for Custom Image")
    print("=" * 60)
    
    from generation import CaptionGenerator
    
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"\n⚠ Image not found: {image_path}")
        print("Provide a valid image path")
        return
    
    print(f"\nLoading image from {image_path}...")
    image = Image.open(image_path)
    
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    
    # Generate caption
    print("\nLoading BLIP2 model (this may take a while)...")
    generator = CaptionGenerator(config)
    generator.load_model()
    
    print("\nGenerating caption...")
    caption = generator.generate_caption(image)
    
    print(f"\n📝 Generated caption:")
    print(f"   {caption}")


def example_evaluate_caption():
    """Example: Evaluate generated caption against reference."""
    print("\n" + "=" * 60)
    print("Example 3: Evaluate Caption Quality")
    print("=" * 60)
    
    from evaluation import CaptionEvaluator
    
    evaluator = CaptionEvaluator()
    
    # Example captions
    reference = "Chest X-ray showing bilateral pulmonary infiltrates consistent with pneumonia"
    
    hypotheses = [
        "Chest X-ray with bilateral infiltrates suggesting pneumonia",  # Good
        "X-ray of chest showing lung abnormalities",  # Medium
        "Medical imaging scan",  # Poor
    ]
    
    print(f"\nReference: {reference}\n")
    
    for i, hyp in enumerate(hypotheses, 1):
        print(f"Hypothesis {i}: {hyp}")
        scores = evaluator.evaluate_single(reference, hyp)
        print(f"  BLEU-4: {scores['bleu_4']:.4f}")
        print(f"  METEOR: {scores['meteor']:.4f}")
        print()


def example_prototype_diversity():
    """Example: Compare diversity of different sampling methods."""
    print("=" * 60)
    print("Example 4: Prototype Sampling Diversity")
    print("=" * 60)
    
    from sampling import PrototypeSampler
    
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load embeddings
    embeddings_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
    
    if not embeddings_file.exists():
        print(f"\n⚠ Embeddings not found")
        return
    
    data = np.load(embeddings_file)
    image_embeddings = data['image_embeddings']
    
    print(f"\nTotal samples: {len(image_embeddings)}")
    
    # Create sampler
    sampler = PrototypeSampler(image_embeddings)
    
    num_prototypes = 50
    
    # Compare methods
    print(f"\nComparing sampling methods ({num_prototypes} prototypes):")
    print()
    
    # Random sampling
    print("1. Random sampling...")
    random_indices = sampler.random_sampling(num_prototypes)
    random_diversity = sampler.diversity_score(random_indices)
    print(f"   Diversity score: {random_diversity:.4f}")
    
    # Farthest-point sampling
    print("\n2. Farthest-point sampling...")
    fps_indices = sampler.farthest_point_sampling(num_prototypes)
    fps_diversity = sampler.diversity_score(fps_indices)
    print(f"   Diversity score: {fps_diversity:.4f}")
    
    # Comparison
    improvement = (fps_diversity - random_diversity) / random_diversity * 100
    print(f"\n📊 Improvement: {improvement:.1f}%")
    print("   Higher diversity = better coverage of embedding space")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Zero-Shot Medical Image Captioning - Usage Examples")
    print("=" * 60)
    print()
    print("This script demonstrates basic usage of the pipeline components.")
    print()
    
    # Example 1: Retrieval
    try:
        example_load_and_retrieve()
    except Exception as e:
        print(f"\n⚠ Example 1 failed: {e}")
    
    # Example 3: Evaluation (doesn't need models)
    try:
        example_evaluate_caption()
    except Exception as e:
        print(f"\n⚠ Example 3 failed: {e}")
    
    # Example 4: Prototype diversity
    try:
        example_prototype_diversity()
    except Exception as e:
        print(f"\n⚠ Example 4 failed: {e}")
    
    # Example 2: Caption generation (requires BLIP2 - skip for now)
    print("\n" + "=" * 60)
    print("Example 2: Custom Image Captioning")
    print("=" * 60)
    print("\nTo generate captions for a custom image:")
    print("  python -c \"from examples import example_generate_caption_from_file; \\")
    print("             example_generate_caption_from_file('path/to/image.jpg')\"")
    print("\nNote: This requires loading BLIP2 model (large download)")
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nFor full pipeline: python src/main.py")
    print("For interactive UI: python ui/app.py")


if __name__ == "__main__":
    main()
