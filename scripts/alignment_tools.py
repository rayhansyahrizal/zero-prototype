# Practical Guide: Visualizing and Debugging Embedding Alignment

## Tool 1: Similarity Distribution Analyzer

Create this script to understand your alignment:

```python
# scripts/analyze_alignment.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from retrieval import CaptionRetriever, PrototypeRetriever

def analyze_similarity_distribution():
    """Analyze how similarities are distributed."""
    
    # Load config and data
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    embeddings_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
    prototypes_path = Path(config['output']['prototypes_path'])
    
    data = np.load(embeddings_file, allow_pickle=True)
    prototype_indices = np.load(prototypes_path)
    
    # Create retriever
    retriever = PrototypeRetriever(
        image_embeddings=data['image_embeddings'],
        text_embeddings=data['text_embeddings'],
        captions=data['captions'].tolist(),
        image_ids=data['image_ids'].tolist(),
        prototype_indices=prototype_indices,
        top_k=50  # Get more results for distribution
    )
    
    # Collect all similarities
    print("Analyzing similarity distribution...")
    all_similarities = []
    
    for query_idx in range(min(100, len(data['image_embeddings']))):
        query_embedding = data['image_embeddings'][query_idx]
        results = retriever.retrieve_by_image_embedding(query_embedding)
        sims = [r['similarity'] for r in results]
        all_similarities.extend(sims)
    
    all_similarities = np.array(all_similarities)
    
    # Analyze
    print(f"\nSimilarity Statistics (100 queries × 50 top-k):")
    print(f"  Count:      {len(all_similarities)}")
    print(f"  Mean:       {np.mean(all_similarities):.4f}")
    print(f"  Median:     {np.median(all_similarities):.4f}")
    print(f"  Std Dev:    {np.std(all_similarities):.4f}")
    print(f"  Min:        {np.min(all_similarities):.4f}")
    print(f"  Max:        {np.max(all_similarities):.4f}")
    print(f"  Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    {p:2d}th:     {np.percentile(all_similarities, p):.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    axes[0, 0].hist(all_similarities, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_similarities), color='red', 
                        linestyle='--', label=f'Mean: {np.mean(all_similarities):.3f}')
    axes[0, 0].set_xlabel('Similarity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Similarities')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Box plot by rank
    all_sims_by_rank = {i: [] for i in range(50)}
    for query_idx in range(min(100, len(data['image_embeddings']))):
        query_embedding = data['image_embeddings'][query_idx]
        results = retriever.retrieve_by_image_embedding(query_embedding)
        for rank, result in enumerate(results):
            all_sims_by_rank[rank].append(result['similarity'])
    
    ranks = list(range(50))
    data_by_rank = [all_sims_by_rank[r] for r in ranks]
    
    axes[0, 1].boxplot(data_by_rank, positions=ranks[::5], widths=2)
    axes[0, 1].set_xlabel('Retrieval Rank')
    axes[0, 1].set_ylabel('Similarity')
    axes[0, 1].set_title('Similarity by Retrieval Rank')
    axes[0, 1].grid(alpha=0.3)
    
    # Cumulative distribution
    sorted_sims = np.sort(all_similarities)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    axes[1, 0].plot(sorted_sims, cumulative, linewidth=2)
    axes[1, 0].axvline(np.mean(all_similarities), color='red', 
                        linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Similarity')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('CDF of Similarities')
    axes[1, 0].grid(alpha=0.3)
    
    # Alignment quality interpretation
    ax = axes[1, 1]
    ax.axis('off')
    
    mean_sim = np.mean(all_similarities)
    if mean_sim > 0.95:
        quality = "EXCELLENT ✓"
        color = 'green'
        reason = "Well-aligned embeddings"
    elif mean_sim > 0.85:
        quality = "GOOD ✓"
        color = 'lightgreen'
        reason = "Reasonably well-aligned"
    elif mean_sim > 0.70:
        quality = "FAIR ⚠"
        color = 'yellow'
        reason = "Moderate alignment"
    else:
        quality = "POOR ✗"
        color = 'red'
        reason = "Poor alignment - check model"
    
    text = f"""
Alignment Quality: {quality}

Mean Similarity: {mean_sim:.4f}
Std Deviation:   {np.std(all_similarities):.4f}

Interpretation:
{reason}

What this means:
• 0.96+ : Embeddings encode strong semantic similarity
• 0.85-0.95 : Good separation between related/unrelated
• 0.70-0.85 : Moderate, may need improvement
• <0.70 : Weak alignment, investigate model

Your mean: {mean_sim:.4f}
Status: {"Top-tier quality" if mean_sim > 0.95 else "Needs review"}
    """
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('alignment_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: alignment_analysis.png")
    plt.close()


def compare_models():
    """Compare alignment across different scenarios."""
    
    print("\nSimulating different embedding scenarios...\n")
    
    # Scenario 1: Your system
    print("Scenario 1: MedImageInsight (Your System)")
    print("  Expected similarity: 0.96 ✓")
    print("  Why: Medical-specialized model")
    
    # Scenario 2: Weak model
    print("\nScenario 2: Generic CNN (ImageNet)")
    print("  Expected similarity: 0.65-0.75")
    print("  Why: Not specialized for medical domain")
    
    # Scenario 3: Random embeddings
    print("\nScenario 3: Random Embeddings")
    # Random unit vectors in high dimensions tend to be orthogonal
    dim = 512
    n_vectors = 1000
    random_embeddings = np.random.randn(n_vectors, dim)
    random_embeddings = random_embeddings / np.linalg.norm(
        random_embeddings, axis=1, keepdims=True
    )
    
    # Get top-5 similarities for first query
    query = random_embeddings[0]
    similarities = np.dot(random_embeddings, query)
    top_sims = np.sort(similarities)[-5:][::-1]
    
    print(f"  Expected similarity: {np.mean(top_sims):.4f}")
    print("  Why: Random vectors are ~orthogonal in high dimensions")
    
    # Scenario 4: Clustered embeddings
    print("\nScenario 4: Well-Clustered Embeddings")
    # Create clusters
    cluster_centers = np.random.randn(10, dim)
    cluster_centers = cluster_centers / np.linalg.norm(
        cluster_centers, axis=1, keepdims=True
    )
    
    clustered = []
    for i in range(100):
        center = cluster_centers[i % 10]
        point = center + np.random.randn(dim) * 0.1
        point = point / np.linalg.norm(point)
        clustered.append(point)
    
    clustered = np.array(clustered)
    query = clustered[0]
    similarities = np.dot(clustered, query)
    top_sims = np.sort(similarities)[-5:][::-1]
    
    print(f"  Expected similarity: {np.mean(top_sims):.4f}")
    print("  Why: Clustering creates high intra-group similarity")


if __name__ == "__main__":
    analyze_similarity_distribution()
    compare_models()
```

---

## Tool 2: Per-Query Alignment Inspector

```python
# scripts/inspect_query_alignment.py

import numpy as np
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from retrieval import PrototypeRetriever

def inspect_query(query_idx: int):
    """Inspect alignment for a specific query."""
    
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    embeddings_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
    prototypes_path = Path(config['output']['prototypes_path'])
    
    data = np.load(embeddings_file, allow_pickle=True)
    prototype_indices = np.load(prototypes_path)
    
    retriever = PrototypeRetriever(
        image_embeddings=data['image_embeddings'],
        text_embeddings=data['text_embeddings'],
        captions=data['captions'].tolist(),
        image_ids=data['image_ids'].tolist(),
        prototype_indices=prototype_indices,
        top_k=100  # Get many results
    )
    
    query_embedding = data['image_embeddings'][query_idx]
    query_caption = data['captions'][query_idx]
    query_id = data['image_ids'][query_idx]
    
    print(f"\n{'='*70}")
    print(f"Query {query_idx}: {query_id}")
    print(f"{'='*70}")
    print(f"Caption: {query_caption}\n")
    
    # Get all similarities
    results = retriever.retrieve_by_image_embedding(query_embedding)
    
    # Analyze alignment
    sims = np.array([r['similarity'] for r in results])
    
    print(f"Alignment Statistics:")
    print(f"  Top-1 similarity:  {sims[0]:.4f}")
    print(f"  Top-5 mean:        {np.mean(sims[:5]):.4f}")
    print(f"  Top-10 mean:       {np.mean(sims[:10]):.4f}")
    print(f"  Top-50 mean:       {np.mean(sims[:50]):.4f}")
    print(f"  Overall mean:      {np.mean(sims):.4f}")
    print(f"  Std deviation:     {np.std(sims):.4f}")
    print(f"  Min:               {np.min(sims):.4f}")
    print(f"  Max:               {np.max(sims):.4f}")
    
    # Show top retrievals
    print(f"\nTop 10 Retrieved Items:")
    for rank, result in enumerate(results[:10], 1):
        print(f"  {rank:2d}. [sim={result['similarity']:.4f}] {result['caption'][:70]}")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    if sims[0] > 0.95:
        print(f"  ✓ Excellent - Top result is nearly identical")
    elif sims[0] > 0.90:
        print(f"  ✓ Good - Top result is very similar")
    elif sims[0] > 0.80:
        print(f"  ⚠ Fair - Top result is somewhat similar")
    else:
        print(f"  ✗ Poor - Top result is not very similar")
    
    # Alignment classification
    alignment_classes = {
        'identical': (sims >= 0.95).sum(),
        'very_similar': ((sims >= 0.90) & (sims < 0.95)).sum(),
        'similar': ((sims >= 0.80) & (sims < 0.90)).sum(),
        'somewhat': ((sims >= 0.70) & (sims < 0.80)).sum(),
        'different': (sims < 0.70).sum()
    }
    
    print(f"\nRetrieval Distribution (100 results):")
    print(f"  Identical (≥0.95):    {alignment_classes['identical']:3d}")
    print(f"  Very similar (0.90-0.95): {alignment_classes['very_similar']:3d}")
    print(f"  Similar (0.80-0.90):  {alignment_classes['similar']:3d}")
    print(f"  Somewhat (0.70-0.80): {alignment_classes['somewhat']:3d}")
    print(f"  Different (<0.70):    {alignment_classes['different']:3d}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=int, default=0, 
                        help='Query index to inspect')
    args = parser.parse_args()
    
    inspect_query(args.query)
```

---

## Running the Tools

```bash
# Analyze overall alignment distribution
cd /mnt/nas-hpg9/rayhan/zero-prototype
source zero-prototype-env/bin/activate
python scripts/analyze_alignment.py

# Inspect a specific query
python scripts/inspect_query_alignment.py --query 0
python scripts/inspect_query_alignment.py --query 50
python scripts/inspect_query_alignment.py --query 100

# Try different queries to find interesting cases
for i in {0..10}; do
  echo "Query $i:"
  python scripts/inspect_query_alignment.py --query $i | grep "Top-5 mean"
done
```

---

## Interpreting Results

### Example Output 1: Excellent Alignment ✓

```
Query 0: ct_001
Caption: CT chest scan showing pneumonia

Alignment Statistics:
  Top-1 similarity:  0.9999  ← Nearly perfect!
  Top-5 mean:        0.9614
  Top-10 mean:       0.9450
  Top-50 mean:       0.8750
  Overall mean:      0.7234
  
Quality Assessment:
  ✓ Excellent - Top result is nearly identical
  
Retrieval Distribution:
  Identical (≥0.95):    18  ← Many very similar items!
  Very similar (0.90-0.95): 35
  Similar (0.80-0.90):  28
  Somewhat (0.70-0.80): 15
  Different (<0.70):     4

Interpretation:
  • System found 18 images nearly identical to query
  • This is GOOD - query has strong semantic clarity
  • Top retrievals are highly aligned
  • System is working well!
```

### Example Output 2: Moderate Alignment ⚠

```
Query 45: xray_ambiguous_001
Caption: Chest X-ray with possible artifact

Alignment Statistics:
  Top-1 similarity:  0.7234  ← Moderate
  Top-5 mean:        0.7012
  Top-10 mean:       0.6890
  Top-50 mean:       0.5234
  Overall mean:      0.4561
  
Quality Assessment:
  ⚠ Fair - Top result is somewhat similar

Retrieval Distribution:
  Identical (≥0.95):     0  ← None perfect!
  Very similar (0.90-0.95): 2  ← Few very similar
  Similar (0.80-0.90):   8
  Somewhat (0.70-0.80):  45  ← Many moderate
  Different (<0.70):     45

Interpretation:
  • Query is ambiguous or rare
  • System struggles to find highly similar items
  • Possible causes:
    - Image quality issues
    - Rare diagnosis
    - Atypical anatomy
    - Cross-modal differences
  • TTA would likely help here!
```

---

## What to Look For

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| Top-1 sim | >0.95 | 0.90-0.95 | 0.80-0.90 | <0.80 |
| Top-5 mean | >0.93 | 0.88-0.93 | 0.78-0.88 | <0.78 |
| # Identical | >20/100 | 10-20 | 5-10 | <5 |
| Interpretation | Perfect alignment | Well-aligned | Moderate | Poor |

Your system: **Top-1 = 0.99+, Top-5 = 0.96+** = **EXCELLENT** ✓

