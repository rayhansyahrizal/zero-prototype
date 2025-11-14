"""
Prototype sampling module for selecting diverse representative samples.
Implements farthest-point sampling strategy.
"""

import numpy as np
from typing import Optional, List
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PrototypeSampler:
    """Sample prototypes using farthest-point sampling."""
    
    def __init__(self, embeddings: np.ndarray):
        """
        Initialize prototype sampler.
        
        Args:
            embeddings: Embedding matrix (N, D)
        """
        self.embeddings = embeddings
        self.num_samples = len(embeddings)
        
        logger.info(f"PrototypeSampler initialized with {self.num_samples} samples")
    
    def farthest_point_sampling(
        self,
        num_prototypes: int,
        seed: Optional[int] = 42
    ) -> np.ndarray:
        """
        Perform farthest-point sampling to select diverse prototypes.
        
        Algorithm:
        1. Start with a random sample as first prototype
        2. For each iteration:
           - Compute minimum distance from each point to existing prototypes
           - Select the point with maximum minimum distance as next prototype
        3. Repeat until num_prototypes are selected
        
        Args:
            num_prototypes: Number of prototypes to select
            seed: Random seed for initial selection
            
        Returns:
            Array of selected prototype indices
        """
        if num_prototypes >= self.num_samples:
            logger.warning(
                f"Requested {num_prototypes} prototypes but only {self.num_samples} "
                f"samples available. Returning all indices."
            )
            return np.arange(self.num_samples)
        
        np.random.seed(seed)
        
        # Initialize with random sample
        selected_indices = [np.random.randint(0, self.num_samples)]
        
        # Track minimum distances to selected prototypes
        min_distances = np.full(self.num_samples, np.inf)
        
        logger.info(f"Selecting {num_prototypes} prototypes using farthest-point sampling...")
        
        for _ in tqdm(range(num_prototypes - 1), desc="Sampling prototypes"):
            # Get last selected prototype
            last_idx = selected_indices[-1]
            last_embedding = self.embeddings[last_idx:last_idx+1]
            
            # Compute distances from all points to last prototype
            # Using cosine distance = 1 - cosine_similarity
            similarities = np.dot(self.embeddings, last_embedding.T).squeeze()
            distances = 1 - similarities
            
            # Update minimum distances
            min_distances = np.minimum(min_distances, distances)
            
            # Set distance of already selected points to -inf
            min_distances[selected_indices] = -np.inf
            
            # Select point with maximum minimum distance
            next_idx = np.argmax(min_distances)
            selected_indices.append(int(next_idx))
        
        prototype_indices = np.array(selected_indices)
        
        logger.info(f"Selected {len(prototype_indices)} prototypes")
        
        return prototype_indices
    
    def random_sampling(
        self,
        num_prototypes: int,
        seed: Optional[int] = 42
    ) -> np.ndarray:
        """
        Random sampling baseline.
        
        Args:
            num_prototypes: Number of prototypes to select
            seed: Random seed
            
        Returns:
            Array of randomly selected indices
        """
        np.random.seed(seed)
        
        num_prototypes = min(num_prototypes, self.num_samples)
        indices = np.random.choice(
            self.num_samples, size=num_prototypes, replace=False
        )
        
        logger.info(f"Randomly sampled {len(indices)} prototypes")
        
        return indices
    
    def diversity_score(self, indices: np.ndarray) -> float:
        """
        Compute diversity score for a set of samples.
        Higher score = more diverse.
        
        Args:
            indices: Indices of samples to evaluate
            
        Returns:
            Average pairwise distance (diversity score)
        """
        if len(indices) < 2:
            return 0.0
        
        # Get embeddings of selected samples
        selected_embeddings = self.embeddings[indices]
        
        # Compute pairwise similarity matrix
        similarity_matrix = np.dot(selected_embeddings, selected_embeddings.T)
        
        # Convert to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        # Get upper triangle (exclude diagonal)
        triu_indices = np.triu_indices_from(distance_matrix, k=1)
        pairwise_distances = distance_matrix[triu_indices]
        
        # Return average distance
        diversity = np.mean(pairwise_distances)
        
        return float(diversity)


def sample_and_save_prototypes(
    config: dict,
    embeddings: np.ndarray,
    force_regenerate: bool = False
) -> np.ndarray:
    """
    Sample prototypes and save to disk.
    
    Args:
        config: Configuration dictionary
        embeddings: Image embeddings (N, D)
        force_regenerate: If True, regenerate even if cached
        
    Returns:
        Array of prototype indices
    """
    prototypes_path = Path(config['output']['prototypes_path'])
    prototypes_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if prototypes already exist
    if prototypes_path.exists() and not force_regenerate:
        logger.info(f"Loading cached prototypes from {prototypes_path}")
        prototype_indices = np.load(prototypes_path)
        logger.info(f"Loaded {len(prototype_indices)} prototypes")
        return prototype_indices
    
    # Sample new prototypes
    sampler = PrototypeSampler(embeddings)
    
    num_prototypes = config['sampling']['num_prototypes']
    method = config['sampling']['method']
    
    if method == "farthest_point":
        prototype_indices = sampler.farthest_point_sampling(num_prototypes)
    elif method == "random":
        prototype_indices = sampler.random_sampling(num_prototypes)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # Compute diversity score
    diversity = sampler.diversity_score(prototype_indices)
    logger.info(f"Prototype diversity score: {diversity:.4f}")
    
    # Save prototypes (NPY)
    logger.info(f"Menyimpan prototypes ke {prototypes_path}")
    np.save(prototypes_path, prototype_indices)

    # Save prototypes + embeddings sebagai JSON (sesuai skripsi)
    json_path = prototypes_path.parent / "prototypes.json"
    logger.info(f"Menyimpan prototypes + embeddings ke {json_path}")

    prototype_data = {
        "num_prototypes": int(len(prototype_indices)),
        "sampling_method": method,
        "diversity_score": float(diversity),
        "indices": prototype_indices.tolist(),
        "embeddings": embeddings[prototype_indices].tolist()
    }

    import json
    with open(json_path, 'w') as f:
        json.dump(prototype_data, f, indent=2)

    logger.info(f"✅ Prototypes tersimpan: {len(prototype_indices)} sampel (diversity={diversity:.4f})")

    return prototype_indices


if __name__ == "__main__":
    # Test prototype sampling
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load embeddings
    embeddings_file = Path(config['output']['embeddings_dir']) / "embeddings.npz"
    
    if not embeddings_file.exists():
        print(f"Embeddings not found at {embeddings_file}")
        print("Run embedding.py first to generate embeddings.")
    else:
        data = np.load(embeddings_file)
        image_embeddings = data['image_embeddings']
        
        print(f"\nTotal samples: {len(image_embeddings)}")
        
        # Test different sampling methods
        sampler = PrototypeSampler(image_embeddings)
        
        num_prototypes = min(100, len(image_embeddings) // 2)
        
        # Farthest-point sampling
        print(f"\nFarthest-point sampling ({num_prototypes} prototypes)...")
        fps_indices = sampler.farthest_point_sampling(num_prototypes)
        fps_diversity = sampler.diversity_score(fps_indices)
        print(f"Diversity score: {fps_diversity:.4f}")
        
        # Random sampling
        print(f"\nRandom sampling ({num_prototypes} prototypes)...")
        random_indices = sampler.random_sampling(num_prototypes)
        random_diversity = sampler.diversity_score(random_indices)
        print(f"Diversity score: {random_diversity:.4f}")
        
        print(f"\nImprovement: {(fps_diversity - random_diversity) / random_diversity * 100:.1f}%")
        
        # Save using config
        prototype_indices = sample_and_save_prototypes(
            config, image_embeddings, force_regenerate=True
        )
        print(f"\nSaved {len(prototype_indices)} prototypes")
