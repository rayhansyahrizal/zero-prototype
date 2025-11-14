"""
Retrieval module for finding similar captions based on image embeddings.
Uses cosine similarity for retrieval.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    import warnings
    warnings.warn("PyTorch not available. Test-time adaptation will be disabled.")

logger = logging.getLogger(__name__)


class CaptionRetriever:
    """Retrieve similar captions using cosine similarity on visual embeddings."""
    
    def __init__(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        captions: List[str],
        image_ids: List[str],
        modalities: Optional[List[str]] = None,
        top_k: int = 5
    ):
        """
        Initialize caption retriever.
        
        Args:
            image_embeddings: Numpy array of image embeddings (N, D)
            text_embeddings: Numpy array of text embeddings (N, D)
            captions: List of caption strings
            image_ids: List of image IDs
            modalities: List of modality labels (XR, CT, MRI, etc.) for filtering
            top_k: Number of top similar captions to retrieve
        """
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.captions = captions
        self.image_ids = image_ids
        self.modalities = modalities if modalities is not None else ['UNKNOWN'] * len(image_ids)
        self.top_k = top_k
        
        # Validate inputs
        assert len(image_embeddings) == len(text_embeddings) == len(captions) == len(image_ids) == len(self.modalities)
        
        logger.info(
            f"CaptionRetriever initialized with {len(captions)} captions, "
            f"top_k={top_k}, modalities={set(self.modalities)}"
        )
    
    @staticmethod
    def compute_cosine_similarity(
        query_embedding: np.ndarray,
        database_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and database embeddings.
        
        Args:
            query_embedding: Query embedding (D,) or (1, D)
            database_embeddings: Database embeddings (N, D)
            
        Returns:
            Similarity scores (N,)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        
        # Compute dot product (assuming embeddings are normalized)
        similarities = np.dot(database_embeddings, query_embedding.T).squeeze()
        
        return similarities
    
    def retrieve_by_image_embedding(
        self,
        query_embedding: np.ndarray,
        exclude_indices: Optional[List[int]] = None,
        filter_modality: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve top-k captions based on image embedding similarity with optional modality filtering.
        
        This is the CORE retrieval method following thesis methodology:
        - Uses VISUAL embedding similarity (not text)
        - Returns associated TEXT captions from top-k visual matches
        - Optionally filters by modality (XR, CT, MRI, etc.)
        
        Args:
            query_embedding: Query image embedding (D,)
            exclude_indices: Indices to exclude from retrieval (e.g., query itself)
            filter_modality: Optional modality to filter by (e.g., 'XR', 'CT', 'MRI')
            
        Returns:
            List of dictionaries with retrieved captions and metadata
        """
        # Step 1: Filter by modality if requested
        if filter_modality:
            # Get indices matching modality
            valid_indices = [
                i for i, mod in enumerate(self.modalities) 
                if mod == filter_modality
            ]
            
            if not valid_indices:
                logger.warning(f"No samples found for modality '{filter_modality}'")
                return []
            
            # Filter embeddings
            filtered_embeddings = self.image_embeddings[valid_indices]
            
            logger.debug(f"Filtered to {len(valid_indices)} samples with modality '{filter_modality}'")
        else:
            valid_indices = list(range(len(self.image_embeddings)))
            filtered_embeddings = self.image_embeddings
        
        # Step 2: Compute similarities with filtered images
        similarities = self.compute_cosine_similarity(
            query_embedding, filtered_embeddings
        )
        
        # Ensure similarities is always a 1D array (handle edge case of single sample)
        similarities = np.atleast_1d(similarities)
        
        # Step 3: Exclude specified indices (map to filtered space)
        if exclude_indices:
            for orig_idx in exclude_indices:
                if orig_idx in valid_indices:
                    filtered_idx = valid_indices.index(orig_idx)
                    similarities[filtered_idx] = -np.inf
        
        # Step 4: Get top-k indices in filtered space
        k = min(self.top_k, len(similarities))
        top_k_filtered_indices = np.argsort(similarities)[-k:][::-1]
        
        # Step 5: Map back to original indices and build results
        results = []
        for filtered_idx in top_k_filtered_indices:
            orig_idx = valid_indices[filtered_idx]
            results.append({
                'index': int(orig_idx),
                'image_id': self.image_ids[orig_idx],
                'caption': self.captions[orig_idx],
                'modality': self.modalities[orig_idx],
                'similarity': float(similarities[filtered_idx])
            })
        
        return results
    
    def retrieve_by_text_embedding(
        self,
        query_embedding: np.ndarray,
        exclude_indices: Optional[List[int]] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve top-k captions based on text embedding similarity.
        
        Args:
            query_embedding: Query text embedding (D,)
            exclude_indices: Indices to exclude from retrieval
            
        Returns:
            List of dictionaries with retrieved captions and metadata
        """
        # Compute similarities with all text embeddings
        similarities = self.compute_cosine_similarity(
            query_embedding, self.text_embeddings
        )
        
        # Exclude specified indices
        if exclude_indices:
            similarities[exclude_indices] = -np.inf
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_k_indices:
            results.append({
                'index': int(idx),
                'image_id': self.image_ids[idx],
                'caption': self.captions[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def retrieve_for_index(
        self,
        query_idx: int,
        use_image_similarity: bool = True,
        filter_modality: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve captions for a specific index in the database.
        
        Args:
            query_idx: Index of the query sample
            use_image_similarity: If True, use image embeddings; else text embeddings
            filter_modality: Optional modality to filter by (e.g., 'XR', 'CT')
            
        Returns:
            List of retrieved captions (excluding query itself)
        """
        if use_image_similarity:
            query_embedding = self.image_embeddings[query_idx]
            return self.retrieve_by_image_embedding(
                query_embedding,
                exclude_indices=[query_idx],
                filter_modality=filter_modality
            )
        else:
            query_embedding = self.text_embeddings[query_idx]
            return self.retrieve_by_text_embedding(
                query_embedding,
                exclude_indices=[query_idx]
            )
    
    def format_retrieved_context(
        self,
        retrieved_results: List[Dict[str, any]],
        add_similarity_scores: bool = False
    ) -> str:
        """
        Format retrieved captions into a single context string.
        
        Args:
            retrieved_results: List of retrieval results
            add_similarity_scores: Whether to include similarity scores
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(retrieved_results, 1):
            caption = result['caption']
            if add_similarity_scores:
                sim = result['similarity']
                context_parts.append(f"{i}. {caption} (sim: {sim:.3f})")
            else:
                context_parts.append(f"{i}. {caption}")
        
        return "\n".join(context_parts)
    
    def batch_retrieve(
        self,
        indices: Optional[List[int]] = None,
        use_image_similarity: bool = True,
        filter_modality: bool = True
    ) -> Dict[int, List[Dict[str, any]]]:
        """
        Retrieve captions for multiple indices.
        
        Args:
            indices: List of query indices (None = all)
            use_image_similarity: Whether to use image or text similarity
            filter_modality: If True, filter by same modality as query
            
        Returns:
            Dictionary mapping index to retrieved results
        """
        if indices is None:
            indices = list(range(len(self.captions)))
        
        results = {}
        for idx in indices:
            # Determine modality filter
            modality_filter = self.modalities[idx] if filter_modality else None
            results[idx] = self.retrieve_for_index(
                idx, 
                use_image_similarity,
                filter_modality=modality_filter
            )
        
        logger.info(f"Retrieved captions for {len(results)} queries")
        
        return results


class PrototypeRetriever(CaptionRetriever):
    """Retriever that uses prototype sampling with Test-Time Adaptation (TTA) for diverse retrieval."""
    
    def __init__(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        captions: List[str],
        image_ids: List[str],
        modalities: Optional[List[str]],
        prototype_indices: np.ndarray,
        top_k: int = 5
    ):
        """
        Initialize prototype-based retriever with TTA support.
        
        Args:
            image_embeddings: Full image embeddings (N, D)
            text_embeddings: Full text embeddings (N, D)
            captions: List of all captions
            image_ids: List of all image IDs
            modalities: List of modality labels
            prototype_indices: Indices of selected prototypes
            top_k: Number of captions to retrieve
        """
        super().__init__(
            image_embeddings, text_embeddings, captions, image_ids, modalities, top_k
        )
        
        self.prototype_indices = prototype_indices
        self.prototype_image_embeddings = image_embeddings[prototype_indices]
        self.prototype_text_embeddings = text_embeddings[prototype_indices]
        self.prototype_captions = [captions[i] for i in prototype_indices]
        self.prototype_ids = [image_ids[i] for i in prototype_indices]
        self.prototype_modalities = [modalities[i] for i in prototype_indices] if modalities else None
        
        logger.info(
            f"PrototypeRetriever initialized with {len(prototype_indices)} prototypes"
        )
    
    def test_time_adaptation(
        self,
        query_embedding: np.ndarray,
        learning_rate: float = 1e-3,
        num_steps: int = 10,
        top_k_for_loss: int = 2,
        weight_variance: float = 0.1,
        weight_entropy: float = 0.01
    ) -> np.ndarray:
        """
        Apply test-time adaptation to query embedding using gradient-based optimization.
        
        The method learns a scaling vector that adapts the query embedding to better
        match the prototype distribution by optimizing three objectives:
        1. Maximize similarity with top-k prototypes
        2. Minimize variance in top-k similarities (promote consistency)
        3. Maximize entropy of similarity distribution (promote diversity)
        
        Args:
            query_embedding: Query embedding to adapt (D,)
            learning_rate: Learning rate for SGD optimizer
            num_steps: Number of optimization steps
            top_k_for_loss: Number of top similar prototypes to consider in loss
            weight_variance: Weight for variance regularization term
            weight_entropy: Weight for entropy regularization term
            
        Returns:
            Adapted query embedding (D,)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Returning original embedding.")
            return query_embedding
        
        # Convert to torch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        query_vec = torch.from_numpy(query_embedding).float().to(device)
        prototype_vecs = torch.from_numpy(self.prototype_image_embeddings).float().to(device)
        
        # Initialize learnable scaling vector
        scale_vec = torch.ones_like(query_vec, requires_grad=True)
        optimizer = torch.optim.SGD([scale_vec], lr=learning_rate)
        
        logger.debug(f"Memulai adaptasi TTA: {num_steps} steps, lr={learning_rate}")

        for step in range(num_steps):
            optimizer.zero_grad()

            # Apply element-wise scaling (Hadamard product)
            adapted_vec = query_vec * scale_vec

            # Normalize to unit length
            adapted_vec = F.normalize(adapted_vec, p=2, dim=-1)

            # Compute cosine similarities with all prototypes
            # cosine_similarity for batch: (N, D) x (D,) -> (N,)
            similarities = F.cosine_similarity(
                prototype_vecs,
                adapted_vec.unsqueeze(0),
                dim=-1
            )

            # Get top-k similarities for loss computation
            top_k = min(top_k_for_loss, len(similarities))
            topk_sims, _ = torch.topk(similarities, top_k)

            # Loss component 1: Maximize mean similarity with top-k
            mean_similarity = torch.mean(topk_sims)

            # Loss component 2: Minimize variance (promote consistent high similarity)
            variance_loss = torch.var(topk_sims)

            # Loss component 3: Maximize entropy (promote diversity in attention)
            # Softmax to convert similarities to probability distribution
            probs = F.softmax(similarities, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9))

            # Combined loss: negative similarity + variance penalty + entropy penalty
            loss = -mean_similarity + weight_variance * variance_loss + weight_entropy * entropy

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Log setiap iterasi untuk tracking
            logger.info(
                f"[TTA] step={step+1}/{num_steps}, "
                f"loss_total={loss.item():.4f}, "
                f"sim={mean_similarity.item():.4f}, "
                f"var={variance_loss.item():.4f}, "
                f"entropy={entropy.item():.4f}"
            )
        
        # Apply final scaling and normalize
        with torch.no_grad():
            adapted_embedding = query_vec * scale_vec
            adapted_embedding = F.normalize(adapted_embedding, p=2, dim=-1)
        
        # Convert back to numpy
        adapted_np = adapted_embedding.cpu().numpy()
        
        logger.debug(
            f"Adaptation complete. Embedding change: "
            f"{np.linalg.norm(adapted_np - query_embedding):.4f}"
        )
        
        return adapted_np
    
    def retrieve_by_image_embedding(
        self,
        query_embedding: np.ndarray,
        exclude_indices: Optional[List[int]] = None,
        filter_modality: Optional[str] = None,
        use_adaptation: bool = False,
        adaptation_params: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve from prototypes with Test-Time Adaptation (TTA).
        
        CRITICAL: TTA is applied FIRST, then retrieval uses the adapted embedding.
        This follows the thesis methodology: TTA → Retrieval → Generation
        
        Args:
            query_embedding: Query image embedding (will be adapted via TTA)
            exclude_indices: Original indices to exclude (mapped to prototype space)
            filter_modality: Optional modality filter (e.g., 'XR', 'CT')
            use_adaptation: Whether to apply test-time adaptation
            adaptation_params: Parameters for test-time adaptation
                - learning_rate: float (default: 1e-3)
                - num_steps: int (default: 10)
                - top_k_for_loss: int (default: 2)
                - weight_variance: float (default: 0.1)
                - weight_entropy: float (default: 0.01)
            
        Returns:
            Retrieved prototype captions with metadata
        """
        # STEP 1: Apply test-time adaptation FIRST (before retrieval)
        adapted_embedding = query_embedding
        if use_adaptation:
            params = adaptation_params or {}
            logger.debug("Running TTA before retrieval...")
            adapted_embedding = self.test_time_adaptation(query_embedding, **params)
        
        # STEP 2: Filter prototypes by modality if requested
        if filter_modality and self.prototype_modalities:
            valid_proto_indices = [
                i for i, mod in enumerate(self.prototype_modalities)
                if mod == filter_modality
            ]
            
            if not valid_proto_indices:
                logger.warning(f"No prototypes found for modality '{filter_modality}'")
                return []
            
            filtered_embeddings = self.prototype_image_embeddings[valid_proto_indices]
            logger.debug(f"Filtered to {len(valid_proto_indices)} prototypes with modality '{filter_modality}'")
        else:
            valid_proto_indices = list(range(len(self.prototype_image_embeddings)))
            filtered_embeddings = self.prototype_image_embeddings
        
        # STEP 3: Compute similarities with (filtered) prototype embeddings using adapted query
        similarities = self.compute_cosine_similarity(
            adapted_embedding, filtered_embeddings
        )
        
        # Ensure similarities is always a 1D array (handle edge case of single sample)
        similarities = np.atleast_1d(similarities)
        
        # STEP 4: Exclude specified indices (map to filtered prototype space)
        if exclude_indices:
            for orig_idx in exclude_indices:
                if orig_idx in self.prototype_indices:
                    # Find position in prototype list
                    proto_idx = np.where(self.prototype_indices == orig_idx)[0]
                    if len(proto_idx) > 0:
                        proto_idx = proto_idx[0]
                        # Check if it's in valid (filtered) indices
                        if proto_idx in valid_proto_indices:
                            filtered_idx = valid_proto_indices.index(proto_idx)
                            similarities[filtered_idx] = -np.inf
        
        # STEP 5: Get top-k in filtered space
        k = min(self.top_k, len(similarities))
        top_k_filtered_indices = np.argsort(similarities)[-k:][::-1]
        
        # STEP 6: Map back and build results
        results = []
        for filtered_idx in top_k_filtered_indices:
            proto_idx = valid_proto_indices[filtered_idx]
            orig_idx = self.prototype_indices[proto_idx]
            
            result = {
                'index': int(orig_idx),
                'image_id': self.prototype_ids[proto_idx],
                'caption': self.prototype_captions[proto_idx],
                'similarity': float(similarities[filtered_idx])
            }
            
            if self.prototype_modalities:
                result['modality'] = self.prototype_modalities[proto_idx]
            
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test retrieval
    import yaml
    from pathlib import Path
    
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
        data = np.load(embeddings_file, allow_pickle=True)
        
        retriever = CaptionRetriever(
            image_embeddings=data['image_embeddings'],
            text_embeddings=data['text_embeddings'],
            captions=data['captions'].tolist(),
            image_ids=data['image_ids'].tolist(),
            top_k=config['retrieval']['top_k']
        )
        
        # Test retrieval for first sample
        print(f"\nQuery: {retriever.captions[0]}")
        print("\nRetrieved captions:")
        
        results = retriever.retrieve_for_index(0)
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['similarity']:.3f}] {result['caption']}")
        
        print("\nFormatted context:")
        context = retriever.format_retrieved_context(results)
        print(context)
