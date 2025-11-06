"""
Embedding generation using MedImageInsight model.
Encodes images and text into normalized feature vectors.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from typing import List, Dict, Optional
from PIL import Image
import base64
import io
import logging
from tqdm import tqdm

# Add MedImageInsights to path
MEDIMAGEINSIGHTS_PATH = Path(__file__).parent.parent / "MedImageInsights"
sys.path.insert(0, str(MEDIMAGEINSIGHTS_PATH))

from medimageinsightmodel import MedImageInsight

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using MedImageInsight model."""
    
    def __init__(self, config: dict):
        """
        Initialize embedding generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        self.embeddings_dir = Path(config['output']['embeddings_dir'])
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Embedding generator initialized on device: {self.device}")
    
    def load_model(self):
        """Load MedImageInsight model."""
        try:
            model_config = self.config['medimageinsight']
            
            logger.info("Loading MedImageInsight model...")
            self.model = MedImageInsight(
                model_dir=model_config['model_dir'],
                vision_model_name=model_config['vision_model_name'],
                language_model_name=model_config['language_model_name']
            )
            self.model.load_model()
            logger.info("MedImageInsight model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MedImageInsight model: {e}")
            raise
    
    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image
            
        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 8,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode images to embeddings.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Numpy array of shape (num_images, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch_images = images[i:i + batch_size]
            
            # Convert to base64 (required by MedImageInsight API)
            batch_base64 = [self.image_to_base64(img) for img in batch_images]
            
            # Encode
            output = self.model.encode(images=batch_base64)
            embeddings = output['image_embeddings']
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        # Normalize if requested
        if normalize:
            all_embeddings = self._normalize_embeddings(all_embeddings)
        
        return all_embeddings
    
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            
            # Encode
            output = self.model.encode(texts=batch_texts)
            embeddings = output['text_embeddings']
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        # Normalize if requested
        if normalize:
            all_embeddings = self._normalize_embeddings(all_embeddings)
        
        return all_embeddings
    
    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms
    
    def generate_and_save_embeddings(
        self,
        data_loader,
        force_regenerate: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for dataset and save to disk.
        
        Args:
            data_loader: ROCODataLoader instance
            force_regenerate: If True, regenerate even if cached embeddings exist
            
        Returns:
            Dictionary containing image_embeddings, text_embeddings, and metadata
        """
        # Check if embeddings already exist
        embeddings_file = self.embeddings_dir / "embeddings.npz"
        
        if embeddings_file.exists() and not force_regenerate:
            logger.info(f"Loading cached embeddings from {embeddings_file}")
            data = np.load(embeddings_file, allow_pickle=True)
            return {
                'image_embeddings': data['image_embeddings'],
                'text_embeddings': data['text_embeddings'],
                'image_ids': data['image_ids'].tolist(),
                'captions': data['captions'].tolist()
            }
        
        logger.info("Generating new embeddings...")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        samples = data_loader.get_all_samples()
        
        if len(samples) == 0:
            raise ValueError("No samples available in data loader")
        
        # Load images
        logger.info("Loading images...")
        images = []
        valid_samples = []
        
        for sample in tqdm(samples, desc="Loading images"):
            try:
                img = data_loader.load_image(sample['image_path'])
                images.append(img)
                valid_samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to load image {sample['image_id']}: {e}")
                continue
        
        if len(images) == 0:
            raise ValueError("No valid images could be loaded")
        
        logger.info(f"Loaded {len(images)} images")
        
        # Extract captions
        captions = [s['caption'] for s in valid_samples]
        image_ids = [s['image_id'] for s in valid_samples]
        
        # Generate embeddings
        logger.info("Generating image embeddings...")
        image_embeddings = self.encode_images(images, normalize=True)
        
        logger.info("Generating text embeddings...")
        text_embeddings = self.encode_texts(captions, normalize=True)
        
        # Save embeddings
        logger.info(f"Saving embeddings to {embeddings_file}")
        np.savez_compressed(
            embeddings_file,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            image_ids=np.array(image_ids),
            captions=np.array(captions)
        )
        
        logger.info(f"Embeddings saved: {image_embeddings.shape[0]} samples")
        
        return {
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            'image_ids': image_ids,
            'captions': captions
        }


if __name__ == "__main__":
    # Test embedding generation
    import yaml
    from data_loader import ROCODataLoader, check_dataset_availability
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check dataset
    available, message = check_dataset_availability(config)
    print(f"\nDataset status: {message}\n")
    
    if not available:
        print("Dataset not available. Exiting.")
        sys.exit(1)
    
    # Load dataset
    loader = ROCODataLoader(
        root_dir=config['dataset']['root_dir'],
        split=config['dataset']['split'],
        modality=config['dataset']['modality'],
        max_samples=10  # Test with small subset
    )
    
    # Generate embeddings
    generator = EmbeddingGenerator(config)
    embeddings = generator.generate_and_save_embeddings(loader)
    
    print(f"\nGenerated embeddings:")
    print(f"  Image embeddings shape: {embeddings['image_embeddings'].shape}")
    print(f"  Text embeddings shape: {embeddings['text_embeddings'].shape}")
    print(f"  Number of samples: {len(embeddings['image_ids'])}")
