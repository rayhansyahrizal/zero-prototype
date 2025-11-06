"""
Data loader for Hugging Face ROCOv2 and other medical imaging datasets.
Supports streaming and caching from Hugging Face Hub.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
from PIL import Image
import logging
import io

logger = logging.getLogger(__name__)


class HuggingFaceDataLoader:
    """Loader for medical imaging datasets on Hugging Face Hub."""
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        stream: bool = False
    ):
        """
        Initialize Hugging Face data loader.
        
        Args:
            dataset_name: Hugging Face dataset identifier (e.g., "eltorio/ROCOv2-radiology")
            split: Dataset split (e.g., "test", "train", "validation")
            max_samples: Maximum number of samples to load (None = all)
            cache_dir: Directory to cache downloaded data (defaults to ~/.cache/huggingface)
            stream: If True, stream data without downloading all (useful for large datasets)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not installed. "
                "Install with: pip install datasets"
            )
        
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        self.stream = stream
        
        logger.info(
            f"Loading dataset: {dataset_name} (split={split}, "
            f"stream={stream}, cache_dir={cache_dir})"
        )
        
        # Load dataset from Hugging Face
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            streaming=stream,
            trust_remote_code=True
        )
        
        # If max_samples specified and not streaming, select subset
        if max_samples and not stream:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        logger.info(
            f"Successfully loaded {len(self.dataset) if not stream else 'streaming'} "
            f"samples from {dataset_name}/{split}"
        )
        
        # Inspect dataset structure
        self._inspect_dataset()
    
    def _inspect_dataset(self):
        """Inspect and log dataset structure."""
        if len(self.dataset) == 0:
            logger.warning("Dataset is empty!")
            return
        
        # Get first sample to understand structure
        first_sample = self.dataset[0]
        logger.info(f"Dataset columns: {list(first_sample.keys())}")
        
        # Log sample structure
        for key, value in first_sample.items():
            if isinstance(value, Image.Image):
                logger.info(f"  {key}: Image {value.size}")
            elif isinstance(value, (bytes, bytearray)):
                logger.info(f"  {key}: bytes (length={len(value)})")
            elif isinstance(value, str):
                logger.info(f"  {key}: str (length={len(value)})")
            elif isinstance(value, dict):
                logger.info(f"  {key}: dict with keys {list(value.keys())}")
            elif isinstance(value, list):
                logger.info(f"  {key}: list (length={len(value)})")
            else:
                logger.info(f"  {key}: {type(value).__name__}")
    
    def __len__(self) -> int:
        """Return number of samples (0 if streaming)."""
        if self.stream:
            logger.warning("Dataset is in streaming mode; length is unknown")
            return 0
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample by index."""
        if self.stream:
            raise ValueError("Cannot index streaming dataset; use iterate() instead")
        return self.dataset[idx]
    
    def iterate(self, max_samples: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        Iterate through dataset samples.
        
        Args:
            max_samples: Limit iteration to N samples
            
        Yields:
            Dictionary containing sample data
        """
        count = 0
        for sample in self.dataset:
            if max_samples and count >= max_samples:
                break
            yield sample
            count += 1
    
    def get_all_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Get all samples as list.
        
        Args:
            max_samples: Limit to N samples
            
        Returns:
            List of sample dictionaries
        """
        if self.stream:
            samples = []
            for i, sample in enumerate(self.iterate()):
                if max_samples and i >= max_samples:
                    break
                samples.append(sample)
            return samples
        else:
            if max_samples:
                return [self.dataset[i] for i in range(min(max_samples, len(self.dataset)))]
            return [self.dataset[i] for i in range(len(self.dataset))]
    
    def load_image(self, image_data) -> Image.Image:
        """
        Load image from various formats.
        
        Args:
            image_data: PIL Image, bytes, or file path
            
        Returns:
            PIL Image in RGB format
        """
        try:
            if isinstance(image_data, Image.Image):
                img = image_data
            elif isinstance(image_data, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str):
                img = Image.open(image_data)
            else:
                raise TypeError(f"Unsupported image data type: {type(image_data)}")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
    
    def normalize_sample(self, sample: Dict) -> Dict[str, str]:
        """
        Normalize sample to standard format: {image_id, image, caption}.
        
        This handles different dataset schemas.
        
        Args:
            sample: Raw sample from dataset
            
        Returns:
            Normalized dictionary with standard keys
        """
        normalized = {}
        
        # Detect and map image field
        image_keys = [k for k in sample.keys() if 'image' in k.lower()]
        if image_keys:
            normalized['image'] = sample[image_keys[0]]
        
        # Detect and map caption field
        caption_keys = [k for k in sample.keys() if any(
            term in k.lower() for term in ['caption', 'text', 'description', 'label']
        )]
        if caption_keys:
            normalized['caption'] = str(sample[caption_keys[0]])
        
        # Detect and map ID field
        id_keys = [k for k in sample.keys() if 'id' in k.lower()]
        if id_keys:
            normalized['image_id'] = str(sample[id_keys[0]])
        else:
            normalized['image_id'] = f"sample_{id(sample)}"
        
        # Include any extra fields
        for key, value in sample.items():
            if key not in normalized and not isinstance(value, Image.Image):
                normalized[f"metadata_{key}"] = value
        
        return normalized


def compare_datasets(
    local_config: Optional[Dict] = None,
    hf_dataset_name: str = "eltorio/ROCOv2-radiology"
) -> Dict:
    """
    Compare local ROCO dataset with Hugging Face version.
    
    Args:
        local_config: Configuration dict with dataset paths
        hf_dataset_name: Hugging Face dataset identifier
        
    Returns:
        Comparison dictionary with stats about both datasets
    """
    comparison = {
        "local": None,
        "huggingface": None,
        "recommendation": None
    }
    
    # Try loading local dataset
    if local_config:
        try:
            from data_loader import ROCODataLoader, check_dataset_availability
            
            available, msg = check_dataset_availability(local_config)
            if available:
                loader = ROCODataLoader(
                    root_dir=local_config['dataset']['root_dir'],
                    split=local_config['dataset']['split'],
                    modality=local_config['dataset']['modality'],
                    max_samples=100  # Quick check
                )
                comparison["local"] = {
                    "status": "available",
                    "samples": len(loader),
                    "message": msg
                }
            else:
                comparison["local"] = {
                    "status": "unavailable",
                    "samples": 0,
                    "message": msg
                }
        except Exception as e:
            comparison["local"] = {
                "status": "error",
                "samples": 0,
                "message": str(e)
            }
    
    # Try loading Hugging Face dataset (quick check)
    try:
        hf_loader = HuggingFaceDataLoader(
            dataset_name=hf_dataset_name,
            split="test",
            max_samples=10,  # Quick check
            stream=False
        )
        comparison["huggingface"] = {
            "status": "available",
            "samples": len(hf_loader),
            "message": f"Successfully loaded from {hf_dataset_name}"
        }
    except Exception as e:
        comparison["huggingface"] = {
            "status": "error",
            "samples": 0,
            "message": str(e)
        }
    
    # Provide recommendation
    local_ok = comparison["local"] and comparison["local"]["status"] == "available"
    hf_ok = comparison["huggingface"] and comparison["huggingface"]["status"] == "available"
    
    if hf_ok and not local_ok:
        comparison["recommendation"] = "Use Hugging Face dataset (local not ready)"
    elif local_ok and hf_ok:
        comparison["recommendation"] = "Both available - use local for speed, HF for guaranteed access"
    elif local_ok:
        comparison["recommendation"] = "Use local dataset"
    else:
        comparison["recommendation"] = "Neither dataset available - check setup"
    
    return comparison


if __name__ == "__main__":
    import yaml
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("Testing Hugging Face ROCOv2 Dataset Loader")
    print("="*70 + "\n")
    
    # Test 1: Load HF dataset
    print("Test 1: Loading ROCOv2-radiology from Hugging Face...")
    print("-" * 70)
    try:
        hf_loader = HuggingFaceDataLoader(
            dataset_name="eltorio/ROCOv2-radiology",
            split="test",
            max_samples=5
        )
        print(f"✓ Successfully loaded {len(hf_loader)} samples")
        
        # Show first sample
        sample = hf_loader[0]
        normalized = hf_loader.normalize_sample(sample)
        print(f"\nFirst sample (normalized):")
        print(f"  Image ID: {normalized.get('image_id')}")
        print(f"  Caption: {normalized.get('caption')[:100]}...")
        
        # Try loading image
        if 'image' in normalized:
            img = hf_loader.load_image(normalized['image'])
            print(f"  Image size: {img.size}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n")
    
    # Test 2: Compare with local dataset
    print("Test 2: Comparing with local ROCO dataset...")
    print("-" * 70)
    try:
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        comparison = compare_datasets(config, hf_dataset_name="eltorio/ROCOv2-radiology")
        
        print(f"\nLocal Dataset:      {comparison['local']['status']}")
        print(f"  Samples: {comparison['local']['samples']}")
        print(f"  Message: {comparison['local']['message']}")
        
        print(f"\nHugging Face Dataset: {comparison['huggingface']['status']}")
        print(f"  Samples: {comparison['huggingface']['samples']}")
        print(f"  Message: {comparison['huggingface']['message']}")
        
        print(f"\n✓ Recommendation: {comparison['recommendation']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*70 + "\n")
