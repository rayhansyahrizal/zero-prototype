"""
Data loader for ROCO medical imaging dataset.
Loads images and captions, handles preprocessing.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def extract_modality(caption: str, image_id: str = "") -> str:
    """
    Extract medical image modality from caption or image ID.
    
    Looks for common modality indicators like:
    - X-ray, XR, radiograph
    - CT, computed tomography
    - MRI, magnetic resonance
    - Ultrasound, US
    - PET, positron emission
    
    Args:
        caption: Image caption text
        image_id: Image identifier
        
    Returns:
        Modality string (XR, CT, MRI, US, PET, or UNKNOWN)
    """
    # Combine caption and ID for searching
    text = f"{caption} {image_id}".lower()
    
    # Define modality patterns (order matters - more specific first)
    modality_patterns = [
        (r'\b(x-ray|xray|radiograph|chest\s+film)\b', 'XR'),
        (r'\b(ct|computed\s+tomography|cat\s+scan)\b', 'CT'),
        (r'\b(mri|magnetic\s+resonance|nmr)\b', 'MRI'),
        (r'\b(ultrasound|sonography|us|echo)\b', 'US'),
        (r'\b(pet|positron\s+emission)\b', 'PET'),
        (r'\b(mammography|mammogram)\b', 'MG'),
        (r'\b(angiography|angiogram)\b', 'ANGIO'),
    ]
    
    for pattern, modality in modality_patterns:
        if re.search(pattern, text):
            return modality
    
    return 'UNKNOWN'


class ROCODataLoader:
    """Loader for ROCO (Radiology Objects in COntext) dataset."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        modality: str = "radiology",
        max_samples: Optional[int] = None
    ):
        """
        Initialize ROCO data loader.
        
        Args:
            root_dir: Root directory of ROCO dataset
            split: Dataset split ("train", "validation", "test")
            modality: Image modality ("radiology" or "non-radiology")
            max_samples: Maximum number of samples to load (None = all)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.modality = modality
        self.max_samples = max_samples
        
        # Set paths
        self.data_dir = self.root_dir / "data" / split / modality
        self.images_dir = self.data_dir / "images"
        self.captions_file = self.data_dir / "captions.txt"
        
        # Load captions
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {split}/{modality}")
        
    def _load_samples(self) -> List[Dict[str, str]]:
        """
        Load image-caption pairs from dataset.
        
        Returns:
            List of dictionaries containing image_id, image_path, and caption
        """
        if not self.captions_file.exists():
            logger.error(f"Captions file not found: {self.captions_file}")
            raise FileNotFoundError(f"Captions file not found: {self.captions_file}")
        
        samples = []
        
        with open(self.captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse line: "IMAGE_ID\tCAPTION"
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    logger.warning(f"Skipping malformed line: {line[:50]}...")
                    continue
                
                image_id, caption = parts
                image_path = self.images_dir / f"{image_id}.jpg"
                
                # Check if image exists (skip if not downloaded yet)
                if not image_path.exists():
                    # Try alternative extensions
                    for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                        alt_path = self.images_dir / f"{image_id}{ext}"
                        if alt_path.exists():
                            image_path = alt_path
                            break
                    else:
                        # Image not found, skip
                        continue
                
                # Extract modality from caption
                modality = extract_modality(caption, image_id)
                
                samples.append({
                    'image_id': image_id,
                    'image_path': str(image_path),
                    'caption': caption.strip(),
                    'modality': modality
                })
                
                # Limit samples if specified
                if self.max_samples and len(samples) >= self.max_samples:
                    break
        
        if len(samples) == 0:
            logger.warning(
                f"No valid samples found. Images may not be downloaded yet. "
                f"Expected location: {self.images_dir}"
            )
        
        return samples
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image_id, image_path, and caption
        """
        return self.samples[idx]
    
    def get_all_samples(self) -> List[Dict[str, str]]:
        """Return all samples."""
        return self.samples
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load and return PIL Image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image in RGB format
        """
        try:
            img = Image.open(image_path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def get_captions(self) -> List[str]:
        """Return list of all captions."""
        return [sample['caption'] for sample in self.samples]
    
    def get_image_ids(self) -> List[str]:
        """Return list of all image IDs."""
        return [sample['image_id'] for sample in self.samples]


def check_dataset_availability(config: dict) -> Tuple[bool, str]:
    """
    Check if dataset is available and ready to use.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_available, message)
    """
    dataset_config = config['dataset']
    root_dir = Path(dataset_config['root_dir'])
    split = dataset_config['split']
    modality = dataset_config['modality']
    
    data_dir = root_dir / "data" / split / modality
    images_dir = data_dir / "images"
    captions_file = data_dir / "captions.txt"
    
    # Check if captions file exists
    if not captions_file.exists():
        return False, f"Captions file not found: {captions_file}"
    
    # Check if images directory exists
    if not images_dir.exists():
        return False, f"Images directory not found: {images_dir}"
    
    # Count available images
    image_files = list(images_dir.glob("*.*"))
    num_images = len(image_files)
    
    if num_images == 0:
        return False, (
            f"No images found in {images_dir}. "
            "Images may still be downloading. Please wait for download to complete."
        )
    
    return True, f"Dataset ready: {num_images} images available"


if __name__ == "__main__":
    # Test data loader
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check dataset
    available, message = check_dataset_availability(config)
    print(f"\nDataset status: {message}\n")
    
    if available:
        # Load dataset
        loader = ROCODataLoader(
            root_dir=config['dataset']['root_dir'],
            split=config['dataset']['split'],
            modality=config['dataset']['modality'],
            max_samples=config['dataset']['max_samples']
        )
        
        print(f"Loaded {len(loader)} samples")
        
        if len(loader) > 0:
            # Show first sample
            sample = loader[0]
            print(f"\nFirst sample:")
            print(f"  Image ID: {sample['image_id']}")
            print(f"  Caption: {sample['caption']}")
            print(f"  Path: {sample['image_path']}")
            
            # Try loading image
            img = loader.load_image(sample['image_path'])
            print(f"  Image size: {img.size}")
