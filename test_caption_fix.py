"""
Quick test to verify the caption generation fix.
Tests that retrieved captions are properly generated, not just echoing prompts.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generation import CaptionGenerator

def test_caption_decoding():
    """Test that captions are generated correctly with context."""
    print("=" * 80)
    print("Testing Caption Generation Fix")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent / "config_prototype_no_tta.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("\n1. Loading BLIP2 model...")
    generator = CaptionGenerator(config)
    generator.load_model()
    print("   ✅ Model loaded")

    # Create a dummy test image (white square)
    test_image = Image.new('RGB', (224, 224), color='white')

    # Test 1: Baseline (no context)
    print("\n2. Testing BASELINE generation (no context)...")
    caption_baseline = generator.generate_caption(test_image)
    print(f"   Caption: {caption_baseline}")
    print(f"   Length: {len(caption_baseline)} chars")

    # Verify it's a proper caption (not a prompt)
    assert not caption_baseline.startswith("Based on the context"), \
        "❌ Baseline should not start with prompt text!"
    print("   ✅ Baseline OK")

    # Test 2: With retrieval context
    print("\n3. Testing RETRIEVAL generation (with context)...")

    # Simulate retrieved captions
    retrieved_captions = [
        {
            'caption': 'CT scan of the chest showing pneumonia',
            'similarity': 0.95
        },
        {
            'caption': 'X-ray image showing fractured rib',
            'similarity': 0.88
        },
        {
            'caption': 'MRI scan of the brain with contrast',
            'similarity': 0.82
        }
    ]

    caption_retrieval = generator.generate_caption(
        test_image,
        retrieved_captions=retrieved_captions
    )

    print(f"   Caption: {caption_retrieval}")
    print(f"   Length: {len(caption_retrieval)} chars")

    # Verify it's NOT the full prompt
    if caption_retrieval.startswith("Based on the context"):
        print("   ❌ FAILED: Caption contains the prompt!")
        print("   This means the fix didn't work - still returning prompt instead of generation")
        return False
    else:
        print("   ✅ SUCCESS: Caption is properly generated (doesn't echo prompt)")
        return True

    # Test 3: Check length sanity
    print("\n4. Sanity checks...")

    # Caption should be reasonable length (not 500+ chars like full prompt)
    if len(caption_retrieval) > 300:
        print(f"   ⚠️  WARNING: Caption is very long ({len(caption_retrieval)} chars)")
        print(f"   Might still include prompt. Expected: <100 chars")
    else:
        print(f"   ✅ Length OK ({len(caption_retrieval)} chars)")

    # Caption should be different from baseline (context should influence it)
    if caption_baseline == caption_retrieval:
        print("   ⚠️  WARNING: Baseline and retrieval captions are identical")
        print("   Context might not be influencing generation")
    else:
        print("   ✅ Captions are different (context is being used)")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        success = test_caption_decoding()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
