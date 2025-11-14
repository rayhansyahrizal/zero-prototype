# Gradio UI for Medical Image Captioning

Quick interactive interface for testing different captioning methods.

## Features

- **Config Selection**: Test different experimental configs (Conservative, Moderate, No TTA, etc.)
- **Single Method Mode**: Test one method at a time with full control
- **Compare All Mode**: Generate all 4 methods at once with side-by-side comparison ⭐ NEW!
- **TTA Controls**: Adjust learning rate and steps in real-time
- **Similarity Filtering**: Filter low-quality retrievals with threshold slider
- **Live Evaluation**: Get BLEU-4, METEOR, and semantic similarity scores
- **Evaluation Table**: Automatic comparison table for all methods ⭐ NEW!

## Usage

### 1. Start the UI

```bash
python ui/app.py
```

Or with options:
```bash
python ui/app.py --port 8080 --share  # Share publicly
```

### 2. Load Models

1. Select a config (Default, Conservative TTA, etc.)
2. Click "Load Models"
3. Wait for all models to load (~2-3 minutes first time)

### 3. Generate Captions

**Option A: Upload Image**
- Upload your own medical image
- Enter ground truth caption (optional but recommended)

**Option B: Use Dataset**
- Go to "Dataset" tab
- Slide to select image index
- Click "Load"

**Option C: Single Method**
1. Go to "Single Method" tab
2. Select method (Baseline, Retrieval, Prototype, Prototype+TTA)
3. Adjust TTA settings if using TTA
4. Set similarity threshold if needed
5. Click "Generate Caption"

**Option D: Compare All Methods ⭐ NEW!**
1. Go to "Compare All Methods" tab
2. Adjust shared settings (TTA LR, Steps, Threshold)
3. Click "Generate All Methods"
4. Wait ~30-60 seconds
5. See all 4 methods side-by-side with evaluation table!

## Config Options

| Config | Description |
|--------|-------------|
| Default | Original TTA settings (LR=0.001, 10 steps) |
| Conservative TTA | Slower learning (LR=0.0001, 30 steps) |
| Moderate TTA | Balanced (LR=0.0005, 20 steps) |
| No TTA | Prototype without TTA (ablation) |
| More Prototypes | 300 prototypes instead of 100 |

## TTA Settings

**Learning Rate (0.00001 - 0.01)**
- Lower = more conservative, stable
- Higher = faster but may overshoot
- Default: 0.001

**Steps (1 - 50)**
- More steps = better convergence
- Default: 10

## Similarity Threshold

**Range: 0 - 1 (0 = off)**
- Filters retrieved captions below threshold
- Higher = stricter filtering
- Recommended: 0.5 - 0.6

## Tips

1. **Try configs first** to see which TTA setting works best
2. **Use threshold** to filter noisy retrievals (especially if retrieval seems off)
3. **Compare methods** side-by-side on same image
4. **Check evaluation scores** if you have ground truth

## Requirements

Before running UI:
- Run pipeline at least once to generate embeddings
- Prototypes should exist (or will use all embeddings)
- BLIP2 model cached (or will download first time)

```bash
# Generate embeddings first
python src/main.py --config config.yaml
```

## Port

Default: 7860

Access at: `http://localhost:7860`

## Compare All Methods

### What It Does

Generates captions for ALL 4 methods in one click:
- Baseline
- Retrieval
- Prototype
- Prototype+TTA

### Output

**Side-by-side captions**: See all generated captions at once

**Retrieved contexts**: View what each method retrieved (top-3)

**Evaluation table**: Automatic BLEU-4, METEOR, Semantic scores if ground truth provided

### Example Output

```
Method        | BLEU-4 | METEOR | Semantic
─────────────────────────────────────────
Baseline      | 0.1234 | 0.3456 | 0.7890
Retrieval     | 0.1456 | 0.3678 | 0.8012
Prototype     | 0.1389 | 0.3598 | 0.7945
Prototype+TTA | 0.1512 | 0.3789 | 0.8123  ← Best!
```

### Use Cases

**Thesis Demo**: Show advisor all methods at once
**Quick Testing**: Find which method works best for image type
**Paper Figures**: Screenshot comparison for thesis
**Config Testing**: Compare TTA settings efficiently

### Tips

- Provide ground truth for automatic evaluation
- Adjust shared settings before clicking generate
- Takes ~30-60 seconds (4x BLIP2 calls)
- Perfect for presentations!

## Troubleshooting

**"Embeddings not found"**
- Run main pipeline first: `python src/main.py`

**"Prototype retriever not available"**
- Prototypes not generated yet
- Use Retrieval method instead, or generate prototypes

**UI slow/hanging**
- First generation takes time (model loading)
- TTA with many steps can be slow (expected)
- Compare All takes longer (4 generations)

**Out of memory**
- Reduce max_samples in config
- Use CPU instead of GPU (slower but works)
