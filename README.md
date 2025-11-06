# Zero-Shot Medical Image Captioning MVP

A complete pipeline for zero-shot medical image captioning using **MedImageInsight** (embedding), **prototype sampling** (retriever), and **BLIP2** (generator).

## 🎯 Features

- **Three Captioning Methods:**
  - Baseline: BLIP2 only
  - Retrieval: BLIP2 + retrieved similar captions
  - Prototype: BLIP2 + prototype sampling for diverse context

- **Complete Pipeline:**
  - Medical image embedding with MedImageInsight
  - Farthest-point prototype sampling
  - Cosine similarity-based retrieval
  - Caption generation with BLIP2
  - Evaluation using BLEU and METEOR metrics

- **Interactive UI:**
  - Gradio interface for testing
  - Upload images or select from test set
  - Compare all three methods side-by-side
  - View retrieved contexts and evaluation metrics

## 📁 Project Structure

```
zero-prototype/
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies
├── src/
│   ├── main.py             # Main pipeline script
│   ├── data_loader.py      # ROCO dataset loader
│   ├── embedding.py        # MedImageInsight embeddings
│   ├── sampling.py         # Prototype sampling
│   ├── retrieval.py        # Caption retrieval
│   ├── generation.py       # BLIP2 caption generation
│   └── evaluation.py       # BLEU & METEOR metrics
├── ui/
│   └── app.py              # Gradio interface
├── data/
│   └── embeddings/         # Cached embeddings
├── results/
│   ├── captions.json       # Generated captions
│   └── metrics.csv         # Evaluation results
├── MedImageInsights/       # MedImageInsight model
└── roco-dataset/           # ROCO dataset
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Make sure PyTorch is installed with appropriate CUDA version if using GPU.

### 2. Check Environment

Verify that all resources are available:

```bash
python src/main.py --check-only
```

This will check:
- ✓ ROCO dataset images and captions
- ✓ MedImageInsight model
- ✓ BLIP2 model (cached or will download)

### 3. Run Pipeline

Run the complete pipeline (generates embeddings, samples prototypes, generates captions, evaluates):

```bash
python src/main.py
```

**Options:**
- `--force-regenerate`: Force regeneration of cached embeddings/prototypes
- `--skip-embedding`: Use cached embeddings
- `--skip-generation`: Use cached generation results
- `--config path/to/config.yaml`: Use custom config file

**First run will:**
1. Load dataset (500-1000 images)
2. Generate embeddings with MedImageInsight (~5-10 min)
3. Sample prototypes using farthest-point sampling
4. Generate captions with BLIP2 (~10-20 min depending on dataset size)
5. Evaluate using BLEU and METEOR
6. Save results to `results/`

### 4. Launch UI

Start the interactive Gradio interface:

```bash
python ui/app.py
```

Then open http://localhost:7860 in your browser.

**Options:**
- `--share`: Create public shareable link
- `--port 7860`: Specify port

## 📊 Pipeline Steps

### Step 1: Embedding Generation
- Uses MedImageInsight to encode images and captions
- Generates normalized 512-d vectors
- Cached in `data/embeddings/embeddings.npz`

### Step 2: Prototype Sampling
- Farthest-point sampling for diverse prototypes
- Selects ~100 representative samples
- Maximizes coverage of the embedding space
- Cached in `data/prototypes.npy`

### Step 3: Retrieval
- Cosine similarity between query and database
- Retrieves top-k (default: 5) similar captions
- Two modes: regular retrieval vs. prototype-only retrieval

### Step 4: Caption Generation
- **Baseline**: Direct BLIP2 generation
- **Retrieval**: BLIP2 with retrieved context
- **Prototype**: BLIP2 with prototype context
- All use beam search (num_beams=5)

### Step 5: Evaluation
- BLEU-1 to BLEU-4 scores
- METEOR score
- Comparison across all methods
- Results saved to `results/metrics.csv`

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
dataset:
  max_samples: 1000          # Limit dataset size
  split: "test"              # train/validation/test

retrieval:
  top_k: 5                   # Number of captions to retrieve

sampling:
  num_prototypes: 100        # Number of prototypes
  method: "farthest_point"   # or "random"

blip2:
  model_name: "Salesforce/blip2-opt-2.7b"
  max_length: 50
  num_beams: 5
```

## 📈 Expected Results

Example metrics (will vary by dataset):

| Method    | BLEU-4 | METEOR |
|-----------|--------|--------|
| Baseline  | 0.12   | 0.18   |
| Retrieval | 0.15   | 0.21   |
| Prototype | 0.16   | 0.22   |

Prototype sampling typically shows:
- Better diversity in retrieved context
- More relevant medical terminology
- Improved zero-shot performance

## 🔧 Individual Module Testing

Test each module independently:

```bash
# Test data loader
python src/data_loader.py

# Test embedding generation (small subset)
python src/embedding.py

# Test retrieval
python src/retrieval.py

# Test prototype sampling
python src/sampling.py

# Test caption generation (3 samples)
python src/generation.py

# Test evaluation
python src/evaluation.py
```

## 📝 Output Files

After running the pipeline:

- `data/embeddings/embeddings.npz`: Cached embeddings
- `data/prototypes.npy`: Selected prototype indices
- `results/captions.json`: All generated captions
- `results/metrics.csv`: Evaluation metrics
- `results/pipeline.log`: Execution log

## 🐛 Troubleshooting

### Images not found
Wait for ROCO dataset download to complete. Check:
```bash
ls roco-dataset/data/test/radiology/images/ | wc -l
```

### CUDA out of memory
Reduce batch size in embedding generation or use CPU:
```yaml
blip2:
  device: "cpu"
```

### BLIP2 download fails
Ensure internet connection. Model will be downloaded on first use (~5GB).

### Missing NLTK resources
Run:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## 🎯 Usage Examples

### Generate captions for custom image
```python
from PIL import Image
from src.generation import CaptionGenerator
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

generator = CaptionGenerator(config)
generator.load_model()

image = Image.open('path/to/medical_image.jpg')
caption = generator.generate_caption(image)
print(caption)
```

### Retrieve similar captions
```python
import numpy as np
from src.retrieval import CaptionRetriever

# Load embeddings
data = np.load('data/embeddings/embeddings.npz', allow_pickle=True)

retriever = CaptionRetriever(
    image_embeddings=data['image_embeddings'],
    text_embeddings=data['text_embeddings'],
    captions=data['captions'].tolist(),
    image_ids=data['image_ids'].tolist(),
    top_k=5
)

# Retrieve for first image
results = retriever.retrieve_for_index(0)
for r in results:
    print(f"{r['similarity']:.3f}: {r['caption']}")
```

## 📚 Dependencies

Core:
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- PIL, numpy, pandas

Models:
- MedImageInsight (local)
- BLIP2 (Salesforce/blip2-opt-2.7b)

Evaluation:
- nltk
- evaluate

UI:
- gradio >= 4.0.0

## 🔗 Resources

- MedImageInsight: Medical vision-language model
- BLIP2: Bootstrapped vision-language pretraining
- ROCO: Radiology Objects in COntext dataset

## ⚡ Performance Tips

1. **Use GPU**: Significant speedup for BLIP2 generation
2. **Cache embeddings**: Only regenerate when dataset changes
3. **Reduce dataset size**: Start with max_samples=100 for testing
4. **Batch processing**: Embeddings are batched automatically

## 📄 License

This is an MVP for research/educational purposes. Please check licenses for:
- MedImageInsight model
- BLIP2 model
- ROCO dataset
