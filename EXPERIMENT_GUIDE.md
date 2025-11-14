# TTA Experiment Guide

Comprehensive guide for running Test-Time Adaptation (TTA) experiments following the thesis experimental design document.

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Quick Start](#quick-start)
4. [Experiment Types](#experiment-types)
5. [Analysis & Visualization](#analysis--visualization)
6. [Configuration Guide](#configuration-guide)

---

## Overview

This experimental framework supports comprehensive evaluation of Test-Time Adaptation (TTA) in medical image captioning, following the experimental design:

- **Section 1**: Dataset setup for domain shift
- **Section 2**: Architecture and component comparison
- **Section 3**: Evaluation metrics (text, clinical, robustness)
- **Section 4**: TTA adaptation procedure
- **Section 5**: TTA ablation studies
- **Section 6**: User study (optional)

---

## File Structure

```
zero-prototype/
├── src/
│   ├── main.py                      # Original pipeline
│   ├── main_tta_experiment.py       # Enhanced TTA experiment pipeline ⭐ NEW
│   ├── experiment_runner.py         # Ablation study runner ⭐ NEW
│   ├── cross_domain_eval.py         # Cross-domain evaluation ⭐ NEW
│   ├── tta_analyzer.py              # TTA analysis & visualization ⭐ NEW
│   ├── generation.py                # Caption generation (BLIP2)
│   ├── retrieval.py                 # Retrieval + TTA implementation
│   ├── evaluation.py                # Evaluation metrics
│   └── ...
├── configs/                          # ⭐ NEW: Ablation configurations
│   ├── config_ablation_lr_low.yaml
│   ├── config_ablation_lr_high.yaml
│   ├── config_ablation_steps_5.yaml
│   ├── config_ablation_steps_20.yaml
│   └── config_ablation_no_regularization.yaml
├── config.yaml                       # Base configuration
└── results/                          # Experiment results
    ├── tta_analysis/                 # TTA convergence plots
    ├── ablation_studies/             # Ablation results
    └── cross_domain/                 # Cross-domain results
```

---

## Quick Start

### 1. Basic TTA Experiment (Pre-TTA vs Post-TTA Comparison)

Run the enhanced pipeline that compares performance before and after TTA:

```bash
# Run TTA experiment with default config
python -m src.main_tta_experiment --config config.yaml

# Use cached embeddings (faster)
python -m src.main_tta_experiment --config config.yaml --skip-embedding
```

**Output:**
- `results/captions_TIMESTAMP.json` - Generated captions for all modes
- `results/metrics_TIMESTAMP.csv` - Evaluation metrics
- `results/delta_metrics_TIMESTAMP.json` - Pre-TTA vs Post-TTA comparison ⭐
- `results/comparison_TIMESTAMP.csv` - Method comparison table

**What it tests:**
- Baseline BLIP2 (no retrieval)
- Retrieval (no TTA)
- Prototype WITHOUT TTA (PRE-TTA baseline)
- Prototype WITH TTA (POST-TTA)

### 2. TTA Ablation Studies

Run systematic ablation studies to find optimal TTA parameters:

#### Learning Rate Ablation

```bash
python -m src.experiment_runner \
    --base-config config.yaml \
    --study learning_rate \
    --results-dir results/ablation_lr
```

Tests: `lr ∈ {0.0001, 0.0005, 0.001, 0.005, 0.01}`

#### Adaptation Steps Ablation

```bash
python -m src.experiment_runner \
    --base-config config.yaml \
    --study num_steps \
    --results-dir results/ablation_steps
```

Tests: `steps ∈ {5, 10, 15, 20, 25}`

#### Regularization Ablation

```bash
python -m src.experiment_runner \
    --base-config config.yaml \
    --study regularization \
    --results-dir results/ablation_reg
```

Tests combinations of:
- `weight_variance ∈ {0.0, 0.05, 0.1, 0.2}`
- `weight_entropy ∈ {0.0, 0.005, 0.01, 0.02}`

#### Full Grid Search

```bash
python -m src.experiment_runner \
    --base-config config.yaml \
    --study full \
    --results-dir results/ablation_full
```

Tests all parameter combinations (27 experiments total).

**Output:**
- `results/ablation_*/STUDY_NAME_results.csv` - All experiment results
- `results/ablation_*/STUDY_NAME_results.json` - Detailed results
- `results/ablation_*/config_*.yaml` - Generated configs for each experiment

### 3. Cross-Domain Evaluation

Evaluate TTA effectiveness across domain shifts:

```bash
# ROCO → MIMIC-CXR (cross-domain)
python -m src.cross_domain_eval \
    --source-domain roco \
    --target-domain mimic \
    --use-tta

# In-domain baseline (ROCO → ROCO)
python -m src.cross_domain_eval \
    --source-domain roco \
    --target-domain roco \
    --use-tta
```

**Output:**
- `results/cross_domain/cross_domain_TIMESTAMP.json` - Full results including domain distance
- Shows performance degradation due to domain shift
- Shows TTA improvement in adapting to new domain

---

## Experiment Types

### A. In-Domain Evaluation (ROCO → ROCO)

**Purpose:** Establish baseline performance without domain shift.

```bash
python -m src.main_tta_experiment \
    --config config.yaml \
    --skip-embedding
```

**Expected Results:**
- TTA should show modest improvement (domain already matches)
- Validates TTA doesn't hurt performance on matched data

### B. Cross-Domain Evaluation (ROCO → MIMIC-CXR)

**Purpose:** Test TTA effectiveness on domain shift.

```bash
python -m src.cross_domain_eval \
    --source-domain roco \
    --target-domain mimic \
    --use-tta
```

**Expected Results:**
- Performance drop without TTA due to domain shift
- TTA should recover performance by adapting to target domain
- Larger Δ improvement compared to in-domain

### C. TTA Ablation Studies

**Purpose:** Find optimal TTA hyperparameters.

See [Quick Start](#quick-start) section for commands.

**Analysis:**
```python
# After running ablation study
import pandas as pd

# Load results
results = pd.read_csv('results/ablation_lr/learning_rate_results.csv')

# Find best config
best = results.loc[results['target_metric_value'].idxmax()]
print(f"Best learning rate: {best['tta.learning_rate']}")
print(f"Best BLEU-4: {best['target_metric_value']}")
```

### D. Modality-Specific Evaluation (CT → X-ray)

**Purpose:** Test zero-shot transfer across imaging modalities.

```bash
# Train on CT scans, test on X-rays
python -m src.main_tta_experiment \
    --config config.yaml \
    # TODO: Add modality filtering flags
```

---

## Analysis & Visualization

### TTA Convergence Analysis

The `tta_analyzer.py` module provides comprehensive TTA analysis:

```python
from src.tta_analyzer import TTAAnalyzer
from pathlib import Path

# Initialize analyzer
analyzer = TTAAnalyzer(save_dir=Path("results/tta_analysis"))

# Load metrics from experiment
analyzer.load_metrics("tta_metrics.json")

# Plot convergence curves
analyzer.plot_all_convergence_metrics()

# Generate full report
analyzer.generate_full_report(
    pre_tta_results=pre_tta_detailed_scores,
    post_tta_results=post_tta_detailed_scores,
    metric_key='bleu_4'
)
```

**Generated Plots:**
1. `convergence_loss_total.png` - Total loss over adaptation steps
2. `convergence_loss_similarity.png` - Similarity term
3. `convergence_loss_variance.png` - Variance regularization
4. `convergence_loss_entropy.png` - Entropy regularization
5. `convergence_embedding_change.png` - L2 norm of embedding change
6. `delta_dist_bleu_4.png` - Distribution of performance changes
7. `perf_vs_adapt_bleu_4.png` - Performance vs adaptation magnitude

### Per-Sample Analysis

Identify which samples benefit most from TTA:

```python
# Compute per-sample deltas
delta_df = analyzer.compute_per_sample_delta(pre_scores, post_scores)

# Top 10 most improved
print(delta_df.nlargest(10, 'delta'))

# Top 10 most degraded
print(delta_df.nsmallest(10, 'delta'))

# Statistics
print(f"Improved: {delta_df['improved'].sum()} / {len(delta_df)}")
print(f"Mean Δ: {delta_df['delta'].mean():.4f}")
```

---

## Configuration Guide

### Base Configuration (`config.yaml`)

Key TTA parameters:

```yaml
tta:
  enabled: true                # Enable/disable TTA
  learning_rate: 0.001         # SGD learning rate (ABLATION: 0.0001 - 0.01)
  num_steps: 10                # Adaptation steps (ABLATION: 5 - 25)
  top_k_for_loss: 2            # Number of prototypes in loss
  weight_variance: 0.1         # Variance regularization weight (ABLATION: 0.0 - 0.2)
  weight_entropy: 0.01         # Entropy regularization weight (ABLATION: 0.0 - 0.02)
```

### Dataset Configuration

```yaml
dataset:
  source: "huggingface"              # 'huggingface' or 'local'
  name: "eltorio/ROCOv2-radiology"  # HuggingFace dataset
  split: "test"                      # 'train', 'validation', or 'test'
  max_samples: 1000                  # Limit for quick testing
```

### Retrieval Configuration

```yaml
retrieval:
  top_k: 5                    # Number of retrieved captions
  filter_modality: true       # Filter by same modality (XR, CT, etc.)
```

### Pre-configured Ablation Configs

Located in `configs/`:

1. **`config_ablation_lr_low.yaml`** - Very low learning rate (0.0001)
2. **`config_ablation_lr_high.yaml`** - High learning rate (0.01)
3. **`config_ablation_steps_5.yaml`** - Few adaptation steps (5)
4. **`config_ablation_steps_20.yaml`** - Many adaptation steps (20)
5. **`config_ablation_no_regularization.yaml`** - No variance/entropy penalty

---

## Expected Results

### TTA Effectiveness Metrics

Based on the experimental design, expect:

1. **Δ BLEU-4**: +0.02 to +0.10 (2-10% relative improvement)
2. **Δ METEOR**: +0.01 to +0.05
3. **Δ Semantic Similarity**: +0.05 to +0.15

### Convergence Behavior

- Loss should decrease over 5-10 steps
- Embedding change: 0.01 - 0.10 (L2 norm)
- Similarity with top-k prototypes: increases
- Variance: decreases (more consistent attention)

### Domain Shift Impact

- **In-domain (ROCO → ROCO)**: Small TTA improvement (Δ ≈ +0.02)
- **Cross-domain (ROCO → MIMIC)**: Larger TTA improvement (Δ ≈ +0.05 to +0.10)
- Domain distance correlates with TTA benefit

---

## Troubleshooting

### Common Issues

1. **OOM (Out of Memory)**
   - Reduce `max_samples` in config
   - Reduce `num_beams` in BLIP2 config
   - Use smaller model variant

2. **Slow experiments**
   - Use `--skip-embedding` after first run
   - Reduce `num_steps` in TTA config
   - Use fewer ablation points

3. **Poor TTA convergence**
   - Try different learning rates (start with 0.001)
   - Increase `num_steps` to 15-20
   - Check regularization weights

4. **No improvement from TTA**
   - Check if dataset has domain shift (use cross-domain eval)
   - Verify prototypes are diverse (check sampling)
   - Try disabling regularization

---

## Citation

If you use this experimental framework, please cite:

```bibtex
@thesis{rayhan2025tta,
  title={Test-Time Adaptation for Medical Image Captioning},
  author={Rayhan},
  year={2025},
  school={Your University}
}
```

---

## Next Steps

1. **Run baseline experiments** - Establish performance without TTA
2. **Run TTA experiments** - Measure TTA effectiveness
3. **Run ablation studies** - Find optimal hyperparameters
4. **Cross-domain evaluation** - Test robustness to domain shift
5. **Analyze results** - Generate plots and statistical analysis

For questions or issues, see [README.md](README.md) or open an issue on GitHub.
