# TTA Experiments - Quick Reference Card

## 🚀 Essential Commands

### 1. Basic TTA Experiment (START HERE!)
```bash
python -m src.main_tta_experiment --config config.yaml --skip-embedding
```
**Output:** Delta metrics showing TTA improvement
**Time:** ~30 min (500 samples)

---

### 2. Ablation Studies

```bash
# Learning rate ablation
python -m src.experiment_runner --study learning_rate --skip-embedding

# Adaptation steps ablation
python -m src.experiment_runner --study num_steps --skip-embedding

# Regularization ablation
python -m src.experiment_runner --study regularization --skip-embedding

# Full grid search (27 experiments)
python -m src.experiment_runner --study full --skip-embedding
```
**Time:** 30 min per ablation, 4 hours for full grid

---

### 3. Cross-Domain Evaluation

```bash
# In-domain (baseline)
python -m src.cross_domain_eval --source roco --target roco --use-tta

# Cross-domain (domain shift)
python -m src.cross_domain_eval --source roco --target mimic --use-tta
```
**Time:** ~30 min per domain pair

---

### 4. Compare Results

```bash
python scripts/compare_experiments.py results/metrics_*.csv
```

---

## 📊 Quick Analysis

### Check Delta Metrics
```python
import json

# Load delta metrics
with open('results/delta_metrics_TIMESTAMP.json') as f:
    delta = json.load(f)

# Print TTA improvement
for metric, values in delta.items():
    print(f"{metric}: {values['delta']:+.4f} ({values['delta_pct']:+.2f}%)")
```

### Find Best Ablation Config
```python
import pandas as pd

results = pd.read_csv('results/ablation_lr/learning_rate_results.csv')
best = results.loc[results['target_metric_value'].idxmax()]
print(f"Best config: {best['tta.learning_rate']}")
```

### Plot Convergence
```python
from src.tta_analyzer import TTAAnalyzer
from pathlib import Path

analyzer = TTAAnalyzer(save_dir=Path("results/tta_analysis"))
analyzer.load_metrics("tta_metrics.json")
analyzer.plot_all_convergence_metrics()
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/main_tta_experiment.py` | Main TTA experiment pipeline |
| `src/experiment_runner.py` | Automated ablation studies |
| `src/cross_domain_eval.py` | Cross-domain evaluation |
| `src/tta_analyzer.py` | Analysis & visualization |
| `configs/*.yaml` | Pre-configured ablation configs |
| `EXPERIMENT_GUIDE.md` | Full documentation |
| `EXPERIMENTS_CHECKLIST.md` | Experiment plan & progress |

---

## 🎯 Expected Results

### TTA Improvement (Typical)

| Metric | Pre-TTA | Post-TTA | Δ | Δ% |
|--------|---------|----------|---|-----|
| BLEU-4 | 0.300 | 0.320 | +0.020 | +6.7% |
| METEOR | 0.350 | 0.370 | +0.020 | +5.7% |
| Semantic | 0.700 | 0.730 | +0.030 | +4.3% |

### Best Hyperparameters (Typical)

- **Learning Rate:** 0.001 - 0.005
- **Num Steps:** 10 - 15
- **Weight Variance:** 0.05 - 0.1
- **Weight Entropy:** 0.005 - 0.01

---

## 🔧 Configuration Quick Reference

### Base Config (`config.yaml`)
```yaml
tta:
  enabled: true
  learning_rate: 0.001    # [0.0001 - 0.01]
  num_steps: 10           # [5 - 25]
  top_k_for_loss: 2       # Fixed
  weight_variance: 0.1    # [0.0 - 0.2]
  weight_entropy: 0.01    # [0.0 - 0.02]
```

### Dataset Config
```yaml
dataset:
  source: "huggingface"
  name: "eltorio/ROCOv2-radiology"
  split: "test"
  max_samples: 500        # Set to null for full dataset
```

---

## 🐛 Troubleshooting

### Issue: OOM Error
```yaml
# Solution: Reduce batch size in config
dataset:
  max_samples: 100  # Start small

blip2:
  num_beams: 3      # Reduce from 5
```

### Issue: Slow Experiments
```bash
# Solution: Use cached embeddings
--skip-embedding

# Solution: Reduce adaptation steps
tta:
  num_steps: 5  # Instead of 10
```

### Issue: No TTA Improvement
- Check if dataset has domain shift (try cross-domain eval)
- Try higher learning rate (0.005)
- Try more steps (15-20)
- Disable regularization (set weights to 0.0)

---

## 📈 Workflow Checklist

- [ ] Run basic TTA experiment (`main_tta_experiment.py`)
- [ ] Verify delta metrics show improvement
- [ ] Run learning rate ablation
- [ ] Run num_steps ablation
- [ ] Run regularization ablation
- [ ] Identify best configuration
- [ ] Run full evaluation with best config
- [ ] Generate all plots and tables
- [ ] (Optional) Run cross-domain evaluation
- [ ] (Optional) Run user study

---

## 📞 Help

- **Full guide:** [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
- **Checklist:** [EXPERIMENTS_CHECKLIST.md](EXPERIMENTS_CHECKLIST.md)
- **Summary:** [NEW_FILES_SUMMARY.md](NEW_FILES_SUMMARY.md)

---

## 💡 Tips

1. **Always use `--skip-embedding`** after first run (uses cached embeddings)
2. **Start with small datasets** (`max_samples: 100`) for debugging
3. **Run ablation studies in parallel** if you have multiple GPUs
4. **Check logs** in `results/pipeline_tta_*.log` for debugging
5. **Backup results** before running new experiments

---

## 🎓 For Thesis

### Tables to Generate
1. Main results (4 methods comparison)
2. Ablation study results
3. Cross-domain results

### Plots to Generate
1. TTA convergence (loss vs steps)
2. Delta distribution (histogram)
3. Per-method comparison (bar chart)
4. Performance vs domain distance (scatter)

### Analysis to Include
- Statistical significance (paired t-test)
- Per-sample analysis (top improved/degraded)
- Failure case analysis

---

**Good luck! 🚀**
