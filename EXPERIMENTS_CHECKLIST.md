# TTA Experiments Checklist

Checklist untuk menyelesaikan semua eksperimen sesuai dokumen rancangan.

## 📋 Status Implementasi

### ✅ Sudah Implemented

- [x] **Main TTA Experiment Pipeline** (`src/main_tta_experiment.py`)
  - [x] Pre-TTA vs Post-TTA comparison
  - [x] Delta metrics computation
  - [x] All 4 modes (baseline, retrieval, prototype, prototype+TTA)

- [x] **TTA Analyzer** (`src/tta_analyzer.py`)
  - [x] Convergence tracking
  - [x] Visualization tools
  - [x] Per-sample delta analysis
  - [x] Performance vs adaptation plots

- [x] **Experiment Runner** (`src/experiment_runner.py`)
  - [x] Automated ablation studies
  - [x] Parameter grid search
  - [x] Results aggregation

- [x] **Cross-Domain Evaluator** (`src/cross_domain_eval.py`)
  - [x] Domain shift evaluation
  - [x] Domain distance computation (MMD)
  - [x] Cross-domain TTA testing

- [x] **Configuration Files**
  - [x] Learning rate variants
  - [x] Adaptation steps variants
  - [x] Regularization variants

---

## 🎯 Rencana Eksperimen (Sesuai Dokumen)

### **1. Setup Dataset untuk Domain Shift** (Bagian 1)

| Skema | Train → Test | Status | Command | Priority |
|-------|--------------|--------|---------|----------|
| **A. In-domain** | ROCO → ROCO | ✅ Ready | `python -m src.main_tta_experiment` | **HIGH** |
| **B. Cross-domain** | ROCO → MIMIC-CXR | ⚠️ Partial | `python -m src.cross_domain_eval --source roco --target mimic` | **MEDIUM** |
| **C. Style shift** | ROCO → ROCO-synthetic | ❌ Not ready | Need synthetic data | LOW |
| **D. Zero-shot** | CT → X-ray | ⚠️ Partial | Need modality split | MEDIUM |

**Action Items:**
- [ ] Verify MIMIC-CXR dataset availability
- [ ] Implement modality-based train/test splitting
- [ ] (Optional) Generate synthetic style-shifted data

---

### **2. Arsitektur dan Komponen** (Bagian 2)

| Model | Retrieval | TTA | Status | Notes |
|-------|-----------|-----|--------|-------|
| BLIP2 pretrained | ❌ | ❌ | ✅ Ready | Baseline |
| + Retrieval | ✅ | ❌ | ✅ Ready | Regular retrieval |
| + TTA only | ❌ | ✅ | ✅ Ready | Prototype + TTA |
| + Retrieval + TTA | ✅ | ✅ | ✅ Ready | Full pipeline |
| + RLCF-style TTA | ✅ | ✅ | ❌ Not implemented | SOTA comparison |

**Action Items:**
- [x] Implement all 4 basic modes (DONE)
- [ ] (Optional) Implement RLCF baseline for comparison

---

### **3. Evaluasi dan Metrik** (Bagian 3)

#### a. Evaluasi Teks ✅

- [x] BLEU-1/2/3/4
- [x] METEOR
- [x] ROUGE-L (available in evaluation.py)
- [x] CIDEr (available in evaluation.py)
- [x] BERTScore / Semantic Similarity (MedImageInsight)

#### b. Evaluasi Klinis ❌

- [ ] CheXpert label match accuracy/F1
- [ ] RadGraph F1 (entity structure)
- [ ] Entity-level recall

**Action Items:**
- [ ] Implement CheXpert label extractor
- [ ] Implement RadGraph F1 metric
- [ ] Add entity extraction and matching

#### c. Evaluasi Robustness dan Adaptasi ✅

- [x] Δ performance (pre vs post TTA)
- [x] Plot "performance vs. domain distance"
- [x] Plot convergence loss vs step

---

### **4. Prosedur Adaptasi TTA** (Bagian 4) ✅

- [x] Online per-image adaptation
- [x] Gradient-based optimization (SGD)
- [x] 1D scaling vector adaptation
- [x] Configurable loss functions
- [x] Regularization (variance + entropy)

---

### **5. Studi Ablasi TTA** (Bagian 5)

#### Jenis TTA (Type of Adaptation)

| Variant | Status | Config File | Priority |
|---------|--------|-------------|----------|
| 1D scaling vector | ✅ Ready | Default | **HIGH** |
| LoRA adaptation | ❌ Not implemented | - | MEDIUM |
| Additive vector | ❌ Not implemented | - | LOW |
| Layer-specific (Q/K/V) | ❌ Not implemented | - | LOW |

**Action Items:**
- [x] Implement 1D scaling vector (DONE)
- [ ] (Optional) Implement LoRA-based TTA
- [ ] (Optional) Implement additive vector TTA

#### Hyperparameter Ablation ✅

| Parameter | Values to Test | Config Files | Command |
|-----------|----------------|--------------|---------|
| Learning Rate | 0.0001, 0.0005, **0.001**, 0.005, 0.01 | `config_ablation_lr_*.yaml` | `experiment_runner.py --study learning_rate` |
| Num Steps | 5, **10**, 15, 20, 25 | `config_ablation_steps_*.yaml` | `experiment_runner.py --study num_steps` |
| Weight Variance | 0.0, 0.05, **0.1**, 0.2 | `config_ablation_no_reg.yaml` | `experiment_runner.py --study regularization` |
| Weight Entropy | 0.0, 0.005, **0.01**, 0.02 | `config_ablation_no_reg.yaml` | `experiment_runner.py --study regularization` |

**Bold** = default value

**Action Items:**
- [x] Create config files for variants (DONE)
- [ ] Run learning rate ablation
- [ ] Run num_steps ablation
- [ ] Run regularization ablation
- [ ] Analyze and report best configuration

---

### **6. User Study** (Bagian 6) - OPTIONAL

- [ ] Select 20-50 representative samples
- [ ] Generate pre-TTA and post-TTA captions
- [ ] Prepare evaluation form (quality 1-5, accuracy, completeness)
- [ ] Recruit radiologist/expert evaluators
- [ ] Statistical analysis (paired t-test)
- [ ] Visualize results (bar chart, heatmap)

**Status:** Not started (optional for publication)

---

## 📝 Eksperimen yang Harus Dijalankan

### Phase 1: Baseline & Validation ⭐ **START HERE**

```bash
# 1. Run basic TTA experiment (in-domain, small dataset)
python -m src.main_tta_experiment \
    --config config.yaml \
    --skip-embedding

# Expected output:
# - Baseline, Retrieval, Prototype (no TTA), Prototype (with TTA)
# - Delta metrics showing TTA improvement
# - Convergence plots
```

**Goal:** Validate that TTA works and improves performance.

**Expected Results:**
- BLEU-4: Baseline < Retrieval < Prototype < Prototype+TTA
- Δ BLEU-4 (TTA improvement): +0.02 to +0.05

---

### Phase 2: Ablation Studies

```bash
# 2a. Learning rate ablation
python -m src.experiment_runner \
    --base-config config.yaml \
    --study learning_rate \
    --skip-embedding

# 2b. Num steps ablation
python -m src.experiment_runner \
    --base-config config.yaml \
    --study num_steps \
    --skip-embedding

# 2c. Regularization ablation
python -m src.experiment_runner \
    --base-config config.yaml \
    --study regularization \
    --skip-embedding
```

**Goal:** Find optimal TTA hyperparameters.

**Analysis:**
```python
import pandas as pd

# Load results
results = pd.read_csv('results/ablation_lr/learning_rate_results.csv')

# Find best configuration
best = results.loc[results['target_metric_value'].idxmax()]
print(f"Best LR: {best['tta.learning_rate']}")
```

---

### Phase 3: Cross-Domain Evaluation

```bash
# 3a. In-domain baseline (ROCO → ROCO)
python -m src.cross_domain_eval \
    --source-domain roco \
    --target-domain roco \
    --use-tta

# 3b. Cross-domain (ROCO → MIMIC-CXR)
python -m src.cross_domain_eval \
    --source-domain roco \
    --target-domain mimic \
    --use-tta
```

**Goal:** Show TTA is more effective when domain shift exists.

**Expected Results:**
- In-domain: Small Δ improvement (~0.02)
- Cross-domain: Larger Δ improvement (~0.05-0.10)
- Domain distance correlates with TTA benefit

---

### Phase 4: Large-Scale Evaluation (Final)

```bash
# 4. Full evaluation with larger dataset
python -m src.main_tta_experiment \
    --config config.yaml \
    # Modify config: max_samples: 5000 or null (all data)
```

**Goal:** Final results for publication.

---

## 📊 Hasil yang Diharapkan

### Tabel Utama (Main Results Table)

| Method | BLEU-4 | METEOR | Semantic Sim | Δ vs Baseline |
|--------|--------|--------|--------------|---------------|
| Baseline | 0.25 | 0.30 | 0.65 | - |
| Retrieval | 0.28 | 0.33 | 0.68 | +0.03 |
| Prototype (no TTA) | 0.30 | 0.35 | 0.70 | +0.05 |
| **Prototype + TTA** | **0.32** | **0.37** | **0.73** | **+0.07** |

### Tabel Ablasi (Ablation Results)

| Config | LR | Steps | Var | Ent | BLEU-4 | Δ |
|--------|-----|-------|-----|-----|--------|---|
| Default | 0.001 | 10 | 0.1 | 0.01 | 0.320 | +0.020 |
| Best LR | 0.005 | 10 | 0.1 | 0.01 | 0.325 | +0.025 |
| Best Steps | 0.001 | 15 | 0.1 | 0.01 | 0.323 | +0.023 |
| No Reg | 0.001 | 10 | 0.0 | 0.0 | 0.315 | +0.015 |

### Grafik (Plots to Generate)

1. **Convergence Plot** - Loss vs adaptation steps
2. **Delta Distribution** - Histogram of per-sample Δ
3. **Performance vs Domain Distance** - Scatter plot
4. **Ablation Comparison** - Bar chart of different configs
5. **Per-Modality Performance** - Bar chart (XR, CT, MRI, etc.)

---

## 🚀 Quick Commands Reference

```bash
# Basic TTA experiment
python -m src.main_tta_experiment --config config.yaml --skip-embedding

# Ablation studies
python -m src.experiment_runner --study learning_rate --skip-embedding
python -m src.experiment_runner --study num_steps --skip-embedding
python -m src.experiment_runner --study regularization --skip-embedding

# Cross-domain
python -m src.cross_domain_eval --source roco --target mimic --use-tta

# Compare experiments
python scripts/compare_experiments.py results/metrics_*.csv
```

---

## ⏱️ Estimated Time

| Task | Time | Priority |
|------|------|----------|
| Phase 1: Baseline (500 samples) | 30 min | **HIGH** |
| Phase 2: Ablation (3 studies × 5 configs) | 4 hours | **HIGH** |
| Phase 3: Cross-domain (2 domains) | 1 hour | MEDIUM |
| Phase 4: Full eval (5000 samples) | 3 hours | MEDIUM |
| Analysis & plots | 2 hours | **HIGH** |
| **TOTAL** | ~10 hours | |

---

## 📦 Deliverables

### For Thesis

- [ ] Main results table (4 methods comparison)
- [ ] Ablation study results (best hyperparameters)
- [ ] Cross-domain evaluation results
- [ ] Convergence plots (TTA adaptation)
- [ ] Delta distribution plots
- [ ] Per-sample analysis (top improved/degraded)

### For Publication

- [ ] All of the above
- [ ] Statistical significance tests (paired t-test)
- [ ] Comparison with SOTA baselines
- [ ] Error analysis
- [ ] Qualitative examples (good/bad cases)
- [ ] (Optional) User study results

---

## 🐛 Known Issues & TODOs

### Critical
- [ ] Verify TTA integration tracks metrics correctly
- [ ] Test on small dataset first (10 samples) for debugging

### High Priority
- [ ] Add clinical evaluation metrics (CheXpert, RadGraph)
- [ ] Verify MIMIC-CXR dataset loader works
- [ ] Add statistical significance testing

### Medium Priority
- [ ] Implement LoRA-based TTA for comparison
- [ ] Add modality-specific train/test split
- [ ] Optimize TTA runtime (batch processing?)

### Low Priority
- [ ] User study framework
- [ ] Style-shifted synthetic data generation
- [ ] Interactive visualization dashboard

---

## 🎓 Next Steps

1. **Run Phase 1** (Basic TTA experiment) to validate everything works
2. **Analyze results** and verify Δ improvement exists
3. **Run Phase 2** (Ablation studies) to find best hyperparameters
4. **Run Phase 4** (Large-scale) with best config for final results
5. **Generate plots and tables** for thesis/paper

**Start with:**
```bash
python -m src.main_tta_experiment --config config.yaml --skip-embedding
```

Good luck! 🚀
