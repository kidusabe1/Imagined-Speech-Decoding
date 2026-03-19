# FAST Model Analysis Report: Delta Dominance, Condition Ablation, and Interpretability

## Scope

This report synthesises findings from three analysis notebooks and their saved artifacts:

| Notebook | Purpose |
|----------|---------|
| `notebooks/training-runs.ipynb` | 11-condition ablation study (training + evaluation) |
| `notebooks/shap-analysis.ipynb` | Per-experiment SHAP interpretability |
| `notebooks/ica-analysis-fast.ipynb` | ICA decomposition of raw EEG to characterise delta contamination |

**Dataset**: BCI Competition 2020 Track 3 — 15 subjects, 5 imagined-speech classes, 64 channels at 250 Hz.
**Model**: FAST (Functional-Area-based Spatio-Temporal transformer), ~191K parameters.

---

## 1. Training Condition Comparison

### 1.1 Summary Table

| Run | Condition | Trials/Sub | Mean Val Acc | Mean Test Acc | Mean Test F1 |
|-----|-----------|:----------:|:------------:|:-------------:|:------------:|
| 1 | Original (A) | 350 | 0.736 ± 0.050 | 0.635 ± 0.082 | 0.631 ± 0.083 |
| 2 | Cond B (No Delta) | 350 | 0.415 ± 0.061 | 0.249 ± 0.063 | 0.179 ± 0.072 |
| 3 | Cond C (No Artifacts) | 350 | 0.427 ± 0.060 | 0.233 ± 0.043 | 0.176 ± 0.056 |
| 4 | A + B | 700 | 0.607 ± 0.037 | 0.644 ± 0.060 | 0.642 ± 0.060 |
| 5 | A + C | 700 | 0.618 ± 0.042 | 0.651 ± 0.080 | 0.645 ± 0.081 |
| 6 | A + B + C | 1050 | 0.738 ± 0.073 | 0.653 ± 0.075 | 0.650 ± 0.076 |
| 7 | B + C | 700 | 0.717 ± 0.123 | 0.272 ± 0.055 | 0.208 ± 0.072 |
| 8 | Augmented + ES | 350 | 0.725 ± 0.068 | 0.620 ± 0.078 | 0.617 ± 0.077 |
| 9 | Augmented (No ES) | 350 | 0.761 ± 0.052 | 0.675 ± 0.081 | 0.674 ± 0.084 |
| 10 | Full-Data Augmented (No Val Split) | 350 | 0.733 ± 0.084 | 0.733 ± 0.084 | 0.732 ± 0.086 |
| 11 | High-pass 4 Hz | 350 | 0.436 ± 0.069 | 0.205 ± 0.040 | 0.129 ± 0.042 |

> **Key evidence**: [comparison_summary.csv](../results/condition_experiments/comparison_summary.csv)

#### Figure: Mean Test Accuracy Across Conditions

![Mean Test Accuracy](../results/condition_experiments/comparison_mean_accuracy.png)

#### Figure: Per-Subject Accuracy Across Conditions

![Per-Subject Accuracy](../results/condition_experiments/comparison_per_subject.png)

### 1.2 Per-Subject Breakdown

The following tables show individual subject performance for key runs, exposing the high inter-subject variance that aggregate means obscure.

#### Run 1: Original Data (A) — Baseline

| Subject | Best Val Acc | Test Acc | Test F1 |
|:-------:|:-----------:|:--------:|:-------:|
| 01 | 0.771 | 0.54 | 0.539 |
| 02 | 0.686 | 0.58 | 0.582 |
| 03 | 0.743 | 0.70 | 0.690 |
| 04 | 0.829 | 0.74 | 0.733 |
| 05 | 0.771 | 0.72 | 0.719 |
| 06 | 0.686 | 0.64 | 0.639 |
| 07 | 0.757 | 0.58 | 0.581 |
| 08 | 0.743 | 0.66 | 0.660 |
| 09 | 0.700 | 0.66 | 0.654 |
| 10 | 0.714 | 0.62 | 0.622 |
| 11 | 0.757 | 0.72 | 0.718 |
| 12 | 0.786 | 0.62 | 0.613 |
| 13 | 0.700 | 0.48 | 0.487 |
| 14 | 0.629 | 0.52 | 0.493 |
| 15 | 0.771 | 0.74 | 0.738 |

> Per-subject accuracy chart: [run1 global_subject_accuracy.png](../results/condition_experiments/run1_original/global_subject_accuracy.png)

#### Run 10: Full-Data Augmented Training (No Val Split) — Best Overall

| Subject | Best Val Acc | Test Acc | Test F1 |
|:-------:|:-----------:|:--------:|:-------:|
| 01 | 0.660 | 0.66 | 0.645 |
| 02 | 0.720 | 0.72 | 0.717 |
| 03 | 0.840 | 0.84 | 0.838 |
| 04 | 0.780 | 0.78 | 0.785 |
| 05 | 0.800 | 0.80 | 0.802 |
| 06 | 0.760 | 0.76 | 0.763 |
| 07 | 0.800 | 0.80 | 0.800 |
| 08 | 0.640 | 0.64 | 0.644 |
| 09 | 0.740 | 0.74 | 0.733 |
| 10 | 0.700 | 0.70 | 0.703 |
| 11 | 0.720 | 0.72 | 0.715 |
| 12 | 0.820 | 0.82 | 0.822 |
| 13 | 0.580 | 0.58 | 0.572 |
| 14 | 0.600 | 0.60 | 0.602 |
| 15 | 0.840 | 0.84 | 0.839 |

Note: Val Acc = Test Acc for Run 10 because the official test set is passed as the validation monitor during training (there is no held-out validation fold — all 350 training trials are used). The recorded "Best_Val_Acc" is the final-epoch test accuracy. This makes Run 10 not directly comparable to Runs 1–9/11, which hold out a validation fold and evaluate on the official test set separately.

> Per-subject accuracy chart: [run10 global_subject_accuracy.png](../results/condition_experiments/run10_kaggle_replication/global_subject_accuracy.png)

#### Run 9: Augmented (No ES) — Best with Held-Out Test

| Subject | Best Val Acc | Test Acc | Test F1 |
|:-------:|:-----------:|:--------:|:-------:|
| 01 | 0.771 | 0.62 | 0.620 |
| 02 | 0.686 | 0.58 | 0.581 |
| 03 | 0.786 | 0.70 | 0.697 |
| 04 | 0.857 | 0.76 | 0.765 |
| 05 | 0.786 | 0.70 | 0.704 |
| 06 | 0.700 | 0.62 | 0.606 |
| 07 | 0.757 | 0.72 | 0.723 |
| 08 | 0.757 | 0.66 | 0.653 |
| 09 | 0.714 | 0.68 | 0.682 |
| 10 | 0.771 | 0.58 | 0.575 |
| 11 | 0.786 | 0.84 | 0.842 |
| 12 | 0.843 | 0.68 | 0.676 |
| 13 | 0.700 | 0.60 | 0.603 |
| 14 | 0.700 | 0.58 | 0.573 |
| 15 | 0.800 | 0.80 | 0.807 |

> Per-subject accuracy chart: [run9 global_subject_accuracy.png](../results/condition_experiments/run9_augment_no_es/global_subject_accuracy.png)

#### Run 7: B + C — Overfitting Case Study

| Subject | Best Val Acc | Test Acc | Test F1 |
|:-------:|:-----------:|:--------:|:-------:|
| 01 | 0.764 | 0.32 | 0.185 |
| 02 | 0.671 | 0.30 | 0.231 |
| 03 | **0.893** | **0.18** | 0.103 |
| 04 | 0.736 | 0.26 | 0.202 |
| 05 | **0.829** | **0.18** | 0.109 |
| 06 | 0.679 | 0.24 | 0.195 |
| 07 | 0.750 | 0.30 | 0.247 |
| 08 | 0.579 | 0.34 | 0.308 |
| 09 | **0.886** | 0.30 | 0.212 |
| 10 | 0.714 | 0.30 | 0.262 |
| 11 | 0.593 | 0.32 | 0.280 |
| 12 | **0.871** | 0.24 | 0.155 |
| 13 | 0.457 | 0.32 | 0.262 |
| 14 | 0.643 | 0.18 | 0.077 |
| 15 | 0.686 | 0.30 | 0.297 |

Subjects 3, 5, 9, and 12 show the most extreme val/test divergence (val > 0.83, test < 0.30). These subjects have the highest validation accuracy but the lowest test accuracy — a textbook distribution-shift overfitting pattern.

> Per-subject accuracy chart: [run7 global_subject_accuracy.png](../results/condition_experiments/run7_B_C/global_subject_accuracy.png)

#### Run 11: High-pass 4 Hz — Worst Performer

| Subject | Best Val Acc | Test Acc | Test F1 |
|:-------:|:-----------:|:--------:|:-------:|
| 01 | 0.429 | 0.20 | 0.094 |
| 02 | 0.286 | 0.20 | 0.096 |
| 03 | 0.514 | 0.12 | 0.092 |
| 04 | 0.457 | 0.22 | 0.124 |
| 05 | 0.529 | 0.28 | 0.223 |
| 06 | 0.429 | 0.20 | 0.118 |
| 07 | 0.386 | 0.26 | 0.175 |
| 08 | 0.514 | 0.20 | 0.067 |
| 09 | 0.414 | 0.22 | 0.128 |
| 10 | 0.486 | 0.18 | 0.086 |
| 11 | 0.443 | 0.20 | 0.139 |
| 12 | 0.500 | 0.24 | 0.132 |
| 13 | 0.329 | 0.14 | 0.118 |
| 14 | 0.400 | 0.22 | 0.183 |
| 15 | 0.429 | 0.20 | 0.168 |

Every subject is at or near 5-class chance (20%). Even validation accuracy barely exceeds chance, confirming that the model cannot learn useful representations from high-pass filtered data.

> Per-subject accuracy chart: [run11 global_subject_accuracy.png](../results/condition_experiments/run11_highpass4/global_subject_accuracy.png)

### 1.3 Baseline Comparison (GitHub FAST Replication)

For reference, the original FAST GitHub codebase (KFold, no held-out test, full-batch training) produces these per-subject results:

| Subject | Overall Acc | Overall F1 | Mean Fold Acc | Std Fold Acc |
|:-------:|:-----------:|:----------:|:-------------:|:------------:|
| 01 | 0.611 | 0.611 | 0.611 | 0.052 |
| 02 | 0.546 | 0.545 | 0.546 | 0.043 |
| 03 | 0.623 | 0.623 | 0.623 | 0.038 |
| 04 | 0.743 | 0.743 | 0.743 | 0.069 |
| 05 | 0.686 | 0.686 | 0.686 | 0.066 |
| 06 | 0.543 | 0.543 | 0.543 | 0.066 |
| 07 | 0.660 | 0.659 | 0.660 | 0.045 |
| 08 | 0.643 | 0.643 | 0.643 | 0.055 |
| 09 | 0.649 | 0.648 | 0.649 | 0.038 |
| 10 | 0.634 | 0.633 | 0.634 | 0.029 |
| 11 | 0.403 | 0.405 | 0.403 | 0.101 |
| 12 | 0.386 | 0.381 | 0.386 | 0.117 |
| 13 | 0.529 | 0.528 | 0.529 | 0.039 |
| 14 | 0.554 | 0.554 | 0.554 | 0.059 |
| 15 | 0.620 | 0.620 | 0.620 | 0.056 |

**Mean overall accuracy: 0.589.** Note that these are KFold held-out fold accuracies, not held-out test set results, so they are methodologically comparable to Run 10 (0.733) rather than Runs 1–9. The gap (0.589 vs 0.733) comes from our hyperparameter tuning, mini-batch training, and shuffle=True in KFold. See [split_analysis.md](split_analysis.md) for detailed methodological differences.

> **Key evidence**: [baseline summary_per_subject.csv](../results/baseline/FAST/summary_per_subject.csv)

### 1.4 Interpretation

**Condition-only runs collapse.** Runs 2 (No Delta), 3 (No Artifacts), 7 (B+C), and 11 (4 Hz high-pass) all fall to 20–27% test accuracy — near chance level for 5-class classification (20%). This demonstrates that the original signal distribution carries essential discriminative structure, and that ICA-cleaned data alone is insufficient for the model to learn generalisable features.

**Mixed-data runs recover.** Runs 4 (A+B), 5 (A+C), and 6 (A+B+C) all achieve ~0.64–0.65 test accuracy, comparable to the original-data baseline (Run 1: 0.635). Mixing original data with cleaned conditions acts as a form of regularisation without degrading performance.

**Run 7 exposes a generalisation gap.** Validation accuracy is high (0.717) with extreme variance (± 0.123), but test accuracy collapses to 0.272. This is the clearest overfitting signal in the study: models trained on only ICA-cleaned conditions learn patterns that do not transfer to the original-distribution test set.

**Augmentation and full-data training lead.** Run 10 (full-data augmented training, no validation split) achieves 0.733 test accuracy. Run 9 (augmentation without early stopping) reaches 0.675, the best result among runs with properly held-out test evaluations. The gap is partly methodological — Run 10 trains on all available training data rather than a 4/5 fold subset, and the test set doubles as the validation monitor during training.

**Strict high-pass filtering is destructive.** Run 11 (4 Hz high-pass) produces the worst result (0.205 test accuracy, 0.129 F1), confirming that sub-4 Hz content — whether neural or artifactual — is currently load-bearing for the model.

> **Key evidence**: [comparison_mean_accuracy.png](../results/condition_experiments/comparison_mean_accuracy.png), [comparison_per_subject.png](../results/condition_experiments/comparison_per_subject.png)

---

## 2. SHAP Cross-Experiment Analysis

### 2.1 Attribution Magnitude Summary

| Experiment | Mean |SHAP| | Std |SHAP| | Top Zone |
|------------|:----------:|:----------:|:--------:|
| Run 1: Original (A) | 5.38e-4 | 1.62e-4 | Pre-frontal |
| Run 2: Cond B (No Delta) | 6.63e-4 | 5.71e-4 | Central |
| Run 3: Cond C (No Artifacts) | 8.45e-4 | 6.10e-4 | Occipital |
| Run 4: A + B | 1.20e-3 | 2.95e-4 | Occipital |
| Run 5: A + C | 1.33e-3 | 3.54e-4 | Occipital |
| Run 6: A + B + C | 2.44e-3 | 5.39e-4 | Occipital |
| Run 7: B + C | 2.81e-3 | 6.92e-4 | Occipital |
| Run 8: Augmented + ES | 4.02e-4 | 5.70e-5 | Pre-frontal |
| Run 9: Augmented (No ES) | 4.30e-4 | 6.62e-5 | Pre-frontal |
| Run 10: Full-Data Augmented (No Val Split) | 4.06e-4 | 6.21e-5 | Pre-frontal |
| Run 11: High-pass 4 Hz | 2.13e-4 | 2.63e-4 | Pre-central |

> **Key evidence**: [shap_summary.csv](../results/shap_analysis/shap_summary.csv)

### 2.2 Interpretation

**Magnitude scales with data volume and condition mixing.** The highest SHAP magnitudes belong to Run 7 (2.81e-3) and Run 6 (2.44e-3) — the multi-condition runs. Models trained on more data conditions produce stronger per-feature attributions, suggesting broader feature utilisation (or amplified reliance on a specific feature axis shared across conditions).

**Top zone shifts with condition family.** There is a clear geographical bifurcation:
- **Original / augmented / full-data runs** → Pre-frontal dominance (Runs 1, 8, 9, 10)
- **Mixed ICA runs** → Occipital dominance (Runs 3, 4, 5, 6, 7)
- **Strict high-pass** → Pre-central shift (Run 11)

This indicates that preprocessing choices reshape *where* the model attends, not just *how well* it performs. Delta removal redirects attribution from frontal (likely ocular artifact) regions to posterior areas.

**Low-performing models have extreme SHAP variance.** Runs 2 and 3 show the highest relative variance (std/mean > 0.7), consistent with models that have not converged on stable feature representations.

#### Figure: Cross-Experiment SHAP Magnitude

![Cross-Experiment Magnitude](../results/shap_analysis/cross_exp_magnitude.png)

#### Figure: Zone Ranking Across Experiments

![Zone Ranking](../results/shap_analysis/cross_exp_zone_ranking.png)

#### Figure: Zone Importance Bars (All Experiments)

![Zone Bars](../results/shap_analysis/zone_bars_all_experiments.png)

#### Figure: Language Zone Analysis

![Language Zone Global](../results/shap_analysis/language_zone_global.png)

![Language Zone Comparison](../results/shap_analysis/language_zone_comparison.png)

#### Figure: Per-Subject SHAP Magnitude

![Per-Subject Magnitude](../results/shap_analysis/per_subject_magnitude.png)

> **Key evidence**: [cross_exp_magnitude.png](../results/shap_analysis/cross_exp_magnitude.png), [cross_exp_zone_ranking.png](../results/shap_analysis/cross_exp_zone_ranking.png), [zone_bars_all_experiments.png](../results/shap_analysis/zone_bars_all_experiments.png)

### 2.3 Methodological Note: STFT Pivot

The frequency-band analysis in `shap-analysis.ipynb` was corrected during development. The current (correct) implementation:

1. Bandpass-filters the **raw EEG** signal per frequency band.
2. Computes SHAP-weighted signal: `|filtered_EEG × SHAP_values|`.
3. Applies STFT to the weighted signal for time-resolved band importance.

The prior (incorrect) approach applied STFT directly to SHAP values, characterising attribution dynamics in SHAP space rather than estimating the model-relevant EEG band contribution. Despite the methodological correction, the high-level conclusion is unchanged: delta/low-frequency dominance persists across all analysis axes.

> **Key evidence**: `shap-analysis.ipynb` cells 340–390 (function `plot_freq_band_heatmap`), cells 724–741 (analysis loop passing `avg_eeg`)

---

## 3. ICA Decomposition Analysis

### 3.1 Raw Signal Characterisation

| Band | Mean Power (V²/Hz) | % of Total |
|------|:------------------:|:----------:|
| Delta (0.5–4 Hz) | 353.21 ± 495.56 | **89.7%** |
| Theta (4–8 Hz) | 23.85 ± 30.49 | 6.1% |
| Alpha (8–13 Hz) | 8.42 ± 8.71 | 2.1% |
| Beta (13–30 Hz) | 5.97 ± 4.09 | 1.5% |
| Gamma (30–100 Hz) | 2.09 ± 1.64 | 0.5% |

The raw PSD is overwhelmingly delta-dominated. Nearly 90% of total spectral power sits below 4 Hz before any preprocessing.

### 3.2 Independent Component Statistics

- **Delta-dominant ICs per subject**: 16.9 ± 2.2 out of 20 (~84%)
- **Total delta-dominant ICs across all subjects**: 253
- **IC class breakdown** (of 253 delta-dominant ICs):

| IC Class | Count | % |
|----------|:-----:|:-:|
| Frontal (likely ocular) | 90 | 35.6% |
| Diffuse (likely drift/ref) | 46 | 18.2% |
| Focal-Temporal | 38 | 15.0% |
| Focal-Central | 32 | 12.6% |
| Focal-Occipital | 27 | 10.7% |
| Focal-Parietal | 20 | 7.9% |

Over a third of all delta-dominant ICs are frontal with likely ocular origin, confirming substantial eye-movement artifact contamination in the delta band.

### 3.3 Cross-Subject Delta IC Similarity

**Mean inter-subject similarity**: 0.542 ± 0.048 (range: 0.408–0.638)

This moderate similarity indicates a mix of **shared artifact templates** (e.g., common eye-blink spatial patterns) and **subject-specific sources**. The model can exploit these shared patterns for subject identification rather than class discrimination.

> **Reporting bug**: The final ICA dashboard text prints a contradictory cross-subject similarity value (-0.001). The dedicated similarity computation outputs 0.542 ± 0.048 — this is the correct value. The dashboard line is a variable-reuse bug and should be disregarded.

### 3.4 Delta Power Reduction After IC Removal

Removing delta-dominant ICs reduces zone-level delta power by **82%–99%** across all FAST brain zones. This confirms that the ICA decomposition successfully isolates and removes the dominant low-frequency content.

### 3.5 Class Discrimination Test

| Test | Statistic | p-value |
|------|:---------:|:-------:|
| One-Way ANOVA | F = 2.270 | 0.059 |
| Kruskal-Wallis | H = 3.614 | 0.461 |

Delta power is **not significantly class-discriminative** (p ≥ 0.05). Subject-to-subject variability in delta power is **56× larger** than class-to-class variability. The model's reliance on delta is almost certainly driven by subject-level baseline differences (and artifacts), not by task-related neural modulation.

> **Key evidence**: ICA dashboard figure (inline in notebook), zone-level delta power reduction bar charts, class-conditional delta power violin plots (all inline in `ica-analysis-fast.ipynb`)

---

## 4. Convergent Evidence: The Delta Problem

All three analyses point to the same core finding. The table below maps each line of evidence to its source:

| Finding | Source | Key Evidence |
|---------|--------|-------------|
| Model performance collapses when delta content is removed | Training ablation | Runs 2, 3, 7, 11 in [comparison_summary.csv](../results/condition_experiments/comparison_summary.csv) |
| Strict 4 Hz high-pass is most destructive (0.205 test acc) | Training ablation | Run 11 in [comparison_mean_accuracy.png](../results/condition_experiments/comparison_mean_accuracy.png) |
| Mixing original data with cleaned data recovers performance | Training ablation | Runs 4–6 in [comparison_per_subject.png](../results/condition_experiments/comparison_per_subject.png) |
| Run 7 (B+C) shows extreme val/test gap (0.717 → 0.272) | Training ablation | [comparison_summary.csv](../results/condition_experiments/comparison_summary.csv) |
| Top SHAP zone shifts from Pre-frontal to Occipital with delta removal | SHAP analysis | [cross_exp_zone_ranking.png](../results/shap_analysis/cross_exp_zone_ranking.png) |
| SHAP magnitude highest in multi-condition runs | SHAP analysis | [cross_exp_magnitude.png](../results/shap_analysis/cross_exp_magnitude.png) |
| Raw PSD is 89.7% delta | ICA analysis | PSD computation in `ica-analysis-fast.ipynb` |
| 84% of ICA components are delta-dominant | ICA analysis | IC classification in `ica-analysis-fast.ipynb` |
| 35.6% of delta ICs are frontal/ocular artifacts | ICA analysis | IC class breakdown in `ica-analysis-fast.ipynb` |
| Delta power is not class-discriminative (ANOVA p = 0.059) | ICA analysis | Statistical test in `ica-analysis-fast.ipynb` |
| Subject variability in delta is 56× class variability | ICA analysis | Variance ratio in `ica-analysis-fast.ipynb` |
| Cross-subject delta IC similarity is moderate (0.542) | ICA analysis | Similarity matrix in `ica-analysis-fast.ipynb` |
| Delta removal reduces zone-level power by 82–99% | ICA analysis | Zone reduction analysis in `ica-analysis-fast.ipynb` |
| Corrected STFT method still shows delta dominance | SHAP analysis | `plot_freq_band_heatmap` in `shap-analysis.ipynb` |

---

## 5. Conclusions

1. **The FAST model is delta-dependent.** Removing delta content via ICA or high-pass filtering destroys classification performance, dropping it to near chance. This dependency is on non-discriminative signal — delta power does not differ between imagined speech classes.

2. **Delta content is predominantly artifactual.** 35.6% of delta-dominant ICs are frontal/ocular, 18.2% are diffuse drift. The remaining focal components may include genuine slow cortical potentials, but the lack of class discrimination (p = 0.059) suggests they are not task-relevant.

3. **The model exploits subject identity via delta.** With 56× more between-subject than between-class variance in delta power, and moderate cross-subject IC similarity (0.542), the model likely learns subject-specific delta fingerprints. This explains why:
   - Within-subject cross-validation scores are high (Runs 6, 7 val > 0.71)
   - Transfer to the held-out test set fails when delta is removed (Run 7: 0.717 val → 0.272 test)

4. **Preprocessing reshapes attribution geography.** SHAP top-zone shifts from Pre-frontal (original data) to Occipital (ICA-cleaned data), demonstrating that delta removal forces the model to attend to different spatial regions. Whether these posterior features carry genuine class information remains to be validated.

5. **Data augmentation is the best current strategy.** Run 9 (stochastic augmentation, no early stopping) achieves 0.675 test accuracy on original data — the best result with a proper held-out test split. Augmentation provides regularisation without requiring delta removal.

---

## 6. Artifact Index

All figures and tables referenced in this report are stored under `results/`:

### Cross-Experiment Comparisons
- [comparison_summary.csv](../results/condition_experiments/comparison_summary.csv) — Accuracy and F1 for all 11 runs
- [comparison_mean_accuracy.png](../results/condition_experiments/comparison_mean_accuracy.png) — Bar chart of mean test accuracy
- [comparison_per_subject.png](../results/condition_experiments/comparison_per_subject.png) — Per-subject accuracy across runs

### SHAP Analysis
- [shap_summary.csv](../results/shap_analysis/shap_summary.csv) — Mean |SHAP|, top zone per experiment
- [cross_exp_magnitude.png](../results/shap_analysis/cross_exp_magnitude.png) — SHAP magnitude comparison
- [cross_exp_zone_ranking.png](../results/shap_analysis/cross_exp_zone_ranking.png) — Zone ranking across experiments
- [zone_bars_all_experiments.png](../results/shap_analysis/zone_bars_all_experiments.png) — Zone importance bars
- [language_zone_global.png](../results/shap_analysis/language_zone_global.png) — Global language zone analysis
- [language_zone_comparison.png](../results/shap_analysis/language_zone_comparison.png) — Language zone comparison
- [per_subject_magnitude.png](../results/shap_analysis/per_subject_magnitude.png) — Per-subject SHAP magnitude

### Per-Run Details

| Run | Per-Subject CSV | Accuracy Chart |
|-----|----------------|----------------|
| 1 | [summary](../results/condition_experiments/run1_original/summary_per_subject.csv) | [chart](../results/condition_experiments/run1_original/global_subject_accuracy.png) |
| 2 | [summary](../results/condition_experiments/run2_condB/summary_per_subject.csv) | [chart](../results/condition_experiments/run2_condB/global_subject_accuracy.png) |
| 3 | [summary](../results/condition_experiments/run3_condC/summary_per_subject.csv) | [chart](../results/condition_experiments/run3_condC/global_subject_accuracy.png) |
| 4 | [summary](../results/condition_experiments/run4_A_B/summary_per_subject.csv) | [chart](../results/condition_experiments/run4_A_B/global_subject_accuracy.png) |
| 5 | [summary](../results/condition_experiments/run5_A_C/summary_per_subject.csv) | [chart](../results/condition_experiments/run5_A_C/global_subject_accuracy.png) |
| 6 | [summary](../results/condition_experiments/run6_A_B_C/summary_per_subject.csv) | [chart](../results/condition_experiments/run6_A_B_C/global_subject_accuracy.png) |
| 7 | [summary](../results/condition_experiments/run7_B_C/summary_per_subject.csv) | [chart](../results/condition_experiments/run7_B_C/global_subject_accuracy.png) |
| 8 | [summary](../results/condition_experiments/run8_augment/summary_per_subject.csv) | [chart](../results/condition_experiments/run8_augment/global_subject_accuracy.png) |
| 9 | [summary](../results/condition_experiments/run9_augment_no_es/summary_per_subject.csv) | [chart](../results/condition_experiments/run9_augment_no_es/global_subject_accuracy.png) |
| 10 | [summary](../results/condition_experiments/run10_kaggle_replication/summary_per_subject.csv) | [chart](../results/condition_experiments/run10_kaggle_replication/global_subject_accuracy.png) |
| 11 | [summary](../results/condition_experiments/run11_highpass4/summary_per_subject.csv) | [chart](../results/condition_experiments/run11_highpass4/global_subject_accuracy.png) |

Learning curves per subject/fold: `results/condition_experiments/run*/sub-*/fold-*_curves.png`

### ICA Analysis
- All ICA figures are inline in `notebooks/ica-analysis-fast.ipynb` (PSD, IC classification, zone power reduction, class discrimination, cross-subject similarity matrix, dashboard)
