# FAST Training Runs — Notebook Documentation & Analysis

## Overview

`training-runs.ipynb` trains the **FAST** (Functional-Area-based Spatio-Temporal) model on imagined-speech EEG data from the BCIC2020 Track 3 dataset. It systematically compares **9 data conditions** (7 preprocessing variants + 2 augmentation variants) using a unified evaluation protocol. Each condition is trained **per-subject** (15 subjects) with **5-fold cross-validation** and evaluated on a held-out official test set.

---

## Notebook Structure

| Section | Cell Purpose |
|---------|-------------|
| **§0** | Title + experiment table |
| **§1** | Imports (`lightning`, `torch`, `sklearn`, project modules) |
| **§2** | Global configuration (hyperparameters, data paths, model config) |
| **§3** | Data loading utilities (`load_h5_subject`, `load_test_h5`, `load_combined_subject`) + `AugmentedDataset` and `AugmentationScheduler` class definitions |
| **§4** | Experiment config dictionary (7 runs) |
| **§5** | `train_experiment()` — the central training function |
| **§6–§12** | Execution cells for Runs 1–7 |
| **§13** | Comparison summary + visualization (grouped bar charts) |
| **§14** | Run 8: Data augmentation + early stopping |
| **§15** | Run 9: Data augmentation, no early stopping |

---

## Experimental Conditions

| Run | Data Source | Augmentation | Early Stopping | Description |
|-----|-----------|:------------:|:--------------:|-------------|
| 1 | Original (A) | ✗ | patience=20 | Raw BCIC2020Track3 data |
| 2 | Condition B | ✗ | patience=20 | ICA-cleaned, ALL delta-dominant ICs removed |
| 3 | Condition C | ✗ | patience=20 | ICA-cleaned, only artifact delta ICs removed |
| 4 | A + B | ✗ | patience=20 | Original + no-delta combined (2× data) |
| 5 | A + C | ✗ | patience=20 | Original + no-artifacts combined (2× data) |
| 6 | A + B + C | ✗ | patience=20 | All three conditions combined (3× data) |
| 7 | B + C | ✗ | patience=20 | Both ICA conditions combined (2× data) |
| 8 | Original (A) | ✓ | patience=20 | Augmentation with early stopping |
| 9 | Original (A) | ✓ | patience=0 (disabled) | Augmentation, full 200 epochs |

---

## Training Protocol

### Model
- **Architecture**: FAST — Zone-based spatio-temporal transformer
  - Conv4Layers head per functional zone → Transformer encoder → classification
  - `dim_cnn=32`, `dim_token=32`, `num_layers=4`, `num_heads=8`, `dropout=0.1`
  - ~191K trainable parameters
- **Input shape**: `(batch, 64 channels, 800 timepoints)` = 3.2s at 250Hz

### Optimization
- **Optimizer**: AdamW, `lr=0.0005`
- **LR Schedule**: Cosine decay from 1.0× to 0.1× over 200 epochs with 10-epoch linear warmup
  - Applied per-step via `LambdaLR`
- **Loss**: CrossEntropyLoss (5-class classification)
- **Precision**: `bf16-mixed` (bfloat16 automatic mixed precision)
- **Batch size**: 64
- **Max epochs**: 200

### Evaluation
- **Validation**: Shuffled 5-fold KFold (seed=42), per subject
  - Best fold selected by highest `val_acc`
  - Model checkpoint saved per fold
- **Test**: Official held-out test set (50 trials per subject)
  - Evaluated using the best-fold model for each subject
  - Metrics: Accuracy, Macro F1
- **Data**: 350 trials/subject for single-source runs; multiplied for combined runs

### Early Stopping
- **Monitor**: `val_acc` (mode=max)
- **Patience**: 20 epochs (configurable; disabled when `patience=0`)
- **Min delta**: 0.0

---

## Data Augmentation (Runs 8–9)

The `AugmentedDataset` class applies stochastic augmentations to the **training set only** (validation always uses `BasicDataset`):

| Augmentation | Parameters | Application Probability |
|-------------|-----------|:----------------------:|
| Temporal Jitter | ±15 samples (60ms at 250Hz), zero-padded | 50% |
| Gaussian Noise | SNR 20–40 dB (randomized per sample) | 50% |
| Channel Dropout | 10% of 64 channels zeroed out | 30% |

### Precision Cool-Down Schedule
All augmentation probabilities are linearly decayed to 0% between epochs 170–200:
```
prob_mult = 1.0                    (epoch < 170)
prob_mult = 1.0 - (epoch-170)/30   (170 ≤ epoch < 200)
prob_mult = 0.0                    (epoch ≥ 200)
```
The `AugmentationScheduler` callback updates the dataset's `current_epoch` attribute at the start of each training epoch to drive this schedule.

---

## Discrepancies & Potential Issues

### 1. ⚠️ LR Scheduler Off-By-One Risk (Global Step Indexing)
```python
# trainer.py, line 52
lambda epoch: self.cosine_lr_list[self.global_step - 1]
```
`self.global_step` starts at `0` on the very first call, so `self.global_step - 1 = -1`. In Python this silently reads the **last element** of the array (the final/minimum LR value) for the first step, meaning the first training step uses the wrong learning rate. This is a **pre-existing issue** in the FAST codebase, not introduced by this notebook.

### 2. ⚠️ Cosine Schedule Length Mismatch with Early Stopping
```python
# train_experiment, notebook cell §5
model = EEG_Encoder_Module(model_config, MAX_EPOCHS, len(train_loader))
```
The cosine LR schedule is pre-computed for exactly `MAX_EPOCHS × niter_per_ep` steps. With early stopping enabled (`patience=20`), training stops early, so the schedule is **never fully consumed** — this is benign behavior (unused tail of the array). However, if early stopping were to trigger at the very beginning of training, the schedule would still be correctly indexed since it starts from step 0.

### 3. ⚠️ `val_f1` Is Logged But Not Saved
```python
# trainer.py, validation_step
self.log('val_acc', acc, ...)   # ← logged
# val_f1 is computed but never logged
```
Looking at `validation_step`, `val_f1` is computed via `self.val_f1(logits, y)` but **only `val_acc` is logged**. The F1 metric object accumulates state each step but its value is never exported to callbacks or the history. This is silent wasted computation, not a bug, but it means the `HistoryCallback` never captures validation F1 per epoch.

### 4. ⚠️ Comparison Summary Won't Include Runs 8–9
The comparison summary cell (§13) iterates over `EXPERIMENTS.items()` to collect results. However, Runs 8 and 9 are defined in cells **after** the comparison cell. If cells are run top-to-bottom, `EXPERIMENTS` won't yet contain `run8_augment` or `run9_augment_no_es` when the comparison executes. The comparison cell needs to be re-run after Runs 8–9 complete, or the runs should be defined before it.

### 5. ⚠️ Comparison Summary `Trials/Subject` Formula Incorrect for Augmented Runs
```python
'Trials/Subject': 350 * len(exp['h5_paths']),
```
For Runs 8–9, `h5_paths = [H5_ORIGINAL]` so `len = 1`, yielding `350`. This is numerically correct (augmentation doesn't increase trial count), but the column name is misleading — augmented runs see effectively more unique training samples due to stochastic transforms.

### 6. ⚠️ `AugmentedDataset` Placed in Data Loading Cell (§3)
The `AugmentedDataset` and `AugmentationScheduler` classes are appended to the bottom of the data loading utilities cell (§3). This is a code organization issue — they are functionally unrelated to data loading utilities (`load_h5_subject`, etc.) and would be better placed in their own cell before the training function.

### 7. ℹ️ Checkpoint Overwrite Between Folds
```python
filename=f'fold-{fold_idx}-best'
```
All folds for a subject are saved in the same directory (`sub-{SID}/`). With `save_top_k=1`, only the best checkpoint per fold is kept. This is correct behavior, but if the notebook is re-run without clearing results, old checkpoints may persist alongside new ones in the same directory.

### 8. ℹ️ No Seed Reset Between Experiments
`seed_all(SEED)` is called once during configuration (§2). Subsequent experiments inherit the evolving random state from previous experiments. This means:
- Runs 1–9 are **not independently reproducible** — each depends on the random state left by the prior run.
- Re-running a single run cell in isolation will produce different results than running all cells sequentially.

### 9. ℹ️ `BasicDataset` Returns Tensors, `AugmentedDataset` Converts Back to NumPy
`BasicDataset.__init__` converts data to `torch.Tensor` via `torch.from_numpy()`. Then `AugmentedDataset.__getitem__` calls `x.numpy()` to apply NumPy-based augmentations, then converts back with `torch.from_numpy(x).float()`. This double conversion is inefficient but functionally correct.

---

## Output Artifacts

Each run produces the following per subject in `results/condition_experiments/<run_name>/`:

```
sub-{SID}/
├── fold-{N}-best.ckpt       # Best model checkpoint per fold
├── fold-{N}_history.csv      # Epoch-level train/val loss & accuracy
├── fold-{N}_curves.png       # Learning curve plots
├── fold_metrics.csv           # Best val_acc per fold
├── best_subject.pth           # Best fold model weights (state_dict)
└── test_predictions.csv       # Test set predictions vs ground truth
```

Global outputs:
```
<run_name>/
├── summary_per_subject.csv        # Per-subject Best_Val_Acc, Test_Acc, Test_F1
├── global_test_predictions.csv    # All subjects' test predictions concatenated
└── global_subject_accuracy.png    # Bar chart of test accuracy per subject
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `lightning` (PyTorch Lightning) | Training loop, callbacks, GPU management |
| `torch` / `torchmetrics` | Model, loss, metrics |
| `transformers.PretrainedConfig` | Model configuration container |
| `h5py` | HDF5 data loading |
| `scikit-learn` | KFold, accuracy_score, f1_score |
| `numpy`, `pandas`, `matplotlib` | Data manipulation, results, plots |
| `fast.*` (project) | FAST model, BasicDataset, EEG_Encoder_Module, HistoryCallback, utilities |
