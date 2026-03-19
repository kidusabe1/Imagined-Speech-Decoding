# FAST: Functional Areas Spatio-Temporal Transformer

Code for paper: **Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)**

This codebase reproduces results on the publicly available dataset [BCI Competition 2020 Track #3: Imagined Speech Classification](https://osf.io/pq7vb/).

## Project Structure

```
FAST/
├── src/fast/                        # Main package
│   ├── models/                      # FAST model architecture
│   ├── data/                        # Data loading and preprocessing
│   ├── train/                       # Training utilities
│   ├── analysis/                    # SHAP comparison helpers
│   └── utils.py                     # Helper functions
├── scripts/
│   ├── train_fast.py                # Main training script (FAST)
│   ├── train_fast_baseline.py       # Baseline variant
│   ├── train_tsception.py           # TSception comparison
│   ├── benchmark.py                 # Metrics aggregation
│   ├── preprocess.py                # Raw data → HDF5
│   ├── explain_fast.py              # SHAP explanations
│   ├── artifact_analysis.py         # ICA-based artifact analysis
│   └── global_shap_analysis.py      # Dataset-wide SHAP
├── configs/
│   └── default.yaml                 # Model and training configuration
├── docs/                            # Analysis reports
├── notebooks/                       # Jupyter notebooks (exploratory analysis)
├── results/                         # Experiment outputs (not tracked by git)
└── tests/                           # Unit tests
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/FAST.git
cd FAST

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install as editable package
pip install -e .
```

## Dataset Preparation

1. Download the dataset from [https://osf.io/pq7vb/](https://osf.io/pq7vb/).
2. Place the dataset in the `BCIC2020Track3/` directory:

```
BCIC2020Track3/
├── Training set/
│   ├── Data_Sample01.mat
│   ├── Data_Sample02.mat
│   └── ...
├── Validation set/
│   ├── Data_Sample01.mat
│   └── ...
└── Test set/
    ├── Data_Sample01_Test.mat
    ├── Track3_Answer Sheet_Test.xlsx
    └── ...
```

## Usage

### Data Preprocessing (Optional)

To preprocess raw data to HDF5 format:
```bash
python scripts/preprocess.py --data_folder BCIC2020Track3 --output_folder data/processed
```

### Training

Train the FAST model with 5-fold cross-validation per subject:
```bash
python scripts/train_fast.py --gpu 0 --epochs 200 --batch_size 64
```

**Baseline parity mode (GitHub FAST)**

To match the original GitHub FAST evaluation (unshuffled KFold, full-batch training,
no validation-based model selection, and no official test set):
```bash
python scripts/train_fast.py --split_mode github
```

**Our stricter split (default in this repo)**

To use shuffled KFold, validation-based model selection, and official test set evaluation:
```bash
python scripts/train_fast.py --split_mode ours
```

Available arguments:
- `--config`: Path to YAML config file (default: `configs/default.yaml`)
- `--gpu`: GPU device ID (default: 0)
- `--epochs`: Max training epochs (default: 200)
- `--batch_size`: Batch size (default: 64; ignored in `--split_mode github`)
- `--n_folds`: Number of CV folds (default: 5)
- `--seed`: Random seed (default: 42)
- `--data_folder`: Path to BCIC2020Track3 folder
- `--output_dir`: Results output directory
- `--split_mode`: `github` (baseline parity) or `ours` (strict evaluation)

### Evaluation

Aggregate and report metrics from training results:
```bash
python scripts/benchmark.py --results_dir results/finetune_official --model FAST
```

## Results

Results will be saved in the specified output directory with the following structure:
```
results/finetune_official/FAST/
├── sub-0/
│   ├── best_subject.pth         # Best model checkpoint
│   ├── fold-X_history.csv       # Training history
│   ├── fold-X_curves.png        # Learning curves
│   └── test_predictions.csv     # Test predictions
├── ...
├── summary_per_subject.csv      # Per-subject metrics
├── global_test_predictions.csv  # All test predictions
└── global_subject_accuracy.png  # Accuracy bar chart
```

In `--split_mode github`, summaries report mean fold accuracy only and no official
test set files are produced.

## Evaluation Protocol and Run 10 vs Run 9

**Runs 1–9, 11** use shuffled 5-fold cross-validation. One fold is held out as validation for model selection, and the official BCI Competition test set is evaluated separately at the end. Val Acc and Test Acc are measured on different data.

**Run 10 (Full-Data Augmented Training, No Val Split)** is a distinct protocol. All 350 training trials are used directly for training with no held-out fold. The official test set is passed as the validation monitor during training (for learning curve logging only — no early stopping is applied). Test Acc is therefore evaluated on the full official test set, but the model had access to the test set's accuracy signal throughout training. Val Acc equals Test Acc by construction because both are measured on the same set.

Run 10 also differs in: batch size 16 (vs 64), FP32 precision (vs bf16-mixed), and data augmentation.

Run 10 achieves 73.3% and Run 9 achieves 67.5%. **These numbers are not directly comparable.** Run 9 is the appropriate result for out-of-sample generalisation. The gap reflects both the protocol difference and the larger effective training set in Run 10.

## License

This project is licensed under the CBCR License - see the [LICENSE](LICENSE) file for details.
