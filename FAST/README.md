# FAST: Functional Areas Spatio-Temporal Transformer

Code for paper: **Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)**

This codebase reproduces results on the publicly available dataset [BCI Competition 2020 Track #3: Imagined Speech Classification](https://osf.io/pq7vb/).

Contact: James Jiang Muyun (james.jiang@ntu.edu.sg)

## Project Structure

```
FAST/
├── src/
│   └── fast/                    # Main package
│       ├── models/              # FAST model architecture
│       ├── data/                # Data loading and preprocessing
│       ├── train/               # Training utilities
│       └── utils.py             # Helper functions
├── scripts/
│   ├── train_fast.py            # Training script
│   ├── benchmark.py             # Evaluation script
│   └── preprocess.py            # Data preprocessing
├── configs/
│   └── default.yaml             # Configuration file
├── notebooks/                   # Jupyter notebooks
├── docs/                        # Documentation
└── tests/                       # Unit tests
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/FAST.git
cd FAST

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

1. Download the dataset from [https://osf.io/pq7vb/](https://osf.io/pq7vb/).
2. Place the dataset in the `data/BCIC2020Track3/` directory:

```
data/BCIC2020Track3/
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
python scripts/preprocess.py --data_folder data/BCIC2020Track3 --output_folder data/processed
```

### Training

Train the FAST model with 5-fold cross-validation per subject:
```bash
python scripts/train_fast.py --gpu 0 --epochs 200 --batch_size 64
```

Available arguments:
- `--config`: Path to YAML config file (default: `configs/default.yaml`)
- `--gpu`: GPU device ID (default: 0)
- `--epochs`: Max training epochs (default: 200)
- `--batch_size`: Batch size (default: 64)
- `--n_folds`: Number of CV folds (default: 5)
- `--seed`: Random seed (default: 42)
- `--data_folder`: Path to BCIC2020Track3 folder
- `--output_dir`: Results output directory

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

## Citation

Please cite our paper if you find this work useful:

```bibtex
@article{jiang2025decoding,
  title={Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer},
  author={Jiang, Muyun and Ding, Yi and Zhang, Wei and Teo, Kok Ann Colin and Fong, LaiGuan and Zhang, Shuailei and Guo, Zhiwei and Liu, Chenyu and Bhuvanakantham, Raghavan and Sim, Wei Khang Jeremy and others},
  journal={arXiv preprint arXiv:2504.03762},
  year={2025}
}
```

## License

This project is licensed under the CBCR License - see the [CBCR License](CBCR%20License) file for details.
