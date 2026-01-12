"""
SHAP Explainability Analysis for FAST (Functional Areas Spatio-Temporal Transformer)
This script computes SHAP values using GradientExplainer and visualizes feature importance
for EEG classification using the trained FAST model.

GradientExplainer is used because FAST contains custom unfold/einops layers
that break standard DeepExplainer backpropagation.

Usage:
    python explain_fast.py --checkpoint Results/FAST/0_model.pth --fold 0

Contact: James Jiang Muyun (james.jiang@ntu.edu.sg)
"""

import os
import argparse
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from transformers import PretrainedConfig

# Import from existing codebase
from FAST import FAST
from BCIC2020Track3_train import load_standardized_h5
from BCIC2020Track3_preprocess import Electrodes, Zones, CLASSES

# Configuration (Matches BCIC2020Track3_train.py)
sfreq = 250
CONFIG = PretrainedConfig(
    electrodes=Electrodes,
    zone_dict=Zones,
    dim_cnn=32,
    dim_token=32,
    seq_len=800,
    window_len=sfreq,
    slide_step=sfreq // 2,
    head='Conv4Layers',
    n_classes=5,
    num_layers=4,
    num_heads=8,
    dropout=0.1,
)


def load_model(checkpoint_path, config):
    """
    Loads FAST model weights (state_dict) into the architecture.
    
    Args:
        checkpoint_path (str): Path to the .pth file containing model weights.
        config (PretrainedConfig): Model configuration object.
        
    Returns:
        FAST: Model instance loaded with weights in evaluation mode.
    """
    model = FAST(config)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    else:
        print("Warning: Checkpoint not found. Using random weights for testing.")
    model.eval()
    return model


def prepare_shap_data(data_path, fold_idx=0, n_bg=50, n_test=5):
    """
    Loads data, selects a specific subject (fold), and splits it into
    background (reference) and test (explanation) sets for SHAP.
    
    The function handles the nested structure from load_standardized_h5,
    which returns (Num_Subjects, Num_Trials, Num_Channels, Num_Timepoints).
    FAST model expects 3D Tensor (Batch, Channels, Time).
    
    Args:
        data_path (str): Path to the .h5 file.
        fold_idx (int): Index of the subject/fold to use (0-14).
        n_bg (int): Number of background samples for Gradient Integration.
        n_test (int): Number of specific samples to explain.
        
    Returns:
        tuple: (X_bg_tensor, X_test_tensor, Y_test_labels)
    """
    # 1. Load raw numpy arrays
    # Note: load_standardized_h5 returns (Subjects, Trials, Channels, Time)
    X_all, Y_all = load_standardized_h5(data_path)
    
    # 2. Select specific fold/subject
    # Shape becomes: (Trials, Channels, Time) e.g., (100, 64, 800)
    X_fold = X_all[fold_idx]
    Y_fold = Y_all[fold_idx]
    
    # 3. Convert to PyTorch Tensor
    # Critical: Must be float32 for model weights compatibility
    X_tensor = torch.tensor(X_fold, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_fold, dtype=torch.long)
    
    # 4. Create Background Distribution
    # Randomly sample 'n_bg' instances to serve as the "average" brain state
    # This acts as the baseline for the Shapley value calculation.
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    perm = torch.randperm(X_tensor.size(0))
    bg_indices = perm[:n_bg]
    test_indices = perm[n_bg : n_bg + n_test]
    
    X_bg = X_tensor[bg_indices]
    
    # 5. Create Test Set (The instances we want to explain)
    X_test = X_tensor[test_indices]
    Y_test = Y_tensor[test_indices]
    
    print(f"Data Prepared - Fold: {fold_idx}")
    print(f"  Background Shape: {X_bg.shape} (Reference)")
    print(f"  Test Shape:       {X_test.shape} (To Explain)")
    print(f"  Test Labels:      {Y_test.tolist()}")
    
    return X_bg, X_test, Y_test


def run_shap_analysis(model, X_background, X_test):
    """
    Performs SHAP analysis using GradientExplainer.
    
    GradientExplainer is used because FAST contains custom unfold/einops layers
    that break standard DeepExplainer backpropagation. GradientExplainer uses
    expected gradients, which integrates gradients over a background distribution
    to approximate Shapley values.
    
    Args:
        model (FAST): Trained FAST model in evaluation mode.
        X_background (torch.Tensor): Background distribution tensor (N_bg, Channels, Time).
        X_test (torch.Tensor): Test samples to explain (N_test, Channels, Time).
        
    Returns:
        list: SHAP values as a list of arrays (one per class).
              Each array has shape (N_test, Channels, Time).
    """
    print("Initializing SHAP GradientExplainer...")
    print(f"  Background samples: {len(X_background)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Background distribution integrates gradients to simulate missing features
    eeg_explainer = shap.GradientExplainer(model, X_background)
    
    print(f"Computing SHAP values for {len(X_test)} samples...")
    print("  This may take a few minutes depending on model complexity...")
    
    # Returns list of arrays (one per class)
    shap_values = eeg_explainer.shap_values(X_test)
    
    # ==================== INSERT THIS FIX ====================
    # FIX: Check if SHAP returns (Channels, Time, Batch) instead of (Batch, Channels, Time)
    # We know batch size is len(X_test) (e.g., 5)
    expected_batch = len(X_test)
    
    # If the first dimension is NOT the batch size, but the last one IS:
    if shap_values[0].shape[0] != expected_batch and shap_values[0].shape[-1] == expected_batch:
        print(f"\n⚠️  Detected transposed SHAP values {shap_values[0].shape}.")
        print("   -> Transposing from (Channels, Time, Batch) to (Batch, Channels, Time)...")
        # Transpose (2, 0, 1) moves the last dim (Batch) to the front
        shap_values = [np.transpose(s, (2, 0, 1)) for s in shap_values]
        print(f"   -> New shape: {shap_values[0].shape}")
    # =========================================================
    
    print(f"SHAP computation complete!")
    print(f"  Number of classes: {len(shap_values)}")
    print(f"  SHAP values shape per class: {shap_values[0].shape}")
    
    return shap_values


def plot_shap_heatmap(shap_values, sample_idx=0, class_idx=0, electrodes=Electrodes, 
                       output_dir='shap_outputs', true_label=None):
    """
    Visualizes SHAP values as a heatmap (Electrodes x Time).
    
    Color interpretation:
        - Red: Increases class probability (positive contribution)
        - Blue: Decreases class probability (negative contribution)
    
    Args:
        shap_values (list): SHAP values from run_shap_analysis.
        sample_idx (int): Index of the sample to visualize.
        class_idx (int): Index of the class to visualize.
        electrodes (list): List of electrode names for y-axis labels.
        output_dir (str): Directory to save output plots.
        true_label (int, optional): True label of the sample for title annotation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    s_vals = shap_values[class_idx][sample_idx]  # Shape: (Channels, Time)
    
    plt.figure(figsize=(14, 10))
    
    # Symmetric color scale centered at 0
    scale = np.max(np.abs(s_vals))
    im = plt.imshow(s_vals, aspect='auto', cmap='coolwarm', vmin=-scale, vmax=scale)
    
    plt.colorbar(im, label='SHAP Value (Feature Importance)', shrink=0.8)
    
    # Title with class information
    class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"Class {class_idx}"
    title = f'SHAP Values: Sample {sample_idx}, Predicted Class: {class_name}'
    if true_label is not None:
        true_class_name = CLASSES[true_label] if true_label < len(CLASSES) else f"Class {true_label}"
        title += f'\n(True Label: {true_class_name})'
    plt.title(title, fontsize=12)
    
    plt.ylabel('Electrodes', fontsize=11)
    plt.xlabel('Time Points (samples @ 250Hz)', fontsize=11)
    
    # Set electrode labels on y-axis
    if electrodes:
        plt.yticks(ticks=np.arange(len(electrodes)), labels=electrodes, fontsize=6)
    
    # Add time markers
    time_ticks = np.arange(0, s_vals.shape[1], 50)
    plt.xticks(ticks=time_ticks, labels=[f'{t/sfreq:.1f}s' for t in time_ticks], fontsize=8)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'shap_heatmap_s{sample_idx}_c{class_idx}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {output_file}")


def plot_electrode_importance(shap_values, sample_idx=0, class_idx=0, electrodes=Electrodes,
                               output_dir='shap_outputs'):
    """
    Creates a bar plot showing aggregated importance per electrode.
    
    Args:
        shap_values (list): SHAP values from run_shap_analysis.
        sample_idx (int): Index of the sample to visualize.
        class_idx (int): Index of the class to visualize.
        electrodes (list): List of electrode names.
        output_dir (str): Directory to save output plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    s_vals = shap_values[class_idx][sample_idx]  # Shape: (Channels, Time)
    
    # Aggregate importance: mean absolute SHAP value per electrode
    electrode_importance = np.mean(np.abs(s_vals), axis=1)
    
    # Sort electrodes by importance
    sorted_indices = np.argsort(electrode_importance)[::-1]
    sorted_electrodes = [electrodes[i] for i in sorted_indices]
    sorted_importance = electrode_importance[sorted_indices]
    
    # Plot top 20 electrodes
    n_show = min(20, len(electrodes))
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.coolwarm(np.linspace(0.8, 0.2, n_show))
    plt.barh(range(n_show), sorted_importance[:n_show][::-1], color=colors[::-1])
    plt.yticks(range(n_show), sorted_electrodes[:n_show][::-1], fontsize=9)
    
    class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"Class {class_idx}"
    plt.title(f'Top {n_show} Electrodes by Importance (Sample {sample_idx}, Class: {class_name})', fontsize=12)
    plt.xlabel('Mean |SHAP Value|', fontsize=11)
    plt.ylabel('Electrode', fontsize=11)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'electrode_importance_s{sample_idx}_c{class_idx}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved electrode importance plot to {output_file}")


def plot_zone_importance(shap_values, sample_idx=0, class_idx=0, electrodes=Electrodes,
                          zones=Zones, output_dir='shap_outputs'):
    """
    Creates a bar plot showing aggregated importance per brain zone.
    
    Args:
        shap_values (list): SHAP values from run_shap_analysis.
        sample_idx (int): Index of the sample to visualize.
        class_idx (int): Index of the class to visualize.
        electrodes (list): List of electrode names.
        zones (dict): Dictionary mapping zone names to electrode lists.
        output_dir (str): Directory to save output plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    s_vals = shap_values[class_idx][sample_idx]  # Shape: (Channels, Time)
    
    # Create electrode to index mapping
    electrode_to_idx = {e: i for i, e in enumerate(electrodes)}
    
    # Aggregate importance per zone
    zone_importance = {}
    for zone_name, zone_electrodes in zones.items():
        zone_indices = [electrode_to_idx[e] for e in zone_electrodes if e in electrode_to_idx]
        if zone_indices:
            zone_vals = s_vals[zone_indices]
            zone_importance[zone_name] = np.mean(np.abs(zone_vals))
    
    # Sort by importance
    sorted_zones = sorted(zone_importance.items(), key=lambda x: x[1], reverse=True)
    zone_names = [z[0] for z in sorted_zones]
    importance_values = [z[1] for z in sorted_zones]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(zone_names)))
    bars = plt.barh(range(len(zone_names)), importance_values[::-1], color=colors)
    plt.yticks(range(len(zone_names)), zone_names[::-1], fontsize=10)
    
    class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"Class {class_idx}"
    plt.title(f'Brain Zone Importance (Sample {sample_idx}, Class: {class_name})', fontsize=12)
    plt.xlabel('Mean |SHAP Value|', fontsize=11)
    plt.ylabel('Brain Zone', fontsize=11)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'zone_importance_s{sample_idx}_c{class_idx}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved zone importance plot to {output_file}")


def generate_all_visualizations(shap_values, Y_test, electrodes=Electrodes, zones=Zones,
                                 output_dir='shap_outputs'):
    """
    Generates all visualization types for all samples and their predicted classes.
    
    Args:
        shap_values (list): SHAP values from run_shap_analysis.
        Y_test (torch.Tensor): True labels for test samples.
        electrodes (list): List of electrode names.
        zones (dict): Dictionary mapping zone names to electrode lists.
        output_dir (str): Directory to save output plots.
    """
    n_samples = shap_values[0].shape[0]
    n_classes = len(shap_values)
    
    print(f"\nGenerating visualizations for {n_samples} samples across {n_classes} classes...")
    
    for sample_idx in range(n_samples):
        true_label = Y_test[sample_idx].item() if Y_test is not None else None
        
        # Find the class with highest mean absolute SHAP (most relevant prediction)
        class_importance = [np.mean(np.abs(shap_values[c][sample_idx])) for c in range(n_classes)]
        predicted_class = np.argmax(class_importance)
        
        print(f"\n--- Sample {sample_idx} (True: {CLASSES[true_label] if true_label is not None else 'N/A'}, "
              f"Predicted: {CLASSES[predicted_class]}) ---")
        
        # Generate plots for the predicted class
        plot_shap_heatmap(shap_values, sample_idx, predicted_class, electrodes, 
                          output_dir, true_label)
        plot_electrode_importance(shap_values, sample_idx, predicted_class, electrodes, output_dir)
        plot_zone_importance(shap_values, sample_idx, predicted_class, electrodes, zones, output_dir)
        
        # Also generate heatmap for true class if different
        if true_label is not None and true_label != predicted_class:
            plot_shap_heatmap(shap_values, sample_idx, true_label, electrodes, 
                              output_dir, true_label)


def main():
    parser = argparse.ArgumentParser(description='SHAP Explainability Analysis for FAST')
    parser.add_argument('--checkpoint', type=str, default='Results/FAST/0_model.pth',
                        help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--data', type=str, default='Processed/BCIC2020Track3.h5',
                        help='Path to the processed data file')
    parser.add_argument('--fold', type=int, default=0,
                        help='Subject/fold index to use (0-14)')
    parser.add_argument('--n_bg', type=int, default=50,
                        help='Number of background samples for SHAP')
    parser.add_argument('--n_test', type=int, default=5,
                        help='Number of test samples to explain')
    parser.add_argument('--output_dir', type=str, default='shap_outputs',
                        help='Directory to save output visualizations')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SHAP Explainability Analysis for FAST")
    print("=" * 60)
    
    # 1. Check data file exists
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found at {args.data}")
        print("Please run BCIC2020Track3_preprocess.py first to generate the data.")
        return
    
    # 2. Prepare data for SHAP analysis
    print("\n[Step 1/4] Loading and preparing data...")
    X_bg, X_explain, Y_explain = prepare_shap_data(
        args.data, 
        fold_idx=args.fold, 
        n_bg=args.n_bg, 
        n_test=args.n_test
    )
    
    # ==================== INSERT THIS FIX ====================
    # PATCH: Ensure Electrodes list matches data dimensions
    actual_n_channels = X_explain.shape[1] # Should be 64
    
    if len(Electrodes) < actual_n_channels:
        print(f"\n⚠️  WARNING: Mismatch detected!")
        print(f"   Data Channels: {actual_n_channels}")
        print(f"   Electrode Names: {len(Electrodes)}")
        print("   -> Padding Electrodes list with placeholders to prevent crash.")
        
        # Add placeholders for the extra channels
        for i in range(len(Electrodes), actual_n_channels):
            Electrodes.append(f"Unk_Ch{i}")
    
    # 3. Load trained model
    print("\n[Step 2/4] Loading trained model...")
    model = load_model(args.checkpoint, CONFIG)
    
    # 4. Run SHAP analysis
    print("\n[Step 3/4] Computing SHAP values...")
    shap_vals = run_shap_analysis(model, X_bg, X_explain)
    
    # 5. Generate visualizations
    print("\n[Step 4/4] Generating visualizations...")
    generate_all_visualizations(shap_vals, Y_explain, Electrodes, Zones, args.output_dir)
    
    print("\n" + "=" * 60)
    print("SHAP Analysis Complete!")
    print(f"Visualizations saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
