import os
import argparse
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import mne  # <--- IMPORT MNE
from transformers import PretrainedConfig

# Import from existing codebase
from FAST import FAST
from BCIC2020Track3_train import load_standardized_h5
from BCIC2020Track3_preprocess import Electrodes, Zones, CLASSES

# --- CONFIGURATION ---
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

def load_model(checkpoint_path, config,device):
    """
    Loads FAST model weights (state_dict) into the architecture.
    """
    model = FAST(config)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # --- FIX: Add weights_only=False ---
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    else:
        print("Warning: Checkpoint not found. Using random weights for testing.")
    model.eval()
    model.to(device)
    return model

def prepare_shap_data(data_path, fold_idx=0, n_bg=50, n_test=5):
    X_all, Y_all = load_standardized_h5(data_path)
    X_fold = X_all[fold_idx]
    Y_fold = Y_all[fold_idx]
    
    X_tensor = torch.tensor(X_fold, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_fold, dtype=torch.long)
    
    torch.manual_seed(42)
    perm = torch.randperm(X_tensor.size(0))
    bg_indices = perm[:n_bg]
    test_indices = perm[n_bg : n_bg + n_test]
    
    X_bg = X_tensor[bg_indices]
    X_test = X_tensor[test_indices]
    Y_test = Y_tensor[test_indices]
    
    return X_bg, X_test, Y_test

def run_shap_analysis(model, X_background, X_test,device):
    """
    Performs SHAP analysis using GradientExplainer.
    Robustly handles both List output and Stacked Array output formats.
    """
    print("Initializing SHAP GradientExplainer...")
    print(f"  Background samples: {len(X_background)}")
    print(f"  Test samples: {len(X_test)}")
    
        # 1. Move Data to GPU
    X_background = X_background.to(device)
    X_test = X_test.to(device)

    eeg_explainer = shap.GradientExplainer(model, X_background)
    
    print(f"Computing SHAP values for {len(X_test)} samples...")
    shap_values = eeg_explainer.shap_values(X_test)

    # --- FIX: Detect and Unpack Single Array Output ---
    if not isinstance(shap_values, list):
        print(f"  -> Detected single SHAP array of shape {shap_values.shape}")
        
        # If shape is (Batch, Channels, Time, Classes) -> (50, 64, 800, 5)
        if shap_values.ndim == 4 and shap_values.shape[-1] == 5:
            print("  -> Unpacking (Batch, Channels, Time, Classes) into List of Classes...")
            shap_values = np.transpose(shap_values, (3, 0, 1, 2))
            shap_values = [shap_values[i] for i in range(shap_values.shape[0])]
            
        # If shape is (Batch, Classes, Channels, Time) -> (50, 5, 64, 800)
        elif shap_values.ndim == 4 and shap_values.shape[1] == 5:
             print("  -> Unpacking (Batch, Classes, Channels, Time) into List of Classes...")
             shap_values = np.transpose(shap_values, (1, 0, 2, 3))
             shap_values = [shap_values[i] for i in range(shap_values.shape[0])]

    # --- Standard Check for Transposed Dimensions ---
    expected_batch = len(X_test)
    
    if isinstance(shap_values, list) and len(shap_values) > 0:
        s_shape = shap_values[0].shape
        # If Batch is the LAST dimension (e.g. 64, 800, 50)
        if s_shape[0] != expected_batch and s_shape[-1] == expected_batch:
            print(f"  -> Transposing dimensions from {s_shape} to (Batch, Channels, Time)...")
            shap_values = [np.transpose(s, (2, 0, 1)) for s in shap_values]

    print(f"SHAP computation complete!")
    print(f"  Number of classes detected: {len(shap_values)} (Should be 5)")
    print(f"  Shape per class: {shap_values[0].shape} (Should be {expected_batch}, 64, 800)")
    
    return shap_values

# --- MNE VISUALIZATION FUNCTIONS ---

def plot_shap_heatmap(shap_values, sample_idx=0, class_idx=0, electrodes=Electrodes, 
                        output_dir='shap_outputs', true_label=None, vmax=None):
    """Standard Heatmap (Electrodes x Time) using Matplotlib."""
    os.makedirs(output_dir, exist_ok=True)
    s_vals = shap_values[class_idx][sample_idx]
    
    plt.figure(figsize=(14, 10))
    # Scale uses abs(max) to ensure symmetric range covers the data
    scale = vmax if vmax is not None else np.max(np.abs(s_vals))    
    
    # cmap='RdBu_r' is Red-Blue reversed (Red=Positive, Blue=Negative)
    im = plt.imshow(s_vals, aspect='auto', cmap='RdBu_r', vmin=-scale, vmax=scale)
    plt.colorbar(im, label='SHAP Value', shrink=0.8)
    
    class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"Class {class_idx}"
    title = f'Heatmap: Sample {sample_idx}, Predicted: {class_name}'
    if true_label is not None:
        title += f' (True: {CLASSES[true_label]})'
    plt.title(title)
    plt.ylabel('Electrodes')
    plt.xlabel('Time')
    if electrodes:
        plt.yticks(ticks=np.arange(len(electrodes)), labels=electrodes, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_s{sample_idx}_c{class_idx}.png'), dpi=150)
    plt.close()

def plot_mne_topomap(shap_values, sample_idx=0, class_idx=0, electrodes=Electrodes, 
                     output_dir='shap_outputs', true_label=None, vmin=0, vmax=None):
    """
    Generates a 2D Brain Topomap using MNE.
    UPDATED: Raw values (No Abs), Symmetric Scale, Divergent Colormap.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Aggregate Time: Get one value per electrode (Mean RAW SHAP)
    s_vals_time = shap_values[class_idx][sample_idx]
    
    # --- CHANGED: Removed np.abs() to preserve direction ---
    s_vals_agg = np.mean(s_vals_time, axis=1) 
    
    # 2. Setup MNE Info
    clean_names = [e.replace('FP', 'Fp').replace('Z', 'z').replace('z', 'z') for e in electrodes]
    info = mne.create_info(ch_names=clean_names, sfreq=250, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(6, 6))

    # Calculate limit based on magnitude to ensure symmetric scale fits
    limit = vmax if vmax else np.max(np.abs(s_vals_agg))
    
    # --- CHANGED: Divergent Colormap & Symmetric Scale ---
    im, _ = mne.viz.plot_topomap(
        s_vals_agg, 
        info, 
        axes=ax, 
        show=False, 
        cmap='RdBu_r',  # Red = Positive, Blue = Negative
        contours=6,
        vlim=(-limit, limit), # Symmetric scale centered at 0
        extrapolate='head', 
        sphere=None         
    )
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Mean SHAP (Net Influence)')
    
    class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"Class {class_idx}"
    title = f'Brain Influence: Sample {sample_idx}\nPred: {class_name}'
    if true_label is not None:
        title += f' (True: {CLASSES[true_label]})'
        
    ax.set_title(title, fontsize=14)
    
    output_file = os.path.join(output_dir, f'topomap_s{sample_idx}_c{class_idx}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved MNE Topomap to {output_file}")

def plot_correct_only_average(shap_values, Y_test, target_class_idx, electrodes=Electrodes, output_dir='shap_outputs', vmax=None):
    """
    Plots the average SHAP topology ONLY for correct predictions.
    UPDATED: Raw values (No Abs), Symmetric Scale.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = shap_values[0].shape[0]
    n_classes = len(shap_values)
    
    correct_indices = []
    
    # 1. Identify Correct Predictions
    for i in range(n_samples):
        # Prediction heuristic: Class with highest total ABSOLUTE impact 
        # (We still use abs here to determine 'active' prediction, but plot raw later)
        class_scores = [np.sum(np.abs(shap_values[c][i])) for c in range(n_classes)]
        predicted_class = np.argmax(class_scores)
        
        true_label = Y_test[i].item()
        
        if true_label == target_class_idx and predicted_class == target_class_idx:
            correct_indices.append(i)
            
    if len(correct_indices) == 0:
        print(f"Skipping Class {CLASSES[target_class_idx]}: No correct predictions found.")
        return

    print(f"Class {CLASSES[target_class_idx]}: Averaging {len(correct_indices)} correct samples.")

    # 2. Filter SHAP values
    relevant_shap = shap_values[target_class_idx][correct_indices]
    
    # 3. Aggregation: Mean RAW Importance (Removed np.abs)
    # Average over time, then average over samples
    avg_saliency = np.mean(np.mean(relevant_shap, axis=2), axis=0)
    
    # 4. MNE Setup
    clean_names = [e.replace('FP', 'Fp').replace('Z', 'z').replace('z', 'z') for e in electrodes]
    info = mne.create_info(ch_names=clean_names, sfreq=250, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')
    
    limit = vmax if vmax is not None else np.max(np.abs(avg_saliency))
    
    # 5. Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # --- CHANGED: Divergent Colormap & Symmetric Scale ---
    im, _ = mne.viz.plot_topomap(
        avg_saliency, 
        info, 
        axes=ax, 
        show=False, 
        cmap='RdBu_r', 
        contours=6,   
        vlim=(-limit, limit),     
        extrapolate='head',
        sphere=None,
        sensors=True
    )
    
    class_name = CLASSES[target_class_idx]
    plt.title(f'CORRECT ONLY Average (Net Influence)\n(Target: {class_name}, N={len(correct_indices)})', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Mean SHAP Value')
    
    output_file = os.path.join(output_dir, f'CorrectOnly_Avg_Class_{class_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Correct-Only Map to: {output_file}")


def plot_error_average_topomap(shap_values, Y_test, target_class_idx, electrodes=Electrodes, output_dir='shap_outputs', vmax=None):
    """
    Plots the average SHAP topology ONLY for incorrect predictions.
    UPDATED: Raw values (No Abs), Symmetric Scale.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = shap_values[0].shape[0]
    n_classes = len(shap_values)
    
    error_indices = []
    
    # 1. Identify Incorrect Predictions
    for i in range(n_samples):
        class_scores = [np.sum(np.abs(shap_values[c][i])) for c in range(n_classes)]
        predicted_class = np.argmax(class_scores)
        
        true_label = Y_test[i].item()
        
        if true_label == target_class_idx and predicted_class != target_class_idx:
            error_indices.append(i)
            
    if len(error_indices) == 0:
        print(f"Skipping Error Map for Class {CLASSES[target_class_idx]}: Model was 100% correct.")
        return

    print(f"Class {CLASSES[target_class_idx]}: Averaging {len(error_indices)} ERROR samples.")

    # 2. Filter SHAP values
    relevant_shap = shap_values[target_class_idx][error_indices]
    
    # 3. Aggregation: Mean RAW Importance (Removed np.abs)
    avg_saliency = np.mean(np.mean(relevant_shap, axis=2), axis=0)
    
    # 4. MNE Setup
    clean_names = [e.replace('FP', 'Fp').replace('Z', 'z').replace('z', 'z') for e in electrodes]
    info = mne.create_info(ch_names=clean_names, sfreq=250, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')
    
    limit = vmax if vmax is not None else np.max(np.abs(avg_saliency))
    
    # 5. Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # --- CHANGED: Divergent Colormap & Symmetric Scale ---
    im, _ = mne.viz.plot_topomap(
        avg_saliency, 
        info, 
        axes=ax, 
        show=False, 
        cmap='RdBu_r', 
        contours=6,   
        vlim=(-limit, limit),     
        extrapolate='head',
        sphere=None,
        sensors=True
    )
    
    class_name = CLASSES[target_class_idx]
    plt.title(f'ERROR ONLY Average (Net Influence)\n(Target: {class_name}, N={len(error_indices)})', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Mean SHAP Value')
    
    output_file = os.path.join(output_dir, f'ErrorOnly_Avg_Class_{class_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Error-Only Map to: {output_file}")


def plot_zone_importance(shap_values, sample_idx=0, class_idx=0, electrodes=Electrodes,
                          zones=Zones, output_dir='shap_outputs', true_label=None):
    """
    Creates a bar plot showing aggregated importance per brain zone.
    UPDATED: Showing Mean Net Influence (Raw) to match other plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    s_vals = shap_values[class_idx][sample_idx]
    
    electrode_to_idx = {e: i for i, e in enumerate(electrodes)}
    
    zone_importance = {}
    for zone_name, zone_electrodes in zones.items():
        zone_indices = [electrode_to_idx[e] for e in zone_electrodes if e in electrode_to_idx]
        if zone_indices:
            zone_vals = s_vals[zone_indices]
            # --- CHANGED: Mean RAW value (Net Influence) ---
            zone_importance[zone_name] = np.mean(zone_vals)
    
    # Sort by raw value (Most Positive to Most Negative)
    sorted_zones = sorted(zone_importance.items(), key=lambda x: x[1], reverse=True)
    zone_names = [z[0] for z in sorted_zones]
    importance_values = [z[1] for z in sorted_zones]
    
    plt.figure(figsize=(10, 6))
    # Using Coolwarm colors based on value
    norm = plt.Normalize(min(importance_values), max(importance_values))
    colors = plt.cm.RdBu_r(norm(importance_values))
    
    plt.barh(range(len(zone_names)), importance_values[::-1], color=colors[::-1])
    plt.yticks(range(len(zone_names)), zone_names[::-1], fontsize=10)
    
    class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"Class {class_idx}"
    title = f'Brain Zone Net Influence (Sample {sample_idx})\nPredicted: {class_name}'
    
    if true_label is not None:
         true_name = CLASSES[true_label] if true_label < len(CLASSES) else f"Class {true_label}"
         title += f' (True: {true_name})'
         
    plt.title(title, fontsize=12)
    plt.xlabel('Mean SHAP Value (Red=Pos, Blue=Neg)', fontsize=11)
    plt.ylabel('Brain Functional Area', fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.8) # Add zero line for reference
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'zone_importance_s{sample_idx}_c{class_idx}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved zone importance plot to {output_file}")

global_vmax = 0
def generate_all_visualizations(shap_values, Y_test, electrodes=Electrodes, zones=Zones,
                                 output_dir='shap_outputs'):
    global global_vmax
    n_samples = shap_values[0].shape[0]
    n_classes = len(shap_values)

    # Calculating global scale from all shap values
    all_vals = []
    for c in range(n_classes):
        all_vals.append(np.abs(shap_values[c])) # Still use ABS to find the Max Extent (Range)
    
    all_vals_flat = np.concatenate([v.flatten() for v in all_vals])
    
    # Using 99th percentile of MAGNITUDE to set the bounds [-vmax, +vmax]
    global_vmax = np.percentile(all_vals_flat, 99)
    print(f"Global Color Scale fixed at: +/- {global_vmax:.4f}")

    print(f"\nGenerating visualizations for {n_samples} samples across {n_classes} classes...")
    
    for sample_idx in range(n_samples):
        true_label = Y_test[sample_idx].item() if Y_test is not None else None
        
        # Find prediction
        class_importance = [np.mean(np.abs(shap_values[c][sample_idx])) for c in range(n_classes)]
        predicted_class = np.argmax(class_importance)
        
        print(f"\n--- Sample {sample_idx} [Pred: {CLASSES[predicted_class]}] ---")
        
        # 1. Heatmap (Raw Matrix)
        plot_shap_heatmap(shap_values, sample_idx, predicted_class, electrodes, output_dir, true_label, vmax=global_vmax)
        
        # 2. Brain Topomap (MNE Saliency)
        plot_mne_topomap(shap_values, sample_idx, predicted_class, electrodes, output_dir, true_label, vmax=global_vmax)

        # 3. Zone Importance
        plot_zone_importance(shap_values, sample_idx, predicted_class, electrodes, zones, output_dir, true_label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/kay/FAST/FAST/Results_finetune_only/FAST/3_best.pth')
    parser.add_argument('--data', type=str, default='Processed/BCIC2020Track3.h5')
    parser.add_argument('--fold', type=int, default=3)
    parser.add_argument('--n_bg', type=int, default=200)
    parser.add_argument('--n_test', type=int, default=150)
    parser.add_argument('--output_dir', type=str, default='shap_outputs_sub_3')
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found at {args.data}")
        return
    
        # --- GPU SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")


    X_bg, X_explain, Y_explain = prepare_shap_data(args.data, args.fold, args.n_bg, args.n_test)
    
    # Patch electrodes if mismatch
    actual_n_channels = X_explain.shape[1]
    if len(Electrodes) < actual_n_channels:
        for i in range(len(Electrodes), actual_n_channels):
            Electrodes.append(f"Ch{i}")

    model = load_model(args.checkpoint, CONFIG,device)
    shap_vals = run_shap_analysis(model, X_bg, X_explain,device)
    generate_all_visualizations(shap_vals, Y_explain, Electrodes, Zones, args.output_dir)
    n_classes = len(shap_vals)
    for c in range(n_classes):
        
        # Correct Only Plot
        plot_correct_only_average(shap_vals, Y_explain, target_class_idx=c, 
                                  electrodes=Electrodes, output_dir=args.output_dir, vmax=global_vmax)
        
        # Error Only Plot
        plot_error_average_topomap(shap_vals, Y_explain, target_class_idx=c, 
                                   electrodes=Electrodes, output_dir=args.output_dir, vmax=global_vmax)
        
    print("\nComplete!")

if __name__ == '__main__':
    main()