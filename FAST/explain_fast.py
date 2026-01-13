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

def load_model(checkpoint_path, config):
    model = FAST(config)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    else:
        print("Warning: Checkpoint not found. Using random weights for testing.")
    model.eval()
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

def run_shap_analysis(model, X_background, X_test):
    print("Initializing SHAP GradientExplainer...")
    eeg_explainer = shap.GradientExplainer(model, X_background)
    
    print(f"Computing SHAP values for {len(X_test)} samples...")
    shap_values = eeg_explainer.shap_values(X_test)

    expected_batch = len(X_test)
    if shap_values[0].shape[0] != expected_batch and shap_values[0].shape[-1] == expected_batch:
        print(f"   -> Transposing from (Channels, Time, Batch) to (Batch, Channels, Time)...")
        shap_values = [np.transpose(s, (2, 0, 1)) for s in shap_values]
    
    return shap_values

# --- MNE VISUALIZATION FUNCTIONS ---

def plot_shap_heatmap(shap_values, sample_idx=0, class_idx=0, electrodes=Electrodes, 
                       output_dir='shap_outputs', true_label=None):
    """Standard Heatmap (Electrodes x Time) using Matplotlib."""
    os.makedirs(output_dir, exist_ok=True)
    s_vals = shap_values[class_idx][sample_idx]
    
    plt.figure(figsize=(14, 10))
    scale = np.max(np.abs(s_vals))
    im = plt.imshow(s_vals, aspect='auto', cmap='coolwarm', vmin=-scale, vmax=scale)
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
                     output_dir='shap_outputs', true_label=None):
    """
    Generates a 2D Brain Topomap using MNE.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Aggregate Time: Get one value per electrode (Mean Absolute SHAP)
    # This represents "Total Saliency" of that electrode
    s_vals_time = shap_values[class_idx][sample_idx]
    s_vals_agg = np.mean(np.abs(s_vals_time), axis=1)
    
    # 2. Setup MNE Info
    # Create an info object with the electrode names and standard montage
    # We filter 'electrodes' to ensure valid names if necessary
    
    # Simple cleaner for electrode names (MNE prefers 'Fp1' over 'FP1')
    clean_names = [e.replace('FP', 'Fp').replace('Z', 'z').replace('z', 'z') for e in electrodes]
    
    info = mne.create_info(ch_names=clean_names, sfreq=250, ch_types='eeg')
    
    # Load standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Apply montage (this maps names to 3D head positions)
    # on_missing='ignore' ensures it doesn't crash if you have a weird custom channel
    info.set_montage(montage, on_missing='ignore')
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # MNE plot_topomap
    # cmap='inferno' or 'Reds' is good for magnitude (Importance)
    im, _ = mne.viz.plot_topomap(
        s_vals_agg, 
        info, 
        axes=ax, 
        show=False, 
        cmap='inferno', 
        contours=6,
        extrapolate='head', # Makes it look like a full circle head
        sphere=None         # Auto-detect sphere from montage
    )
    
    # Add colorbar manually since MNE's internal handling can be tricky with subplots
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Mean |SHAP| (Feature Importance)')
    
    # Title
    class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"Class {class_idx}"
    title = f'Brain Saliency: Sample {sample_idx}\nPred: {class_name}'
    if true_label is not None:
        title += f' (True: {CLASSES[true_label]})'
        
    ax.set_title(title, fontsize=14)
    
    output_file = os.path.join(output_dir, f'topomap_s{sample_idx}_c{class_idx}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved MNE Topomap to {output_file}")

def plot_grand_average_topomap(shap_values, Y_test, target_class_idx, electrodes=Electrodes, output_dir='shap_outputs'):
    """
    Computes the Mean Absolute SHAP value across ALL test samples for a specific class
    and plots a single, smoothed topography map.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Filter: Find all samples that actually belong to this class
    # (Or you can average all samples if you want 'Global Importance' regardless of class)
    # For now, let's take ALL test samples provided to see general importance
    class_shap_vals = shap_values[target_class_idx] # Shape: (N_samples, Channels, Time)
    
    # 2. Aggregation:
    # Step A: Take Absolute Value (Magnitude of importance)
    # Step B: Average over Time (Collapse time dimension) -> (N_samples, Channels)
    # Step C: Average over Samples (Collapse batch dimension) -> (Channels,)
    
    # NOTE: We use ABS because EEG oscillates. If we average raw values, 
    # positive and negative peaks cancel out, resulting in zero.
    avg_saliency = np.mean(np.mean(np.abs(class_shap_vals), axis=2), axis=0)
    
    # 3. MNE Setup
    clean_names = [e.replace('FP', 'Fp').replace('Z', 'z').replace('z', 'z') for e in electrodes]
    info = mne.create_info(ch_names=clean_names, sfreq=250, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')
    
    # 4. Plotting (Matches your image style)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im, _ = mne.viz.plot_topomap(
        avg_saliency, 
        info, 
        axes=ax, 
        show=False, 
        cmap='Reds',       # Use 'Reds' or 'viridis' for Magnitude. Use 'RdBu_r' if using raw signed values.
        contours=6,        # Adds the contour lines seen in your image
        extrapolate='head',# Extends map to the full head circle
        sphere=None,
        names=clean_names, # Adds Electrode Names
        show_names=True    # Shows the Fp1, Cz labels
    )
    
    class_name = CLASSES[target_class_idx]
    plt.title(f'Grand Average Saliency Map\n(Target: {class_name})', fontsize=14)
    
    # Add Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Mean Feature Importance')
    
    output_file = os.path.join(output_dir, f'GrandAverage_Class_{class_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Grand Average Map to: {output_file}")

def plot_zone_importance(shap_values, sample_idx=0, class_idx=0, electrodes=Electrodes,
                          zones=Zones, output_dir='shap_outputs', true_label=None):
    """
    Creates a bar plot showing aggregated importance per brain zone.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    s_vals = shap_values[class_idx][sample_idx]
    
    electrode_to_idx = {e: i for i, e in enumerate(electrodes)}
    
    zone_importance = {}
    for zone_name, zone_electrodes in zones.items():
        zone_indices = [electrode_to_idx[e] for e in zone_electrodes if e in electrode_to_idx]
        if zone_indices:
            zone_vals = s_vals[zone_indices]
            zone_importance[zone_name] = np.mean(np.abs(zone_vals))
    
    sorted_zones = sorted(zone_importance.items(), key=lambda x: x[1], reverse=True)
    zone_names = [z[0] for z in sorted_zones]
    importance_values = [z[1] for z in sorted_zones]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(zone_names)))
    
    plt.barh(range(len(zone_names)), importance_values[::-1], color=colors)
    plt.yticks(range(len(zone_names)), zone_names[::-1], fontsize=10)
    
    # --- Title Logic (Updated) ---
    class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"Class {class_idx}"
    title = f'Brain Zone Importance (Sample {sample_idx})\nPredicted: {class_name}'
    
    if true_label is not None:
         true_name = CLASSES[true_label] if true_label < len(CLASSES) else f"Class {true_label}"
         title += f' (True: {true_name})'
         
    plt.title(title, fontsize=12)
    plt.xlabel('Mean |SHAP Value| (Impact)', fontsize=11)
    plt.ylabel('Brain Functional Area', fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'zone_importance_s{sample_idx}_c{class_idx}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved zone importance plot to {output_file}")


def generate_all_visualizations(shap_values, Y_test, electrodes=Electrodes, zones=Zones,
                                 output_dir='shap_outputs'):
    n_samples = shap_values[0].shape[0]
    n_classes = len(shap_values)
    
    print(f"\nGenerating visualizations for {n_samples} samples across {n_classes} classes...")
    
    for sample_idx in range(n_samples):
        true_label = Y_test[sample_idx].item() if Y_test is not None else None
        
        # Find prediction
        class_importance = [np.mean(np.abs(shap_values[c][sample_idx])) for c in range(n_classes)]
        predicted_class = np.argmax(class_importance)
        
        print(f"\n--- Sample {sample_idx} [Pred: {CLASSES[predicted_class]}] ---")
        
        # 1. Heatmap (Raw Matrix)
        plot_shap_heatmap(shap_values, sample_idx, predicted_class, electrodes, output_dir, true_label)
        
        # 2. Brain Topomap (MNE Saliency)
        plot_mne_topomap(shap_values, sample_idx, predicted_class, electrodes, output_dir, true_label)

        # 3. Zone Importance (Add this line)
        plot_zone_importance(shap_values, sample_idx, predicted_class, electrodes, zones, output_dir, true_label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Results/FAST/6_best.pth')
    parser.add_argument('--data', type=str, default='Processed/BCIC2020Track3.h5')
    parser.add_argument('--fold', type=int, default=6)
    parser.add_argument('--n_bg', type=int, default=125)
    parser.add_argument('--n_test', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='shap_outputs')
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found at {args.data}")
        return
    
    X_bg, X_explain, Y_explain = prepare_shap_data(args.data, args.fold, args.n_bg, args.n_test)
    
    # Patch electrodes if mismatch
    actual_n_channels = X_explain.shape[1]
    if len(Electrodes) < actual_n_channels:
        for i in range(len(Electrodes), actual_n_channels):
            Electrodes.append(f"Ch{i}")

    model = load_model(args.checkpoint, CONFIG)
    shap_vals = run_shap_analysis(model, X_bg, X_explain)
    generate_all_visualizations(shap_vals, Y_explain, Electrodes, Zones, args.output_dir)
    n_classes = len(shap_vals)
    for c in range(n_classes):
        plot_grand_average_topomap(shap_vals, Y_explain, target_class_idx=c, electrodes=Electrodes, output_dir=args.output_dir)

    print("\nComplete!")

if __name__ == '__main__':
    main()