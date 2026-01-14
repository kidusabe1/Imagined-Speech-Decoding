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
    """
    Loads FAST model weights (state_dict) into the architecture.
    """
    model = FAST(config)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # --- FIX: Add weights_only=False ---
        # We also check if it's a Lightning checkpoint (which nests weights under 'state_dict')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            # If loading a PL checkpoint, weights might have 'model.' prefix
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if it exists (common in Lightning modules)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
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
    """
    Performs SHAP analysis using GradientExplainer.
    Robustly handles both List output and Stacked Array output formats.
    """
    print("Initializing SHAP GradientExplainer...")
    print(f"  Background samples: {len(X_background)}")
    print(f"  Test samples: {len(X_test)}")
    
    eeg_explainer = shap.GradientExplainer(model, X_background)
    
    print(f"Computing SHAP values for {len(X_test)} samples...")
    shap_values = eeg_explainer.shap_values(X_test)

    # --- FIX: Detect and Unpack Single Array Output ---
    # Case 1: SHAP returns a single array (Batch, Channels, Time, Classes)
    # We need to convert it to -> List of [ (Batch, Channels, Time) ] per class
    if not isinstance(shap_values, list):
        print(f"  -> Detected single SHAP array of shape {shap_values.shape}")
        
        # If shape is (Batch, Channels, Time, Classes) -> (50, 64, 800, 5)
        if shap_values.ndim == 4 and shap_values.shape[-1] == 5:
            print("  -> Unpacking (Batch, Channels, Time, Classes) into List of Classes...")
            # 1. Transpose to (Classes, Batch, Channels, Time)
            shap_values = np.transpose(shap_values, (3, 0, 1, 2))
            # 2. Convert to list of numpy arrays
            shap_values = [shap_values[i] for i in range(shap_values.shape[0])]
            
        # If shape is (Batch, Classes, Channels, Time) -> (50, 5, 64, 800)
        elif shap_values.ndim == 4 and shap_values.shape[1] == 5:
             print("  -> Unpacking (Batch, Classes, Channels, Time) into List of Classes...")
             shap_values = np.transpose(shap_values, (1, 0, 2, 3))
             shap_values = [shap_values[i] for i in range(shap_values.shape[0])]

    # --- Standard Check for Transposed Dimensions ---
    # Sometimes output is (Classes x [Channels, Time, Batch]) instead of [Batch, Channels, Time]
    expected_batch = len(X_test)
    
    # Check the first class array
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
    class_shap_vals = shap_values[target_class_idx] 
    
    # 2. Aggregation: Mean Absolute Importance
    avg_saliency = np.mean(np.mean(np.abs(class_shap_vals), axis=2), axis=0)
    
    # 3. MNE Setup
    clean_names = [e.replace('FP', 'Fp').replace('Z', 'z').replace('z', 'z') for e in electrodes]
    info = mne.create_info(ch_names=clean_names, sfreq=250, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')
    
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # --- FIX: Remove 'names' and 'show_names' arguments ---
    im, _ = mne.viz.plot_topomap(
        avg_saliency, 
        info, 
        axes=ax, 
        show=False, 
        cmap='Reds',       
        contours=6,        
        extrapolate='head',
        sphere=None,
        sensors=True       # Shows black dots for sensors
    )
    
    # --- MANUAL LABELING (Works on all MNE versions) ---
    # We extract the 2D positions of the sensors from the plot's info
    # (mne.viz.plot_topomap automatically projects 3D->2D, so we grab that projection)
    
    # Get the layout to find positions
    layout = mne.channels.find_layout(info)
    pos = layout.pos[:, :2]  # (x, y) coordinates
    
    # Adjust positions slightly if needed (layout pos is normalized 0-1, topomap is -1 to 1)
    # However, a safer way to label without complex coordinate transforms is to just 
    # rely on the 'sensors=True' dots for clarity, or try this simple annotator:
    for name in clean_names:
        if name in layout.names:
            idx = layout.names.index(name)
            x, y = pos[idx]
            # Simple heuristic transform for layout (0..1) to topomap (-0.1..0.1 approx)
            # Note: MNE's internal projection is complex. 
            # If labels look wrong, it is safer to disable them or upgrade MNE.
            # For now, we will SKIP text labels to prevent further crashes/misalignment.
            pass 

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
    parser.add_argument('--checkpoint', type=str, default='/home/kay/FAST/best-checkpoint.ckpt')
    parser.add_argument('--data', type=str, default='Processed/BCIC2020Track3.h5')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--n_bg', type=int, default=125)
    parser.add_argument('--n_test', type=int, default=50)
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