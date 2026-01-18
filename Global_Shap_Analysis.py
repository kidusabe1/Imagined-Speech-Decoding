import os
import argparse
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import mne
import scipy.signal
from transformers import PretrainedConfig
import gc # Garbage collection

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

TARGET_SUBJECTS = [10, 11] # Define which subjects to visualize

def load_model(checkpoint_path, config, device):
    """Loads FAST model weights and moves to GPU."""
    model = FAST(config)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False) 
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found.")
        return None
    
    model.eval()
    model.to(device)
    return model

def prepare_shap_data(data_path, fold_idx=0, n_bg=50, n_test=5):
    """Loads data for a specific fold (Subject)."""
    X_all, Y_all = load_standardized_h5(data_path)
    
    if fold_idx >= len(X_all):
        return None, None, None

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

def run_shap_analysis(model, X_background, X_test, device):
    """Performs SHAP analysis on GPU and returns Numpy arrays."""
    print(f"Initializing SHAP GradientExplainer on {device}...")
    
    X_background = X_background.to(device)
    X_test = X_test.to(device)

    eeg_explainer = shap.GradientExplainer(model, X_background)
    
    print(f"Computing SHAP values for {len(X_test)} samples...")
    shap_values = eeg_explainer.shap_values(X_test)

    # Clean up formatting
    if isinstance(shap_values, list):
        shap_values = [s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s for s in shap_values]
    elif isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.detach().cpu().numpy()

    shap_values = np.array(shap_values) 

    # Handle dimensions (Batch, Channels, Time, Classes) -> List[Classes]
    if shap_values.ndim == 4 and shap_values.shape[-1] == 5:
        shap_values = np.transpose(shap_values, (3, 0, 1, 2))
        shap_values = [shap_values[i] for i in range(shap_values.shape[0])]
    elif shap_values.ndim == 4 and shap_values.shape[1] == 5:
            shap_values = np.transpose(shap_values, (1, 0, 2, 3))
            shap_values = [shap_values[i] for i in range(shap_values.shape[0])]

    # Ensure (Batch, Channels, Time) orientation per class
    if isinstance(shap_values, list) and len(shap_values) > 0:
        s_shape = shap_values[0].shape
        if s_shape[0] != len(X_test) and s_shape[-1] == len(X_test):
            shap_values = [np.transpose(s, (2, 0, 1)) for s in shap_values]
    
    return shap_values

# --- NEW VISUALIZATION FUNCTIONS ---

def plot_frequency_band_heatmap(avg_shap, sfreq, title, output_path):
    """
    Computes Time-Frequency representation of the SHAP values to identify 
    which frequency bands are driving the model's decisions.
    """
    # 1. Collapse Channels: Get the global "Attention Signal" over time
    # We use ABS here because we want to know if the frequency was 'active' 
    # regardless of whether it pushed positive or negative.
    shap_signal = np.mean(np.abs(avg_shap), axis=0) # Shape: (Time,)
    
    # 2. Compute Spectrogram (Short-Time Fourier Transform)
    # nperseg controls the time/freq resolution trade-off
    f, t, Zxx = scipy.signal.stft(shap_signal, fs=sfreq, nperseg=64, noverlap=32)
    
    # Zxx is complex, we want Magnitude
    Sxx = np.abs(Zxx)
    
    # 3. Define Standard EEG Bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }
    
    # 4. Aggregate Frequencies into Bands
    n_bins = len(t)
    band_matrix = np.zeros((len(bands), n_bins))
    band_names = list(bands.keys())
    
    for i, (name, (low, high)) in enumerate(bands.items()):
        # Find indices of frequencies in this band
        idx = np.where((f >= low) & (f <= high))[0]
        if len(idx) > 0:
            # Average power in this band
            band_matrix[i, :] = np.mean(Sxx[idx, :], axis=0)
            
    # 5. Plot
    plt.figure(figsize=(10, 5))
    # Normalize for better visualization (0 to 1 scale per band or globally)
    # Using global normalization to preserve relative importance between bands
    plt.imshow(band_matrix, aspect='auto', cmap='inferno', origin='lower',
               extent=[t[0]*1000, t[-1]*1000, 0, len(bands)])
    
    plt.yticks(np.arange(len(bands)) + 0.5, band_names)
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency Band')
    plt.title(f"{title}\n(Frequency Importance)", fontsize=14)
    plt.colorbar(label='SHAP Spectral Magnitude')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  -> Saved Freq Band Heatmap: {output_path}")

def compute_zone_time_matrix(avg_shap_channels, electrodes, zone_dict):
    """
    Converts Channel-level SHAP (Channels x Time) to Zone-level SHAP (Zones x Time).
    """
    n_time = avg_shap_channels.shape[1]
    zone_names = list(zone_dict.keys())
    zone_matrix = np.zeros((len(zone_names), n_time))
    
    # Map electrode names to indices
    e_to_idx = {e: i for i, e in enumerate(electrodes)}
    
    for i, zone in enumerate(zone_names):
        # Find indices for this zone
        indices = [e_to_idx[e] for e in zone_dict[zone] if e in e_to_idx]
        
        if indices:
            # Extract rows for these channels and Average them
            # Result is (Time,)
            zone_signal = np.mean(avg_shap_channels[indices, :], axis=0)
            zone_matrix[i, :] = zone_signal
            
    return zone_matrix, zone_names

def plot_class_topomap(avg_shap, electrodes, title, output_path):
    """
    Plots the Topomap (averaged over time).
    """
    # Collapse Time: (Channels, Time) -> (Channels,)
    # We use raw mean to preserve direction (Red=Pos, Blue=Neg)
    topo_data = np.mean(avg_shap, axis=1)
    
    # MNE Setup
    clean_names = [e.replace('FP', 'Fp').replace('Z', 'z').replace('z', 'z') for e in electrodes]
    info = mne.create_info(ch_names=clean_names, sfreq=250, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')
    
    # Symmetric Scale
    limit = np.max(np.abs(topo_data))
    
    fig, ax = plt.subplots(figsize=(5, 5))
    im, _ = mne.viz.plot_topomap(
        topo_data, info, axes=ax, show=False, 
        cmap='RdBu_r', contours=6, vlim=(-limit, limit),
        extrapolate='head', sphere=None, sensors=True
    )
    
    plt.title(title, fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Mean SHAP (Red=Pos, Blue=Neg)')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved Topomap: {output_path}")

def plot_zone_time_heatmap(avg_shap, electrodes, zone_dict, title, output_path):
    """
    Plots the 2D Matrix: Y=Zone, X=Time.
    """
    # Convert to Zone Matrix (Zones x Time)
    zone_matrix, zone_names = compute_zone_time_matrix(avg_shap, electrodes, zone_dict)
    
    # Symmetric Scale
    limit = np.max(np.abs(zone_matrix))
    
    plt.figure(figsize=(10, 6))
    plt.imshow(zone_matrix, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
    
    plt.colorbar(label='Mean SHAP Importance')
    plt.title(title, fontsize=14)
    
    # X-Axis: Time
    plt.xlabel("Time (samples)", fontsize=12)
    # Optional: Add tick labels for seconds if you know sfreq (0, 250, 500...)
    
    # Y-Axis: Zones
    plt.yticks(range(len(zone_names)), zone_names, fontsize=10)
    plt.ylabel("Brain Region", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  -> Saved Zone Matrix: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/home/kay/FAST/FAST/Results_finetune_only/FAST/') 
    parser.add_argument('--data', type=str, default='/home/kay/FAST/FAST/Processed/BCIC2020Track3.h5')
    parser.add_argument('--n_bg', type=int, default=200)
    parser.add_argument('--n_test', type=int, default=100) # Number of samples to average per class
    parser.add_argument('--output_dir', type=str, default='shap_subject_analysis')
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found at {args.data}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- SUBJECT LOOP ---
    for subject_idx in TARGET_SUBJECTS:
        print(f"\n==========================================")
        print(f"PROCESSING SUBJECT {subject_idx}")
        print(f"==========================================")
        
        # 1. Load Model for this subject
        ckpt_path = os.path.join(args.model_dir, f"{subject_idx}_best.pth")
        model = load_model(ckpt_path, CONFIG, device)
        
        if model is None:
            print(f"Skipping Subject {subject_idx}: No model found.")
            continue
            
        # 2. Load Data for this subject
        # Note: We load more test samples to get a stable average
        X_bg, X_explain, Y_explain = prepare_shap_data(args.data, subject_idx, args.n_bg, args.n_test)
        
        if X_bg is None:
            print(f"Skipping Subject {subject_idx}: No data found.")
            continue
            
        # 3. Patch Electrodes if needed
        curr_electrodes = Electrodes.copy()
        actual_n_channels = X_explain.shape[1]
        if len(curr_electrodes) < actual_n_channels:
            for i in range(len(curr_electrodes), actual_n_channels):
                curr_electrodes.append(f"Ch{i}")

        # 4. Run SHAP
        # Returns List of 5 Arrays: [ (Batch, Channels, Time) ] per class
        shap_vals = run_shap_analysis(model, X_bg, X_explain, device)
        
        # 5. Visualize Each Class (CORRECTED)
        # Convert Y_test to numpy for easier indexing
        Y_test_np = Y_explain.cpu().numpy()

        for class_idx in range(len(shap_vals)):
            class_name = CLASSES[class_idx]
            
            # --- CRITICAL FIX: FILTER BY TRUE LABEL ---
            # Find indices where the TRUE label is the current class
            relevant_indices = np.where(Y_test_np == class_idx)[0]
            
            if len(relevant_indices) == 0:
                print(f"Skipping {class_name}: No samples of this class found in test set.")
                continue
                
            print(f"\n--- Visualizing Subject {subject_idx} | Class: {class_name} (N={len(relevant_indices)}) ---")
            
            # Get SHAP values ONLY for relevant samples
            # shap_vals[class_idx] has shape (Total_Samples, Channels, Time)
            relevant_shap = shap_vals[class_idx][relevant_indices]
            
            # Now average only the relevant samples
            avg_shap = np.mean(relevant_shap, axis=0)
            
            # --- PLOTTING (Same as before) ---
            
            # A. Topomap Plot
            topo_fname = f"Sub{subject_idx}_Class{class_idx}_{class_name}_Topomap.png"
            plot_class_topomap(
                avg_shap, 
                curr_electrodes, 
                title=f"Sub {subject_idx}: {class_name} (True Positives)", 
                output_path=os.path.join(args.output_dir, topo_fname)
            )
            
            # B. Zone-Time Matrix Plot
            matrix_fname = f"Sub{subject_idx}_Class{class_idx}_{class_name}_ZoneMatrix.png"
            plot_zone_time_heatmap(
                avg_shap, 
                curr_electrodes, 
                Zones, 
                title=f"Sub {subject_idx}: {class_name} (Time x Region)", 
                output_path=os.path.join(args.output_dir, matrix_fname)
            )

            # C. Frequency Band Heatmap
            freq_fname = f"Sub{subject_idx}_Class{class_idx}_{class_name}_FreqBands.png"
            plot_frequency_band_heatmap(
                avg_shap, sfreq,
                title=f"Sub {subject_idx}: {class_name}", 
                output_path=os.path.join(args.output_dir, freq_fname)
            )

        # Cleanup
        del model, shap_vals, X_bg, X_explain
        torch.cuda.empty_cache()
        gc.collect()

    print("\nAnalysis Complete!")

if __name__ == '__main__':
    main()