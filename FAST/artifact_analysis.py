import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

# Import your specific loaders
from BCIC2020Track3_train import load_standardized_h5
from BCIC2020Track3_preprocess import Electrodes  # Ensure this list matches your 64 channels

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = '/home/kay/FAST/FAST/Processed/BCIC2020Track3.h5'
SUBJECT_IDX = 11   # We are analyzing Subject 0 (the noisy one)
SFREQ = 250       # Your sampling frequency

# ==========================================
# 1. LOAD DATA & CONVERT TO MNE
# ==========================================
print(f"Loading data from {DATA_PATH}...")
X_all, Y_all = load_standardized_h5(DATA_PATH)

# Extract Subject 0 data: Shape (Trials, Channels, Time)
X_sub = X_all[SUBJECT_IDX]
print(f"Subject {SUBJECT_IDX} Data Shape: {X_sub.shape}")

# Create MNE Info Object
# We assume standard 10-20 montage based on your channel names
clean_names = [e.replace('FP', 'Fp').replace('Z', 'z').replace('z', 'z') for e in Electrodes]
info = mne.create_info(ch_names=clean_names, sfreq=SFREQ, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage, on_missing='ignore')

# Create MNE EpochsArray
# Note: X_sub is likely standardized (Z-scored). 
# MNE usually expects Volts (e.g., 1e-5). If your data is Z-scored (unitless ~0-5), 
# the absolute amplitude values in plots will be "arbitrary units", but patterns (PSD/ICA) remain valid.
epochs = mne.EpochsArray(X_sub, info)

# ==========================================
# 2. SPECTRAL ANALYSIS (The "Delta" Check)
# ==========================================
print("\n--- Plotting Power Spectral Density (PSD) ---")
# Capture the figure and save it
fig = epochs.compute_psd(fmin=0.1, fmax=40).plot(average=True, picks='eeg', exclude='bads', show=False)
plt.savefig(f"/home/kay/FAST/FAST/artifact_analysis/Subject{SUBJECT_IDX}_PSD_Analysis.png")
plt.close()
print("Saved: Subject0_PSD_Analysis.png")

# ==========================================
# 3. AMPLITUDE "EXPLOSION" CHECK
# ==========================================
# (This part only printed text, which you already saw. It confirms 100% artifacts!)
print("\n--- Checking for High-Amplitude Artifacts (Z-score > 6) ---")
# ... (your existing loop) ...

# ==========================================
# 4. ICA ANALYSIS
# ==========================================
print("\n--- Running ICA on Epochs ---")
ica = ICA(n_components=15, random_state=97, max_iter=800)
ica.fit(epochs)

print("Plotting ICA Components...")
# MNE plots usually output a figure, or we can save the current matplotlib figure
ica.plot_components(show=False)
plt.savefig(f"/home/kay/FAST/FAST/artifact_analysis/Subject{SUBJECT_IDX}_ICA_Components.png")
plt.close()
print("Saved: Subject0_ICA_Components.png")

# Visualize sources (Time Courses)
print("Plotting ICA Sources...")
ica.plot_sources(epochs, picks=range(5), title="ICA Component Time Courses", show=False)
plt.savefig(f"/home/kay/FAST/FAST/artifact_analysis/Subject{SUBJECT_IDX}_ICA_Sources.png")
plt.close()
print("Saved: Subject0_ICA_Sources.png")

print("\nAnalysis Complete. Please check the .png files in your current directory.")