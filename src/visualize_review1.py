import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt # type: ignore
from sklearn.decomposition import FastICA # type: ignore
from data.bci_dataset_loader import load_bci_competition_iii
from src.preprocessing import bandpass_filter
from src.ica_artifact_removal import apply_fastica_artifact_removal


# 1. Pipeline Execution & Data Extraction

print("Executing Pipeline Steps 2-4 for Visualization...")
file_path = "data/Subject_A_Train.mat"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Missing {file_path}. Please place the dataset in data/ directory.")

data = load_bci_competition_iii(file_path)
raw_signal = data['signal']
sfreq = data['sfreq']

# Run Bandpass Filter (Step 3)
filtered_signal = bandpass_filter(raw_signal, sfreq=sfreq)

# Run FastICA to get sources and cleaned signal (Step 4)
signal_T = filtered_signal.T
ica = FastICA(n_components=20, max_iter=500, random_state=42)
sources = ica.fit_transform(signal_T)

# Identify ocular component (highest variance)
variances = np.var(sources, axis=0)
blink_idx = np.argmax(variances)

# Reconstruct clean signal by zeroing out ocular component
sources_clean = sources.copy()
sources_clean[:, blink_idx] = 0.0
clean_signal = ica.inverse_transform(sources_clean).T

# Select 3-second sample window (720 samples) for clear visualization
samples = 720
time_axis = np.arange(samples) / sfreq


# 2. Multi-Panel Figure Generation

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("BCI P300 Pipeline: Step-by-Step Signal Cleaning Proof (Review 1)", fontsize=14, fontweight='bold')

# Figure 1: Raw vs. 0.5-20 Hz Bandpass Filtered EEG (Frontal Channel 0)
axes[0].plot(time_axis, raw_signal[0, :samples], label="Raw Signal (High Noise & Drift)", color="gray", alpha=0.6)
axes[0].plot(time_axis, filtered_signal[0, :samples], label="0.5–20.0 Hz Bandpass Filtered", color="blue", linewidth=1.5)
axes[0].set_title("Figure 1: Frequency Filtering (Removing Baseline Drift & High-Frequency Power Line Noise)", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Voltage (µV)")
axes[0].legend(loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.5)

# Figure 2: FastICA Independent Components Breakdown
axes[1].plot(time_axis, sources[:samples, blink_idx], label=f"Ocular Component #{blink_idx} (Isolated Eye Blink Spike)", color="red", linewidth=1.5)
# Plot a non-blink neural component for direct comparison
other_idx = (blink_idx + 1) % 20
axes[1].plot(time_axis, sources[:samples, other_idx], label=f"Neural Component #{other_idx} (Clean Brain Signals)", color="darkgreen", alpha=0.7)
axes[1].set_title("Figure 2: FastICA Decomposition (Isolating Ocular Noise into Independent Component)", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Amplitude (A.U.)")
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.5)

# Figure 3: Filtered vs. FastICA Reconstructed Signal
axes[2].plot(time_axis, filtered_signal[0, :samples], label="Before ICA (Contains Eye Blink Artifacts)", color="blue", alpha=0.5)
axes[2].plot(time_axis, clean_signal[0, :samples], label="After ICA (Eye Blink Artifacts Flattened Out)", color="green", linewidth=1.5)
axes[2].set_title("Figure 3: Artifact Removal Proof (Reconstructed EEG Signal Free of Ocular Spikes)", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Time (Seconds)")
axes[2].set_ylabel("Voltage (µV)")
axes[2].legend(loc="upper right")
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save high-resolution PNG for Review presentation slides
output_img = "review1_signal_cleaning_proof.png"
plt.savefig(output_img, dpi=300)
print(f"\n✓  Proof Plots Successfully Generated!")
print(f"  - Image saved to project root: {output_img}")

plt.show()