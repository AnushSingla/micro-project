import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.bci_dataset_loader import load_bci_competition_iii
from src.preprocessing import bandpass_filter
from src.epoch_extraction import extract_p300_epochs

data = load_bci_competition_iii("data/Subject_A_Train.mat")
filtered = bandpass_filter(data['signal'], sfreq=data['sfreq'])
X_raw, y = extract_p300_epochs(filtered, data['events'], sfreq=data['sfreq'])

# Calculate mean waveforms across all epochs for Target vs Non-Target
target_mean = np.mean(X_raw[y == 1], axis=0)      # Shape: (64, 144)
nontarget_mean = np.mean(X_raw[y == 0], axis=0)   # Shape: (64, 144)

# Calculate peak-to-peak amplitude difference
diff = target_mean - nontarget_mean
max_channel_idx = np.argmax(np.max(np.abs(diff), axis=1))

print("="*50)
print("           P300 ERP DIAGNOSTIC REPORT            ")
print("="*50)
print(f"Total Epochs Processed: {len(y)}")
print(f"Target Count (y=1):     {np.sum(y == 1)}")
print(f"Non-Target Count (y=0): {np.sum(y == 0)}")
print(f"Max Voltage Diff Channel Index: {max_channel_idx}")
print(f"Target Voltage Mean Range:     [{np.min(target_mean):.3f}, {np.max(target_mean):.3f}]")
print(f"Non-Target Voltage Mean Range: [{np.min(nontarget_mean):.3f}, {np.max(nontarget_mean):.3f}]")
print("="*50)