import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# Primary P300 parietal/occipital channel indices for 64-channel 10-20 system
# (e.g., Cz, Pz, Oz, PO7, PO8, P1, P2, CPz)
P300_CHANNEL_INDICES = [10, 11, 12, 13, 14, 28, 29, 30, 31, 32, 47, 48, 49, 50, 51]

def prepare_lda_features(
    X_epochs: np.ndarray, 
    scaler: StandardScaler = None, 
    fit_scaler: bool = True,
    downsample_factor: int = 3
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Extracts P300-relevant channels and time window (150ms-550ms post-stimulus),
    downsamples time resolution, and normalizes features.
    """
    # 1. Channel Filter: Retain primary parietal/occipital channels if available
    n_channels = X_epochs.shape[1]
    valid_channels = [c for c in P300_CHANNEL_INDICES if c < n_channels]
    if len(valid_channels) > 0:
        X_sub = X_epochs[:, valid_channels, :]
    else:
        X_sub = X_epochs
        
    # 2. Time Window Slicing: 150ms to 550ms post-stimulus (samples 36 to 132 at 240 Hz)
    X_window = X_sub[:, :, 36:132]
    
    # 3. Temporal Decimation / Downsampling
    X_downsampled = X_window[:, :, ::downsample_factor]
    
    # 4. Flatten into 2D feature matrix
    n_epochs, n_ch, n_samples = X_downsampled.shape
    X_flat = X_downsampled.reshape(n_epochs, n_ch * n_samples)
    
    # 5. Feature Scaling
    if fit_scaler:
        scaler = StandardScaler()
        X_flat = scaler.fit_transform(X_flat)
    else:
        if scaler is None:
            raise ValueError("Scaler instance must be provided when fit_scaler=False")
        X_flat = scaler.transform(X_flat)
        
    return X_flat, scaler


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from data.bci_dataset_loader import load_bci_competition_iii
    from src.preprocessing import bandpass_filter
    from src.ica_artifact_removal import apply_fastica_artifact_removal
    from src.epoch_extraction import extract_p300_epochs
    
    print("Testing Step 6: Downsampled Feature Extraction...")
    file_path = "data/Subject_A_Train.mat"
    
    data = load_bci_competition_iii(file_path)
    filtered = bandpass_filter(data['signal'], sfreq=data['sfreq'])
    cleaned = apply_fastica_artifact_removal(filtered, n_components=20)
    X_raw, y = extract_p300_epochs(cleaned, data['events'], sfreq=data['sfreq'])
    
    # Run Step 6 with decimation
    X_features, scaler = prepare_lda_features(X_raw, fit_scaler=True, downsample_factor=6)
    
    print("\n Feature Reduction Complete!")
    print(f"  - Original 3D Shape:     {X_raw.shape}")
    print(f"  - Reduced Feature Matrix: {X_features.shape} [n_epochs, n_features]")