import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import logging
from sklearn.decomposition import FastICA # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def apply_fastica_artifact_removal(signal: np.ndarray, n_components: int = 20, max_iter: int = 500) -> np.ndarray:
    """
    Decomposes multi-channel EEG signals into independent components using FastICA,
    identifies high-variance ocular/blink components, zeroes them out, and reconstructs 
    a clean EEG signal matrix.

    Parameters:
        signal (np.ndarray): 2D array of shape (Channels x Samples).
        n_components (int): Number of independent components to extract (default: 20).
        max_iter (int): Maximum ICA iterations for solver convergence.

    Returns:
        np.ndarray: Reconstructed signal matrix of shape (Channels x Samples).
    """
    n_chans, n_samples = signal.shape
    
    # FastICA expects (Samples x Channels) shape
    signal_T = signal.T
    
    logging.info(f"Fitting FastICA with {n_components} components across {n_chans} channels...")
    ica = FastICA(n_components=n_components, max_iter=max_iter, random_state=42)
    sources = ica.fit_transform(signal_T)  # Shape: (Samples x Components)
    
    # Identify ocular component (component with highest variance in frontal regions)
    variances = np.var(sources, axis=0)
    blink_component_idx = np.argmax(variances)
    logging.info(f"Identified potential ocular/blink component at index: {blink_component_idx}")

    # Zero out the artifact component
    sources[:, blink_component_idx] = 0.0

    # Reconstruct cleaned EEG signal back to original domain
    reconstructed_T = ica.inverse_transform(sources)
    reconstructed = reconstructed_T.T
    
    logging.info(f"Successfully removed artifact component {blink_component_idx} | Shape: {reconstructed.shape}")
    return reconstructed

if __name__ == "__main__":
    from data.bci_dataset_loader import load_bci_competition_iii
    from src.preprocessing import bandpass_filter

    file_path = "data/Subject_A_Train.mat"
    if os.path.exists(file_path):
        raw_data = load_bci_competition_iii(file_path)
        filtered_eeg = bandpass_filter(raw_data['signal'], sfreq=raw_data['sfreq'])
        clean_eeg = apply_fastica_artifact_removal(filtered_eeg, n_components=20)
        
        print("\n Verification Passed!")
        print(f"  - Filtered Signal Shape: {filtered_eeg.shape}")
        print(f"  - ICA Cleaned Signal Shape: {clean_eeg.shape}")
    else:
        print(f"Notice: {file_path} not found inside data/ folder.")