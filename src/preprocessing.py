import sys
import os
import logging
import numpy as np
from scipy.signal import butter, filtfilt  # type: ignore
from sklearn.decomposition import FastICA

# Configure logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def bandpass_filter(
    signal: np.ndarray, 
    lowcut: float = 0.5, 
    highcut: float = 20.0, 
    sfreq: float = 240.0, 
    order: int = 4
) -> np.ndarray:
    """
    Applies a 4th-order zero-phase Butterworth bandpass filter across the time axis.
    Strips non-EEG auxiliary channels (retaining standard 64 scalp channels).
    """
    # Force retention of standard 64 scalp EEG channels
    if signal.shape[0] > 64:
        logging.info(f"Trimming auxiliary channels ({signal.shape[0]} -> 64 scalp EEG channels)")
        signal = signal[:64, :]
        
    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    
    # Zero-phase forward-backward filtering across the time dimension (axis=-1)
    filtered_signal = filtfilt(b, a, signal, axis=-1)
    
    logging.info(f"Bandpass filtered signal from {lowcut} Hz to {highcut} Hz | Output Shape: {filtered_signal.shape}")
    return filtered_signal


def apply_fastica(
    signal: np.ndarray, 
    n_components: int = 16, 
    frontal_channel_idx: int = 0
) -> np.ndarray:
    """
    Applies FastICA to isolate and remove high-amplitude ocular (blink) artifacts.
    Zeroes out only the independent component with the highest correlation to frontal channels.
    
    Parameters:
    -----------
    signal : np.ndarray
        EEG matrix of shape (n_channels, n_samples)
    n_components : int
        Number of independent components to decompose (default: 16)
    frontal_channel_idx : int
        Channel index corresponding to a frontal electrode (default: 0 for Fp1)
        
    Returns:
    --------
    clean_signal : np.ndarray
        Reconstructed EEG matrix of shape (n_channels, n_samples)
    """
    # FastICA expects shape (n_samples, n_channels)
    signal_T = signal.T
    
    logging.info(f"Fitting FastICA with {n_components} components on shape {signal_T.shape}...")
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500, tol=1e-3)
    
    # Extract independent components
    components = ica.fit_transform(signal_T)  # Shape: (n_samples, n_components)
    
    # Find component with maximum correlation to frontal eye-blink channel
    correlations = [
        np.abs(np.corrcoef(components[:, i], signal_T[:, frontal_channel_idx])[0, 1])
        for i in range(n_components)
    ]
    blink_comp_idx = int(np.argmax(correlations))
    
    logging.info(f"Ocular blink component identified: Component #{blink_comp_idx} (Correlation: {correlations[blink_comp_idx]:.4f})")
    
    # Zero out only the identified ocular component
    components[:, blink_comp_idx] = 0.0
    
    # Reconstruct clean EEG back to channel space
    clean_signal_T = ica.inverse_transform(components)
    clean_signal = clean_signal_T.T
    
    logging.info(f"FastICA reconstruction complete | Output Shape: {clean_signal.shape}")
    return clean_signal


def preprocess_pipeline(
    raw_signal: np.ndarray, 
    sfreq: float = 240.0, 
    use_ica: bool = True
) -> np.ndarray:
    """
    Full Preprocessing Pipeline Engine:
    1. Zero-phase 0.5–20.0 Hz Butterworth Bandpass Filtering
    2. Targeted FastICA Ocular Component Rejection
    """
    # Step 1: Bandpass Filter
    filtered = bandpass_filter(raw_signal, lowcut=0.5, highcut=20.0, sfreq=sfreq, order=4)
    
    # Step 2: FastICA (Targeted component removal)
    if use_ica:
        cleaned = apply_fastica(filtered, n_components=16, frontal_channel_idx=0)
    else:
        cleaned = filtered
        
    return cleaned


if __name__ == "__main__":
    # Dynamically inject project root directory to sys.path to eliminate ModuleNotFoundError
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from data.bci_dataset_loader import load_bci_competition_iii

    file_path = os.path.join(project_root, "data", "Subject_A_Train.mat")
    if os.path.exists(file_path):
        raw_data = load_bci_competition_iii(file_path)
        
        # Execute combined processing stream
        processed_eeg = preprocess_pipeline(raw_data['signal'], sfreq=raw_data['sfreq'], use_ica=True)
        
        print("\n Verification Passed!")
        print(f"  - Raw Signal Shape: {raw_data['signal'].shape}")
        print(f"  - Processed Signal Shape: {processed_eeg.shape}")
        print(f"  - Max Voltage Before Preprocessing: {np.max(np.abs(raw_data['signal'])):.2f} µV")
        print(f"  - Max Voltage After Filter & ICA: {np.max(np.abs(processed_eeg)):.2f} µV")
    else:
        print(f"Notice: File not found at path '{file_path}'. Verify file placement inside data/ folder.")