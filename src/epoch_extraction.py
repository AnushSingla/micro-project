import sys
import os
import numpy as np
from typing import Tuple, List, Dict, Union

def extract_p300_epochs(
    clean_signal: np.ndarray, 
    events: Union[List[Dict], np.ndarray], 
    sfreq: float = 240.0, 
    tmin: float = 0.0, 
    tmax: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trims auxiliary channels to standard 64 EEG channels and slices into 0-600ms epochs.
    """
    # Force retention of only true 64 EEG scalp channels
    if clean_signal.shape[0] > 64:
        clean_signal = clean_signal[:64, :]
        
    n_channels, total_samples = clean_signal.shape
    sample_window = int((tmax - tmin) * sfreq)  # 0.6 * 240 = 144 samples
    
    epochs = []
    labels = []
    
    for event in events:
        if isinstance(event, dict):
            onset_idx = event.get('sample_index', event.get('onset', event.get('sample_idx', event.get('sample'))))
            is_target = event.get('is_target', event.get('target', event.get('label')))
        else:
            onset_idx = event['onset'] if 'onset' in event.dtype.names else event[0]
            is_target = event['is_target'] if 'is_target' in event.dtype.names else event[1]

        onset_idx = int(onset_idx)
        end_idx = onset_idx + sample_window
        
        if end_idx <= total_samples:
            epoch = clean_signal[:, onset_idx:end_idx]
            epochs.append(epoch)
            labels.append(int(is_target))
            
    X = np.array(epochs, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    return X, y


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from data.bci_dataset_loader import load_bci_competition_iii
    from src.preprocessing import bandpass_filter
    from src.ica_artifact_removal import apply_fastica_artifact_removal
    
    print("Testing Step 5: Epoch Extraction Pipeline...")
    file_path = "data/Subject_A_Train.mat"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing dataset at {file_path}")
        
    data = load_bci_competition_iii(file_path)
    
    # Print first event structure for debugging if needed
    if len(data['events']) > 0:
        print(f"DEBUG - Sample Event Structure: {data['events'][0]}")
    
    # Run Steps 3 & 4
    filtered = bandpass_filter(data['signal'], sfreq=data['sfreq'])
    cleaned = apply_fastica_artifact_removal(filtered, n_components=20)
    
    # Run Step 5
    X, y = extract_p300_epochs(cleaned, data['events'], sfreq=data['sfreq'])
    
    print("\n Execution Successful!")
    print(f"  - Extracted Feature Tensor Shape (X): {X.shape}  [n_epochs, channels, samples]")
    print(f"  - Target Label Vector Shape (y):      {y.shape}")
    print(f"  - Class Balance -> Target P300 (1): {np.sum(y == 1)} | Non-Target (0): {np.sum(y == 0)}")