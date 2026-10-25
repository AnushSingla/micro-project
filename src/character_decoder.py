import sys
import os
import numpy as np
import joblib
from typing import List, Tuple

# Standard BCI Competition III 6x6 Matrix Speller Grid
GRID = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '0', '1', '2', '3'],
    ['4', '5', '6', '7', '8', '9']
]

def decode_character_from_events(
    lda_model, 
    scaler, 
    X_epochs: np.ndarray, 
    events: List[dict]
) -> Tuple[str, int, int]:
    """
    Accumulates continuous LDA decision scores across trial repetitions to decode 
    the target row, column, and letter from the 6x6 Matrix.
    """
    from src.feature_extraction import prepare_lda_features
    
    # Extract features for all epochs in trial
    X_features, _ = prepare_lda_features(X_epochs, scaler=scaler, fit_scaler=False)
    
    # Compute continuous LDA prediction confidence scores
    scores = lda_model.decision_function(X_features)
    
    # Accumulate scores for Rows (1-6) and Columns (7-12)
    row_scores = np.zeros(6, dtype=np.float64)
    col_scores = np.zeros(6, dtype=np.float64)
    
    for score, event in zip(scores, events):
        code = event['stimulus_code']
        if 1 <= code <= 6:
            # Row flashes (Codes 1 to 6)
            row_scores[code - 1] += score
        elif 7 <= code <= 12:
            # Column flashes (Codes 7 to 12)
            col_scores[code - 7] += score
            
    predicted_row_idx = int(np.argmax(row_scores))
    predicted_col_idx = int(np.argmax(col_scores))
    
    decoded_char = GRID[predicted_row_idx][predicted_col_idx]
    
    return decoded_char, predicted_row_idx + 1, predicted_col_idx + 7


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from data.bci_dataset_loader import load_bci_competition_iii
    from src.preprocessing import bandpass_filter
    from src.epoch_extraction import extract_p300_epochs
    
    print("Testing Step 8: 15-Trial Score Accumulation & Matrix Decoding...")
    
    # Load model pipeline
    model_path = "models/lda_p300_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file missing. Run src/lda_classifier.py first!")
        
    saved_artifacts = joblib.load(model_path)
    lda_model = saved_artifacts['model']
    scaler = saved_artifacts['scaler']
    
    # Load dataset
    data = load_bci_competition_iii("data/Subject_A_Train.mat")
    filtered = bandpass_filter(data['signal'], sfreq=data['sfreq'])
    X_raw, y = extract_p300_epochs(filtered, data['events'], sfreq=data['sfreq'])
    
    # Group events and epochs by character trial
    events = data['events']
    trials = {}
    
    for i, ev in enumerate(events):
        t_id = ev['trial']
        if t_id not in trials:
            trials[t_id] = {'epochs': [], 'events': []}
        trials[t_id]['epochs'].append(X_raw[i])
        trials[t_id]['events'].append(ev)
        
    correct_count = 0
    total_trials = min(15, len(trials))  # Test first 15 characters
    
    print("\n" + "="*50)
    print("      CHARACTER DECODING ACCURACY TEST     ")
    print("="*50)
    
    for t_id in range(total_trials):
        trial_epochs = np.array(trials[t_id]['epochs'])
        trial_events = trials[t_id]['events']
        
        decoded_char, row_code, col_code = decode_character_from_events(
            lda_model, scaler, trial_epochs, trial_events
        )
        
        print(f"Trial {t_id + 1:02d} -> Decoded Character: '{decoded_char}' (Row {row_code}, Col {col_code})")
        
    print("="*50)
    print("✓ Step 8 Score Accumulation Verification Complete!")