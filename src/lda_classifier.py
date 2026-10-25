import sys
import os
import numpy as np
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from typing import Tuple, Dict

def train_regularized_lda(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[LinearDiscriminantAnalysis, Dict[str, float]]:
    """
    Trains a Regularized Linear Discriminant Analysis (LDA) classifier
    optimized for P300 confidence scoring.
    """
    # Stratified split to preserve 1:5 class balance in train/val sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standard Regularized LDA using Least Squares with automatic shrinkage
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    print(f"Training Regularized LDA on {X_train.shape[0]} samples ({X_train.shape[1]} features)...")
    model.fit(X_train, y_train)
    print("✓ Model fit complete!")
    
    # Compute continuous decision function scores (P300 ERP confidence)
    y_scores = model.decision_function(X_val)
    y_probs = model.predict_proba(X_val)[:, 1]
    
    auc_score = roc_auc_score(y_val, y_scores)
    
    # Adjust prediction threshold based on target class prior distribution (~1/6 = 0.166)
    threshold = np.percentile(y_scores, 83.3)  # Top 16.7% scored as target P300
    y_pred_adj = (y_scores >= threshold).astype(int)
    
    acc_score = np.mean(y_pred_adj == y_val)
    
    metrics = {
        'val_accuracy': float(acc_score),
        'val_roc_auc': float(auc_score)
    }
    
    print("\n" + "="*50)
    print("        LDA CLASSIFIER VALIDATION RESULTS        ")
    print("="*50)
    print(f"Adjusted Validation Accuracy: {acc_score * 100:.2f}%")
    print(f"Validation ROC-AUC Score:    {auc_score:.4f}")
    print("\nConfusion Matrix (at Top 16.7% P300 Threshold):")
    print(confusion_matrix(y_val, y_pred_adj))
    print("\nDetailed Classification Report:")
    print(classification_report(y_val, y_pred_adj, target_names=['Non-Target (0)', 'Target P300 (1)']))
    print("="*50)
    
    return model, metrics

def save_model(model: LinearDiscriminantAnalysis, scaler, filepath: str = "models/lda_p300_model.pkl"):
    """Saves trained model and scaler pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, filepath)
    print(f"\n✓ Saved model pipeline to '{filepath}'")

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from data.bci_dataset_loader import load_bci_competition_iii
    from src.preprocessing import bandpass_filter
    from src.epoch_extraction import extract_p300_epochs
    from src.feature_extraction import prepare_lda_features
    
    print("P300 LDA Classifier Execution...")
    file_path = "data/Subject_A_Train.mat"
    
    # Load and preprocess without continuous FastICA subtraction distorting phase
    data = load_bci_competition_iii(file_path)
    filtered = bandpass_filter(data['signal'], sfreq=data['sfreq'])
    X_raw, y = extract_p300_epochs(filtered, data['events'], sfreq=data['sfreq'])
    X_features, scaler = prepare_lda_features(X_raw, fit_scaler=True)
    
    # Train LDA
    lda_model, metrics = train_regularized_lda(X_features, y)
    
    # Save trained model artifacts
    save_model(lda_model, scaler, filepath="models/lda_p300_model.pkl")