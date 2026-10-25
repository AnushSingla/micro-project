"""Run once on your own computer:  python make_demo_cache.py
Creates data/demo_cache.npz (well under 5 MB) so the deployed app does not need
the 178 MB recording or the model. Commit that file to GitHub.
"""
import os
import numpy as np
import joblib

from data.bci_dataset_loader import load_bci_competition_iii
from src.preprocessing import bandpass_filter
from src.epoch_extraction import extract_p300_epochs
from src.feature_extraction import prepare_lda_features

DATA_FILE = "data/Subject_A_Train.mat"
MODEL_FILE = "models/lda_p300_model.pkl"
OUT_FILE = "data/demo_cache.npz"
N_PREVIEW = 1200

data = load_bci_competition_iii(DATA_FILE)
sfreq = data["sfreq"]
events = data["events"]

filtered = bandpass_filter(data["signal"], sfreq=sfreq)
X_raw, _ = extract_p300_epochs(filtered, events, sfreq=sfreq)
assert len(X_raw) == len(events), "epochs and events must line up one-to-one"

art = joblib.load(MODEL_FILE)
features, _ = prepare_lda_features(X_raw, scaler=art["scaler"], fit_scaler=False)
scores = art["model"].decision_function(features).astype(np.float32)

trial = np.array([e["trial"] for e in events], dtype=np.int32)
code = np.array([e["stimulus_code"] for e in events], dtype=np.int16)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
np.savez_compressed(
    OUT_FILE,
    scores=scores,
    trial=trial,
    code=code,
    sfreq=np.float32(sfreq),
    signal_preview=np.asarray(data["signal"][:, :N_PREVIEW], dtype=np.float32),
    filtered_preview=np.asarray(filtered[:, :N_PREVIEW], dtype=np.float32),
)
print(f"Saved {OUT_FILE}: {os.path.getsize(OUT_FILE) / 1e6:.2f} MB, {len(scores)} flashes")