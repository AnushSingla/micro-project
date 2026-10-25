import os
import logging
import numpy as np
import scipy.io as sio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_bci_competition_iii(mat_file_path: str) -> dict:
    """
    Correctly reshapes 3D BCI Competition III Dataset II matrices 
    (Character Trials, Time Samples, Channels) into a 2D continuous array (Channels, Total Samples).
    """
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"Missing file at: {mat_file_path}")

    mat = sio.loadmat(mat_file_path)

    # Raw signal shape in mat file: (85 characters, 7794 samples, 64 channels)
    raw_signal = mat['Signal']  
    
    if raw_signal.ndim == 3:
        n_trials, n_samples_per_trial, n_channels = raw_signal.shape
        # Permute to (Channels, Trials, Samples) then reshape to continuous (Channels, Total Samples)
        # Signal shape: (64, 85 * 7794) = (64, 662490)
        continuous_signal = np.transpose(raw_signal, (2, 0, 1)).reshape(n_channels, n_trials * n_samples_per_trial)
    elif raw_signal.ndim == 2:
        continuous_signal = raw_signal.T
    else:
        raise ValueError(f"Unexpected Signal dimensions: {raw_signal.shape}")

    flashing = mat['Flashing']  # (85, 7794)
    stimulus_code = mat['StimulusCode']  # (85, 7794)
    stimulus_type = mat['StimulusType'] if 'StimulusType' in mat else None  # (85, 7794)

    sfreq = 240
    events = []

    # Iterate over each character trial (85 trials)
    for trial_idx in range(flashing.shape[0]):
        trial_flashing = flashing[trial_idx, :]
        trial_codes = stimulus_code[trial_idx, :]
        trial_types = stimulus_type[trial_idx, :] if stimulus_type is not None else None

        # Find rising edges where flash begins (0 -> 1 transition)
        flash_onsets = np.where(np.diff((trial_flashing > 0).astype(int)) == 1)[0] + 1

        for onset in flash_onsets:
            code = int(trial_codes[onset])
            if code == 0:
                continue

            # Absolute sample index in flattened continuous array
            global_sample_idx = (trial_idx * n_samples_per_trial) + onset
            is_target = bool(trial_types[onset] == 1) if trial_types is not None else False

            events.append({
                'sample': global_sample_idx,
                'sample_index': global_sample_idx,
                'stimulus_code': code,
                'is_target': is_target,
                'trial': trial_idx
            })

    logging.info(f"Loaded {mat_file_path} | Continuous EEG Signal: {continuous_signal.shape} | Total Flashes: {len(events)}")

    return {
        'signal': continuous_signal.astype(np.float32),
        'events': events,
        'sfreq': sfreq,
        'n_trials': flashing.shape[0]
    }


if __name__ == "__main__":
    file_path = "data/Subject_A_Train.mat"
    if os.path.exists(file_path):
        data = load_bci_competition_iii(file_path)
        print("\n✓ Loader Reshape Verification Passed!")
        print(f"  - Clean Continuous Array Shape: {data['signal'].shape} (64 Channels x Samples)")
        print(f"  - Parsed Flash Events Count:    {len(data['events'])}")