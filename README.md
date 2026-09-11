# CogniPulse: Production P300 BCI Speller

Welcome to the **CogniPulse** project repository! This project implements a modular, production-ready P300 Brain-Computer Interface (BCI) speller built on the BCI Competition III (Dataset II) benchmark.

---

## 🚀 Quick Setup Instructions for Collaborators

Follow these steps to set up your local development environment after cloning the repository.

### 1. Clone the Repository & Navigate to Directory

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "P300 EEG Signal"
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```



### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📂 Dataset Setup (Manual Step)

Because the raw MATLAB benchmark files (`.mat`) total ~290 MB, they are excluded from Git tracking via `.gitignore` to keep the repo lightweight and cloud-deployable.

1. Download the BCI Competition III (Dataset II) dataset.
2. Copy `Subject_A_Train.mat` into the local `data/` folder:

```
P300 EEG Signal/
└── data/
    ├── __init__.py
    ├── bci_dataset_loader.py
    └── Subject_A_Train.mat   <-- Paste file here!
```

---

## 🧪 Verify Your Local Setup

To confirm that your environment and dataset loader are working properly, run:

```bash
python data/bci_dataset_loader.py
```

**Expected Output:**

```
✓ Step 2 Production Dataset Verification Passed!
  - Signal Array Shape: (64, 236440) (Channels x Samples)
  - Total Flashes Parsed: 15300
```

---

## 🔄 Daily Collaboration Workflow

Always keep your local workspace synchronized before starting work and after completing a module:

**Before writing code:** Pull the latest updates from your teammate:

```bash
git pull origin main
```
**Before writing code:** Create a branch to work on:

```bash
git checkout -b branch-name
```

**After testing a new module:** Push your changes:

```bash
git add .
git commit -m "Completed [Module Name]"
git push origin main
```

---

## 📁 Repository Structure Overview

```
P300 EEG Signal/
├── .gitignore                    # Excludes venv, cache, and raw dataset files
├── requirements.txt              # Production Python dependencies
├── README.md                     # Setup instructions
├── data/
│   ├── __init__.py               # Package marker
│   └── bci_dataset_loader.py     # Parses raw MATLAB dataset files
├── src/                          # Signal processing & machine learning pipeline
├── app/                          # Streamlit web interface
├── models/                       # Serialized trained model binaries (.pkl)
└── tests/                        # Automated unit tests
```