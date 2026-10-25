import os
import sys
import html as html_lib
import numpy as np
import pandas as pd
import streamlit as st
import joblib

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:  # falls back to Streamlit's built-in charts
    HAS_PLOTLY = False

# Add src to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from data.bci_dataset_loader import load_bci_competition_iii
from src.preprocessing import bandpass_filter
from src.epoch_extraction import extract_p300_epochs
from src.feature_extraction import prepare_lda_features

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="P300 Speller",
    page_icon="🧠",
    layout="wide",
)

# BCI Competition III (Dataset II) speller matrix
GRID = [
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ1234"),
    list("56789_"),
]

SIGNAL = "#5eead4"   # decoded / winning element
MUTED = "#8b95a9"
LINE = "#1f2736"
BAR = "#2c3650"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

.stApp, .stApp button, .stApp input, .stApp textarea {
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: #0b0e14; color: #e7ebf3; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1240px; }

h1 { font-weight: 800; letter-spacing: -0.03em; font-size: 2.4rem !important; margin-bottom: 0.2rem; }
h2, h3 { font-weight: 700; letter-spacing: -0.01em; }
.hero p { color: #8b95a9; max-width: 62ch; margin: 0 0 1.4rem; line-height: 1.55; }

[data-testid="stSidebar"] { background: #0e121b; border-right: 1px solid #1c2433; }

[data-testid="stVerticalBlockBorderWrapper"] {
  background: #121722;
  border: 1px solid #1f2736 !important;
  border-radius: 16px;
  padding: 8px 12px;
}

[data-testid="stMetric"] {
  background: #0e121b; border: 1px solid #1c2433; border-radius: 12px; padding: 12px 14px;
}
[data-testid="stMetricLabel"] p { color: #8b95a9; }
[data-testid="stMetricValue"] { font-weight: 700; }

.stButton > button {
  border-radius: 10px; border: 1px solid #2a3347; background: #161c2a;
  color: #e7ebf3; font-weight: 600;
}
.stButton > button:hover { border-color: #5eead4; color: #5eead4; }
.stButton > button:focus-visible { outline: 2px solid #5eead4; outline-offset: 2px; }
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
  background: #5eead4; border-color: #5eead4; color: #04201b;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover { background: #7ff0de; color: #04201b; }

[data-baseweb="tab-highlight"] { background: #5eead4 !important; }

.speller { border-collapse: separate; border-spacing: 7px; width: 100%; table-layout: fixed; }
.speller td {
  height: 58px; padding: 0; text-align: center; font-size: 24px; font-weight: 600;
  border-radius: 10px; background: #0e121b; color: #6c7690; border: 1px solid #1c2433;
}
.speller td.line { background: #1a2234; color: #e7ebf3; border-color: #2a3550; }
.speller td.hit {
  background: #5eead4; color: #04201b; border-color: #5eead4;
  box-shadow: 0 0 0 4px rgba(94, 234, 212, 0.18);
}

.readout { display: flex; align-items: baseline; gap: 18px; margin: 4px 0 10px; }
.readout .char { font-size: 5.5rem; font-weight: 800; line-height: 1; color: #5eead4; }
.readout .where { color: #8b95a9; font-size: 1rem; }

.message {
  font-size: 1.6rem; font-weight: 600; letter-spacing: 0.04em; min-height: 2.2rem;
  word-break: break-all; color: #e7ebf3; border-top: 1px solid #1f2736;
  padding-top: 10px; margin-top: 12px;
}
.message.empty { color: #5b6479; font-size: 1rem; font-weight: 500; letter-spacing: 0; }

@media (max-width: 700px) {
  .speller td { height: 44px; font-size: 18px; }
  .readout .char { font-size: 4rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------ pipeline loading
@st.cache_resource
def load_pipeline_components():
    model_path = "models/lda_p300_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file missing. Run 'python src/lda_classifier.py' first, then reload this page.")
        st.stop()
    artifacts = joblib.load(model_path)
    return artifacts['model'], artifacts['scaler']


@st.cache_data
def load_and_preprocess_dataset():
    data_path = DATA_FILE
    data = load_bci_competition_iii(data_path)
    filtered_signal = bandpass_filter(data['signal'], sfreq=data['sfreq'])
    X_raw, y = extract_p300_epochs(filtered_signal, data['events'], sfreq=data['sfreq'])
    return data, filtered_signal, X_raw, y



# ---------------------------------------------------- dataset download (Drive)
DATA_FILE = "data/Subject_A_Train.mat"

GDRIVE_FILE_ID_FALLBACK = "1bjJT5qaF9xnAD_fLn7h3OZrUWJ7e3qPk"


@st.cache_resource(show_spinner="Downloading dataset from Google Drive (first run only)...")
def ensure_dataset():
    """Download the large .mat file once if it is not already on disk."""
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 5_000_000:
        return DATA_FILE

    try:
        file_id = st.secrets["GDRIVE_FILE_ID"]
    except Exception:
        file_id = GDRIVE_FILE_ID_FALLBACK
    if file_id == "PASTE_YOUR_FILE_ID_HERE":
        st.error("Dataset ID missing. Add GDRIVE_FILE_ID to Streamlit secrets or set it in app.py.")
        st.stop()

    import gdown
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={file_id}", DATA_FILE, quiet=False)

    # A tiny file usually means Drive returned an HTML page (link not shared publicly).
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) < 5_000_000:
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        st.error("Dataset download failed. In Google Drive, set the file to 'Anyone with the link' and check the file ID.")
        st.stop()
    return DATA_FILE

# -------------------------------------------------------------------- helpers
def style_fig(fig, height, legend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="Plus Jakarta Sans, sans-serif", size=12),
        showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


def score_bars(labels, values, winner):
    if HAS_PLOTLY:
        colors = [SIGNAL if i == winner else BAR for i in range(len(values))]
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
        st.plotly_chart(style_fig(fig, 210), use_container_width=True, config={"displayModeBar": False})
    else:
        st.bar_chart(pd.DataFrame({"Score": values}, index=labels))


def history_chart(hist, labels, winner):
    reps = list(range(1, hist.shape[0] + 1))
    if HAS_PLOTLY:
        fig = go.Figure()
        for j in range(hist.shape[1]):
            is_win = j == winner
            fig.add_trace(go.Scatter(
                x=reps, y=hist[:, j], mode="lines", name=labels[j],
                line=dict(color=SIGNAL if is_win else "#3a4560", width=3 if is_win else 1.5),
            ))
        fig.update_xaxes(title_text="Repetitions combined", dtick=1 if len(reps) <= 15 else None)
        st.plotly_chart(style_fig(fig, 240), use_container_width=True, config={"displayModeBar": False})
    else:
        st.line_chart(pd.DataFrame(hist, index=reps, columns=labels))


def speller_html(pred_row, pred_col):
    body = ""
    for r in range(6):
        cells = ""
        for c in range(6):
            if r == pred_row and c == pred_col:
                cls = "hit"
            elif r == pred_row or c == pred_col:
                cls = "line"
            else:
                cls = ""
            cells += f"<td class='{cls}'>{GRID[r][c]}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table class='speller'>{body}</table>"


def lead(values):
    """Gap between the best and second-best score (larger = more certain)."""
    s = np.sort(values)
    return float(s[-1] - s[-2])


def add_to_message(char):
    st.session_state.message += char


def clear_message():
    st.session_state.message = ""


st.session_state.setdefault("message", "")

# --------------------------------------------------------------------- header
st.markdown(
    "<div class='hero'><h1>P300 speller</h1>"
    "<p>Pick a character, choose how many repetitions to combine, and see how the "
    "classifier's scores point to one row and one column of the matrix.</p></div>",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
trial_idx = st.sidebar.number_input(
    "Character to decode (1–85)", min_value=1, max_value=85, value=1, step=1
) - 1
n_trials_to_accumulate = st.sidebar.slider(
    "Repetitions to combine", min_value=1, max_value=15, value=15,
    help="Each repetition flashes all 6 rows and 6 columns once. More repetitions give a more reliable result.",
)

# ---------------------------------------------------------- load pipeline/data
lda_model, scaler = load_pipeline_components()
ensure_dataset()
with st.spinner("Processing continuous EEG data..."):
    dataset, filtered_signal, X_raw, y = load_and_preprocess_dataset()

# Filter events and epochs for the selected trial
events = dataset['events']
trial_epochs = []
trial_events = []

for i, ev in enumerate(events):
    if ev['trial'] == trial_idx:
        trial_epochs.append(X_raw[i])
        trial_events.append(ev)

if not trial_epochs:
    st.warning("No flashes found for this character. Choose another character in the sidebar.")
    st.stop()

trial_epochs = np.array(trial_epochs)

# Limit to the selected repetition count (each repetition = 12 flashes)
max_flashes = n_trials_to_accumulate * 12
active_epochs = trial_epochs[:max_flashes]
active_events = trial_events[:max_flashes]

# Features and LDA scores
X_features, _ = prepare_lda_features(active_epochs, scaler=scaler, fit_scaler=False)
scores = lda_model.decision_function(X_features)

# Accumulate scores repetition by repetition
n_reps = max(1, int(np.ceil(len(scores) / 12)))
inc_rows = np.zeros((n_reps, 6), dtype=np.float64)
inc_cols = np.zeros((n_reps, 6), dtype=np.float64)

for i, (score, event) in enumerate(zip(scores, active_events)):
    rep = i // 12
    code = event['stimulus_code']
    if 1 <= code <= 6:
        inc_rows[rep, code - 1] += score   # codes 1-6 treated as rows (same as your original code)
    elif 7 <= code <= 12:
        inc_cols[rep, code - 7] += score   # codes 7-12 treated as columns

rows_hist = np.cumsum(inc_rows, axis=0)
cols_hist = np.cumsum(inc_cols, axis=0)
row_scores = rows_hist[-1]
col_scores = cols_hist[-1]

pred_row = int(np.argmax(row_scores))
pred_col = int(np.argmax(col_scores))
decoded_char = GRID[pred_row][pred_col]

# ------------------------------------------------------------------ main area
col_matrix, col_result = st.columns([5, 4], gap="large")

with col_matrix:
    with st.container(border=True):
        st.subheader("Speller matrix")
        st.caption("The brighter row and column scored highest. The glowing cell is the decoded character.")
        st.markdown(speller_html(pred_row, pred_col), unsafe_allow_html=True)

with col_result:
    with st.container(border=True):
        st.subheader("Decoded character")
        st.markdown(
            f"<div class='readout'><span class='char'>{html_lib.escape(decoded_char)}</span>"
            f"<span class='where'>Row {pred_row + 1}, column {pred_col + 1}</span></div>",
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns(2)
        b1.button("Add to message", type="primary", on_click=add_to_message,
                  args=(decoded_char,), use_container_width=True)
        b2.button("Clear message", on_click=clear_message, use_container_width=True)

        if st.session_state.message:
            st.markdown(
                f"<div class='message'>{html_lib.escape(st.session_state.message)}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='message empty'>Your message appears here after you add a character.</div>",
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Repetitions", n_reps)
        m2.metric("Row lead", f"{lead(row_scores):.2f}")
        m3.metric("Column lead", f"{lead(col_scores):.2f}")
        st.caption("Lead is the gap between the best and second-best score. A larger gap means a more certain result.")

with st.container(border=True):
    tab_scores, tab_build, tab_wave = st.tabs(["Row and column scores", "How scores build up", "EEG waveform"])

    row_labels = [f"Row {i + 1}" for i in range(6)]
    col_labels = [f"Col {i + 1}" for i in range(6)]

    with tab_scores:
        a, b = st.columns(2)
        with a:
            st.write("**Row scores**")
            score_bars(row_labels, row_scores, pred_row)
        with b:
            st.write("**Column scores**")
            score_bars(col_labels, col_scores, pred_col)

    with tab_build:
        st.caption("Each line is the running total for one row or column. The winner pulls away as repetitions accumulate.")
        a, b = st.columns(2)
        with a:
            st.write("**Rows**")
            history_chart(rows_hist, row_labels, pred_row)
        with b:
            st.write("**Columns**")
            history_chart(cols_hist, col_labels, pred_col)

    with tab_wave:
        sample_channel = st.selectbox("EEG channel", options=list(range(64)), index=10)
        n_samples = 1200
        sfreq = dataset['sfreq']
        t = np.arange(n_samples) / sfreq
        raw = dataset['signal'][sample_channel, :n_samples]
        filt = filtered_signal[sample_channel, :n_samples]

        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=raw, mode="lines", name="Raw (µV)",
                                     line=dict(color="#566178", width=1)))
            fig.add_trace(go.Scatter(x=t, y=filt, mode="lines", name="Filtered 0.5–20 Hz (µV)",
                                     line=dict(color=SIGNAL, width=1.6)))
            fig.update_xaxes(title_text="Time (s)")
            st.plotly_chart(style_fig(fig, 300, legend=True), use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.line_chart(pd.DataFrame({
                'Raw EEG (uV)': raw,
                'Filtered EEG 0.5-20Hz (uV)': filt,
            }, index=t))