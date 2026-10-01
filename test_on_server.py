"""
test_on_server.py
=================
Server-side CSI HAR inference.

Called as subprocess by collector.py:
  python test_on_server.py <rx1_csv_path> <rx2_csv_path>

Outputs ONE JSON line to stdout then exits.

JSON format:
  {"count":2,"activities":["standing","sitting"],
   "locations":[{"x":1.23,"y":0.87},{"x":3.12,"y":2.01}],
   "confidence":87.4,"windows":15}
"""

import sys
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
PROCESSED  = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# CONSTANTS  (must match train.py exactly)
# ============================================================

WINDOW_SIZE     = 30
STEP_SIZE       = 3
N_FEATURES      = 384
RX_FEATURES     = 192
MAX_PERSONS     = 4
N_ACTIVITIES    = 7
N_COUNT_CLASSES = 5
CNN_CHANNELS    = 128
NHEAD           = 8
TF_LAYERS       = 4
TF_FF           = 512
HEAD_HIDDEN     = 64
DROPOUT         = 0.1

ACTIVITY_NAMES = [
    "empty", "falling", "lying", "running",
    "sitting", "standing", "walking"
]


# ============================================================
# MODEL  (identical copy of TEDNet from train.py)
# ============================================================

class TEDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(N_FEATURES, CNN_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm1d(CNN_CHANNELS),
            nn.GELU(),
            nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm1d(CNN_CHANNELS),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, WINDOW_SIZE, CNN_CHANNELS) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = CNN_CHANNELS,
            nhead          = NHEAD,
            dim_feedforward = TF_FF,
            dropout        = DROPOUT,
            activation     = "gelu",
            batch_first    = True,
            norm_first     = False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=TF_LAYERS
        )
        self.pool_norm    = nn.LayerNorm(CNN_CHANNELS)
        self.head_activity = nn.Sequential(
            nn.Linear(CNN_CHANNELS, HEAD_HIDDEN),
            nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, N_ACTIVITIES),
        )
        self.head_count = nn.Sequential(
            nn.Linear(CNN_CHANNELS, HEAD_HIDDEN),
            nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, N_COUNT_CLASSES),
        )
        self.head_location = nn.Sequential(
            nn.Linear(CNN_CHANNELS, HEAD_HIDDEN),
            nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, MAX_PERSONS * 2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.pool_norm(x)
        logits_act = self.head_activity(x)
        logits_cnt = self.head_count(x)
        loc        = self.head_location(x).view(-1, MAX_PERSONS, 2)
        return logits_act, logits_cnt, loc


# ============================================================
# CSI SYNC + FEATURE EXTRACTION
# ============================================================

def load_and_sync(rx1_path, rx2_path):
    """
    Load RX1 and RX2 CSVs, synchronize by host timestamp,
    extract amplitude features.
    Returns combined float32 array (N, 384) or None on failure.
    """
    try:
        rx1 = pd.read_csv(rx1_path)
        rx2 = pd.read_csv(rx2_path)
    except Exception as e:
        return None, f"CSV read error: {e}"

    def clean(df):
        t = pd.to_datetime(df["host_timestamp_iso"], utc=True, errors="coerce")
        df = df.copy()
        df["_t"] = t
        df = df.dropna(subset=["_t"])
        df = df.sort_values("_t")
        df = df.drop_duplicates(subset=["_t"], keep="first")
        return df.reset_index(drop=True)

    rx1 = clean(rx1)
    rx2 = clean(rx2)

    if len(rx1) < 2 or len(rx2) < 2:
        return None, "Too few rows after cleaning"

    t1 = rx1["_t"].astype("int64").to_numpy(dtype=np.float64) / 1e9
    t2 = rx2["_t"].astype("int64").to_numpy(dtype=np.float64) / 1e9

    def extract_amp(df):
        cols = []
        for sc in range(RX_FEATURES):
            i_col = f"SC{sc}_I"
            q_col = f"SC{sc}_Q"
            if i_col not in df.columns or q_col not in df.columns:
                return None
            cols.extend([i_col, q_col])
        raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        if np.isnan(raw).any():
            raw = pd.DataFrame(raw).interpolate(
                axis=0, limit_direction="both"
            ).to_numpy(np.float32)
        iq = raw.reshape(len(df), RX_FEATURES, 2)
        return np.sqrt(iq[:, :, 0] ** 2 + iq[:, :, 1] ** 2)

    f1 = extract_amp(rx1)
    f2 = extract_amp(rx2)
    if f1 is None or f2 is None:
        return None, "Missing CSI columns"

    # Adaptive sync tolerance
    def mdt(t):
        d = np.diff(t)
        d = d[d > 0]
        return float(np.median(d)) if len(d) else 0.05

    max_delta = max(0.050, 0.60 * max(mdt(t1), mdt(t2), 0.001))

    indices     = np.searchsorted(t2, t1)
    left        = np.clip(indices - 1, 0, len(t2) - 1)
    right       = np.clip(indices,     0, len(t2) - 1)
    ld          = np.abs(t1 - t2[left])
    rd          = np.abs(t1 - t2[right])
    use_right   = rd < ld
    nearest_idx = np.where(use_right, right, left)
    nearest_delta = np.where(use_right, rd, ld)
    valid = nearest_delta <= max_delta

    if not np.any(valid):
        return None, "No synced pairs"

    rx1_idx = np.flatnonzero(valid)
    rx2_idx = nearest_idx[valid]

    best = {}
    for i1, i2, d in zip(rx1_idx, rx2_idx, nearest_delta[valid]):
        prev = best.get(int(i2))
        if prev is None or d < prev[0]:
            best[int(i2)] = (float(d), int(i1))

    pairs   = sorted([(i1, i2) for i2, (_, i1) in best.items()], key=lambda p: p[0])
    rx1_i   = np.array([p[0] for p in pairs])
    rx2_i   = np.array([p[1] for p in pairs])
    combined = np.concatenate([f1[rx1_i], f2[rx2_i]], axis=1)  # (N, 384)

    return combined, None


# ============================================================
# SLIDING WINDOWS
# ============================================================

def make_windows(combined):
    windows = []
    for start in range(0, len(combined) - WINDOW_SIZE + 1, STEP_SIZE):
        w = combined[start : start + WINDOW_SIZE]
        if np.isfinite(w).all():
            windows.append(w.astype(np.float32))
    return windows


# ============================================================
# INFERENCE
# ============================================================

def run_inference(model, windows, mean, std):
    X = np.stack(windows, axis=0)              # (W, 30, 384)
    X = ((X - mean) / std).astype(np.float32)
    X_t = torch.from_numpy(X)

    model.eval()
    with torch.no_grad():
        logits_act, logits_cnt, pred_loc = model(X_t)

    # Average sigmoid across windows → per-class confidence
    sig_act  = torch.sigmoid(logits_act).mean(dim=0).numpy()
    detected = sig_act > 0.5

    # Average count logits → argmax
    avg_cnt         = logits_cnt.mean(dim=0)
    predicted_count = int(avg_cnt.argmax().item())

    # Average location
    avg_loc = pred_loc.mean(dim=0).numpy()   # (4, 2)

    confidence = float(sig_act.max() * 100.0)

    return {
        "count":      predicted_count,
        "activities": [ACTIVITY_NAMES[i] for i in range(N_ACTIVITIES) if detected[i]],
        "locations":  [
            {
                "x": round(float(avg_loc[i, 0]), 3),
                "y": round(float(avg_loc[i, 1]), 3)
            }
            for i in range(predicted_count)
        ],
        "confidence": round(confidence, 1),
        "windows":    len(windows),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: test_on_server.py rx1.csv rx2.csv"}))
        sys.exit(1)

    rx1_path = sys.argv[1]
    rx2_path = sys.argv[2]

    # ── Load normalization ────────────────────────────────
    mean_path = PROCESSED / "feature_mean.npy"
    std_path  = PROCESSED / "feature_std.npy"

    if not mean_path.exists() or not std_path.exists():
        print(json.dumps({
            "error": "feature_mean/std missing. Run preprocess first.",
            "count": 0, "activities": [], "locations": [],
            "confidence": 0.0, "windows": 0
        }))
        sys.exit(0)

    mean = np.load(str(mean_path)).astype(np.float32)
    std  = np.load(str(std_path)).astype(np.float32)
    std[std < 1e-8] = 1.0

    # ── Load model ────────────────────────────────────────
    candidates = sorted(
        MODELS_DIR.glob("*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not candidates:
        print(json.dumps({
            "error": "No model found. Run train.py first.",
            "count": 0, "activities": [], "locations": [],
            "confidence": 0.0, "windows": 0
        }))
        sys.exit(0)

    ckpt = torch.load(
        str(candidates[0]), map_location="cpu", weights_only=False
    )
    state_dict = (
        ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt
    )

    model = TEDNet()
    model.load_state_dict(state_dict)

    # ── Sync and extract ──────────────────────────────────
    combined, err = load_and_sync(rx1_path, rx2_path)
    if combined is None or len(combined) < WINDOW_SIZE:
        print(json.dumps({
            "error": err or "Not enough synced frames",
            "count": 0, "activities": [], "locations": [],
            "confidence": 0.0, "windows": 0
        }))
        sys.exit(0)

    # ── Windows ───────────────────────────────────────────
    windows = make_windows(combined)
    if not windows:
        print(json.dumps({
            "error": "No valid windows",
            "count": 0, "activities": [], "locations": [],
            "confidence": 0.0, "windows": 0
        }))
        sys.exit(0)

    # ── Inference ─────────────────────────────────────────
    result = run_inference(model, windows, mean, std)

    # Single JSON line to stdout — collector.py reads this
    print(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()