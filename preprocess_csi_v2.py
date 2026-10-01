"""
preprocess_csi_v2.py
====================
Wi-Fi CSI Multi-Person HAR — Preprocessing Pipeline

════════════════════════════════════════════════════════
INPUT
════════════════════════════════════════════════════════

dataset/
    session_XXXX/
        metadata.json   ← multi-person format (new)
        rx1.csv         ← raw CSI from RX1
        rx2.csv         ← raw CSI from RX2

════════════════════════════════════════════════════════
WHAT THIS SCRIPT DOES
════════════════════════════════════════════════════════

1.  Reads every session directory.
2.  Parses metadata to get persons list
    (activity, x, y) for each person in the scene.
3.  Builds three label arrays per window:
      Y_activity  (multi-hot, 7 classes)
      Y_count     (integer 0-4)
      Y_location  (padded float pairs, shape [4, 2])
4.  Synchronises RX1 and RX2 CSI streams by timestamp.
5.  Extracts amplitude = sqrt(I^2 + Q^2) per subcarrier.
6.  Slides a window of size 30 with step 3 over
    the synchronised 384-feature stream.
7.  Splits sessions into train / val  (session-level,
    NOT random window shuffle — no data leakage).
8.  Computes feature mean and std from TRAINING sessions
    only and applies them to both splits.
9.  Saves everything to processed/.

════════════════════════════════════════════════════════
OUTPUT  (processed/)
════════════════════════════════════════════════════════

X_train.npy             (N_train, 30, 384)  float32
Y_activity_train.npy    (N_train, 7)         float32  multi-hot
Y_count_train.npy       (N_train,)           int64
Y_location_train.npy    (N_train, 4, 2)     float32  -1 = absent

X_val.npy               (N_val, 30, 384)    float32
Y_activity_val.npy      (N_val, 7)           float32
Y_count_val.npy         (N_val,)             int64
Y_location_val.npy      (N_val, 4, 2)       float32

feature_mean.npy        (1, 1, 384)          float32
feature_std.npy         (1, 1, 384)          float32
label_map.json          {"empty":0, ...}
session_summary.json    per-session stats

════════════════════════════════════════════════════════
ACTIVITY CLASS ORDER  (fixed — must match firmware)
════════════════════════════════════════════════════════

  0  empty
  1  falling
  2  lying
  3  running
  4  sitting
  5  standing
  6  walking

════════════════════════════════════════════════════════
BUGS FIXED vs ORIGINAL
════════════════════════════════════════════════════════

  Bug 2 — Normalisation computed over ALL data before
           the train/val split (validation contamination).
           Fixed: mean/std computed from training only.

  Bug 3 — Train/val split was random across windows.
           Overlapping windows from the same session
           appeared in both splits → inflated accuracy.
           Fixed: whole sessions assigned to one split.

Usage:
    python preprocess_csi_v2.py
    python preprocess_csi_v2.py --dataset dataset --output processed
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE      = 30
STEP_SIZE        = 3
FEATURE_MODE     = "amplitude"   # "amplitude" | "iq"
MAX_PERSONS      = 4
VALIDATION_RATIO = 0.20
SEED             = 42

# Fixed order — index = class ID used in model and firmware
ACTIVITY_CLASSES = [
    "empty",
    "falling",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]

ACTIVITY_TO_IDX = {a: i for i, a in enumerate(ACTIVITY_CLASSES)}

# Sentinel value marking an absent person slot in Y_location
LOCATION_SENTINEL = -1.0


# ============================================================
# METADATA PARSING
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_persons(metadata_path):
    """
    Return a list of person dicts from metadata.json.

    New multi-person format:
        {
            "num_persons": 2,
            "persons": [
                {"activity": "standing", "x": 1.5, "y": 1.0},
                {"activity": "sitting",  "x": 3.0, "y": 2.0}
            ]
        }

    Old single-person format (backward compatible):
        {
            "activity": "standing",
            "x": "1.5",
            "y": "1.0"
        }

    Empty room is represented as an empty list [].
    """
    meta = load_json(metadata_path)

    # ── New multi-person format ──────────────────────────
    if "persons" in meta and isinstance(meta["persons"], list):
        persons = []
        for p in meta["persons"]:
            act = str(p.get("activity", "empty")).lower().strip()
            try:
                x = float(p.get("x", 0.0))
            except (ValueError, TypeError):
                x = 0.0
            try:
                y = float(p.get("y", 0.0))
            except (ValueError, TypeError):
                y = 0.0
            persons.append({"activity": act, "x": x, "y": y})
        return persons

    # ── Old single-person format (backward compatible) ───
    activity = str(meta.get("activity", "empty")).lower().strip()

    if activity == "empty":
        return []   # empty room

    try:
        x = float(meta.get("x", 0.0))
    except (ValueError, TypeError):
        x = 0.0
    try:
        y = float(meta.get("y", 0.0))
    except (ValueError, TypeError):
        y = 0.0

    return [{"activity": activity, "x": x, "y": y}]


def build_labels(persons):
    """
    Convert a persons list into three label arrays.

    Returns
    -------
    activity_vec : float32 (7,)
        Multi-hot vector.  activity_vec[i] = 1.0 if activity i
        is present in the scene.

    count : int
        Number of persons (0 … MAX_PERSONS).

    location : float32 (MAX_PERSONS, 2)
        location[i] = (x, y) of person i.
        Absent slots are filled with LOCATION_SENTINEL (-1.0).
    """
    activity_vec = np.zeros(len(ACTIVITY_CLASSES), dtype=np.float32)
    location     = np.full((MAX_PERSONS, 2), LOCATION_SENTINEL,
                           dtype=np.float32)
    count        = min(len(persons), MAX_PERSONS)

    for i, p in enumerate(persons[:MAX_PERSONS]):
        act = p["activity"]
        if act in ACTIVITY_TO_IDX:
            activity_vec[ACTIVITY_TO_IDX[act]] = 1.0
        else:
            print(f"    [WARN] Unknown activity '{act}' — skipped.")
        location[i, 0] = p["x"]
        location[i, 1] = p["y"]

    return activity_vec, count, location


# ============================================================
# CSI FEATURE EXTRACTION
# ============================================================

def get_csi_columns(df):
    """
    Build ordered list of SC0_I, SC0_Q, SC1_I, SC1_Q, ...
    Raises ValueError if any expected column is missing.
    """
    cols = []
    for sc in range(192):
        i_col = f"SC{sc}_I"
        q_col = f"SC{sc}_Q"
        if i_col not in df.columns or q_col not in df.columns:
            raise ValueError(
                f"Missing CSI columns: {i_col} / {q_col}. "
                f"Check that collector.py wrote 192 subcarriers."
            )
        cols.extend([i_col, q_col])
    return cols


def extract_features(df):
    """
    Convert the raw I/Q dataframe rows into a feature matrix.

    FEATURE_MODE = "amplitude"  →  shape (N, 192)
        amp_i = sqrt(I_i^2 + Q_i^2)

    FEATURE_MODE = "iq"         →  shape (N, 384)
        raw interleaved I/Q values
    """
    csi_cols = get_csi_columns(df)
    raw = (
        df[csi_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    # Interpolate NaN values (bad packets)
    if np.isnan(raw).any():
        raw = (
            pd.DataFrame(raw)
            .interpolate(axis=0, limit_direction="both")
            .to_numpy(dtype=np.float32)
        )

    iq = raw.reshape(len(df), 192, 2)

    if FEATURE_MODE == "amplitude":
        amp = np.sqrt(iq[:, :, 0] ** 2 + iq[:, :, 1] ** 2)
        return amp.astype(np.float32)          # (N, 192)

    if FEATURE_MODE == "iq":
        return raw.astype(np.float32)          # (N, 384)

    raise ValueError(f"Unknown FEATURE_MODE={FEATURE_MODE!r}")


# ============================================================
# TIMESTAMP HANDLING
# ============================================================

def sort_and_clean(df):
    """
    Parse host_timestamp_iso, drop NaN rows, sort ascending,
    remove duplicates.  Returns a clean, index-reset DataFrame.
    """
    t = pd.to_datetime(df["host_timestamp_iso"], utc=True,
                       errors="coerce")
    df = df.copy()
    df["_t"] = t
    df = df.dropna(subset=["_t"])
    df = df.sort_values("_t")
    df = df.drop_duplicates(subset=["_t"], keep="first")
    return df.reset_index(drop=True)


def to_unix_seconds(df):
    """
    Convert host_timestamp_iso column to float64 Unix seconds.
    """
    t = pd.to_datetime(
        df["host_timestamp_iso"], utc=True, errors="coerce"
    )
    if t.isna().any():
        raise ValueError(
            "Invalid host_timestamp_iso values found. "
            "Check that collector.py is writing timestamps correctly."
        )
    return t.astype("int64").to_numpy(dtype=np.float64) / 1e9


# ============================================================
# RX1 / RX2 SYNCHRONISATION
# ============================================================

def synchronize(rx1_df, rx2_df):
    """
    Align RX1 and RX2 CSI streams by host timestamp.

    For each RX1 frame we find the nearest RX2 frame.
    If their timestamps differ by more than max_delta we
    discard the pair.  Among all RX1 frames that map to
    the same RX2 frame we keep only the closest one.

    Returns
    -------
    sync_time  : float64 (M,)  Unix seconds of each pair
    combined   : float32 (M, 384)
        First 192 columns = RX1 amplitude
        Last  192 columns = RX2 amplitude
    """
    rx1_df = sort_and_clean(rx1_df)
    rx2_df = sort_and_clean(rx2_df)

    f1 = extract_features(rx1_df)    # (N1, 192)
    f2 = extract_features(rx2_df)    # (N2, 192)
    t1 = to_unix_seconds(rx1_df)     # (N1,)
    t2 = to_unix_seconds(rx2_df)     # (N2,)

    if len(t1) == 0 or len(t2) == 0:
        raise ValueError("RX1 or RX2 has zero valid frames.")

    # ── Adaptive sync tolerance ──────────────────────────
    def median_interval(t):
        if len(t) < 2:
            return 0.0
        d = np.diff(t)
        d = d[d > 0.0]
        return float(np.median(d)) if len(d) else 0.0

    dt1       = median_interval(t1)
    dt2       = median_interval(t2)
    max_delta = max(0.050, 0.60 * max(dt1, dt2, 0.001))

    # ── Nearest-neighbour match: for each t1 find closest t2 ─
    indices     = np.searchsorted(t2, t1)
    left        = np.clip(indices - 1, 0, len(t2) - 1)
    right       = np.clip(indices,     0, len(t2) - 1)
    left_delta  = np.abs(t1 - t2[left])
    right_delta = np.abs(t1 - t2[right])
    use_right   = right_delta < left_delta
    nearest_idx = np.where(use_right, right, left)
    nearest_delta = np.where(use_right, right_delta, left_delta)

    valid = nearest_delta <= max_delta
    if not np.any(valid):
        raise ValueError(
            f"No synchronised RX1/RX2 pairs found. "
            f"dt1={dt1*1000:.1f} ms  dt2={dt2*1000:.1f} ms  "
            f"max_delta={max_delta*1000:.1f} ms. "
            f"Check that both receivers are on the same network "
            f"and collector.py timestamps are accurate."
        )

    rx1_idx = np.flatnonzero(valid)
    rx2_idx = nearest_idx[valid]
    deltas  = nearest_delta[valid]

    # ── Keep best RX1 per RX2 frame ─────────────────────
    best_for_rx2 = {}
    for i1, i2, delta in zip(rx1_idx, rx2_idx, deltas):
        prev = best_for_rx2.get(int(i2))
        if prev is None or delta < prev[0]:
            best_for_rx2[int(i2)] = (float(delta), int(i1))

    # Sort by RX1 index so time order is preserved
    pairs = sorted(
        [(i1, i2) for i2, (_, i1) in best_for_rx2.items()],
        key=lambda p: p[0]
    )

    if not pairs:
        raise ValueError("Zero unique synchronised pairs after deduplication.")

    rx1_i = np.array([p[0] for p in pairs], dtype=np.int64)
    rx2_i = np.array([p[1] for p in pairs], dtype=np.int64)

    sync1    = f1[rx1_i]                          # (M, 192)
    sync2    = f2[rx2_i]                          # (M, 192)
    combined = np.concatenate([sync1, sync2], axis=1)  # (M, 384)
    sync_t   = t1[rx1_i]

    print(
        f"    RX1={len(t1):4d}  RX2={len(t2):4d}  "
        f"dt1={dt1*1000:5.1f} ms  dt2={dt2*1000:5.1f} ms  "
        f"max_delta={max_delta*1000:.0f} ms  "
        f"synced={len(combined):4d}"
    )

    return sync_t, combined


# ============================================================
# PROCESS ONE SESSION
# ============================================================

def process_session(session_dir):
    """
    Load rx1.csv, rx2.csv, synchronise, slide windows,
    attach labels.

    Returns
    -------
    X        : float32 (W, 30, 384)
    Y_act    : float32 (W, 7)
    Y_cnt    : int64   (W,)
    Y_loc    : float32 (W, 4, 2)
    """
    rx1_path  = session_dir / "rx1.csv"
    rx2_path  = session_dir / "rx2.csv"
    meta_path = session_dir / "metadata.json"

    for p in [rx1_path, rx2_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    # ── Labels from metadata ─────────────────────────────
    persons = get_persons(meta_path)
    activity_vec, count, location = build_labels(persons)

    # ── Load CSVs ────────────────────────────────────────
    rx1_df = pd.read_csv(rx1_path)
    rx2_df = pd.read_csv(rx2_path)

    if len(rx1_df) < WINDOW_SIZE:
        raise ValueError(
            f"{session_dir.name}: RX1 has only {len(rx1_df)} rows "
            f"(need ≥ {WINDOW_SIZE})."
        )
    if len(rx2_df) < WINDOW_SIZE:
        raise ValueError(
            f"{session_dir.name}: RX2 has only {len(rx2_df)} rows "
            f"(need ≥ {WINDOW_SIZE})."
        )

    # ── Synchronise RX1 and RX2 ──────────────────────────
    _, combined = synchronize(rx1_df, rx2_df)

    if len(combined) < WINDOW_SIZE:
        raise ValueError(
            f"{session_dir.name}: Only {len(combined)} synchronised "
            f"frames, need ≥ {WINDOW_SIZE}."
        )

    # ── Slide windows ────────────────────────────────────
    X_wins   = []
    act_wins = []
    cnt_wins = []
    loc_wins = []

    for start in range(0, len(combined) - WINDOW_SIZE + 1, STEP_SIZE):
        window = combined[start : start + WINDOW_SIZE]   # (30, 384)

        # Skip windows with non-finite values
        if not np.isfinite(window).all():
            continue

        X_wins.append(window.astype(np.float32))
        act_wins.append(activity_vec)
        cnt_wins.append(count)
        loc_wins.append(location)

    if not X_wins:
        raise ValueError(
            f"{session_dir.name}: No valid windows after sliding."
        )

    return (
        np.stack(X_wins,   axis=0),               # (W, 30, 384)
        np.stack(act_wins, axis=0),               # (W, 7)
        np.array(cnt_wins, dtype=np.int64),       # (W,)
        np.stack(loc_wins, axis=0),               # (W, 4, 2)
    )


# ============================================================
# SESSION-LEVEL TRAIN / VAL SPLIT
# ============================================================

def session_split(session_dirs, val_ratio):
    """
    Assign ENTIRE sessions to train or val.

    Rationale
    ---------
    Windows from the same session are highly correlated
    (overlapping window step = 3).  Splitting at the window
    level would leak near-identical windows into both train
    and val, giving a falsely optimistic validation accuracy.

    Strategy
    --------
    Group sessions by primary activity for stratification.
    Within each group, reserve val_ratio of sessions for val.
    If a group has only one session, it always goes to train.

    Returns train_sessions, val_sessions  (lists of Paths)
    """
    random.seed(SEED)

    groups = {}   # primary_activity → [session_dir, ...]

    for sd in session_dirs:
        meta_path = sd / "metadata.json"
        try:
            persons = get_persons(meta_path)
            key = persons[0]["activity"] if persons else "empty"
        except Exception:
            key = "unknown"
        groups.setdefault(key, []).append(sd)

    train_sessions = []
    val_sessions   = []

    for key, sessions in groups.items():
        random.shuffle(sessions)

        # At least 1 session in train; val only if group has > 1
        n_val = (
            max(1, int(len(sessions) * val_ratio))
            if len(sessions) > 1
            else 0
        )
        val_sessions.extend(sessions[:n_val])
        train_sessions.extend(sessions[n_val:])

    return train_sessions, val_sessions


# ============================================================
# NORMALISATION
# ============================================================

def compute_normalization(X_train):
    """
    Per-feature mean and std over all training windows
    and all time steps.

    mean shape : (1, 1, 384)  — broadcast-ready for (N, 30, 384)
    std  shape : (1, 1, 384)

    std values smaller than 1e-8 are replaced with 1.0
    to avoid division by zero on zero-variance subcarriers.
    """
    # Collapse N and T axes → stats over (N * T) per feature
    flat = X_train.reshape(-1, X_train.shape[-1])    # (N*30, 384)
    mean = flat.mean(axis=0, keepdims=True)           # (1, 384)
    std  = flat.std(axis=0,  keepdims=True)           # (1, 384)
    std[std < 1e-8] = 1.0

    # Add time axis so shapes become (1, 1, 384)
    mean = mean[:, np.newaxis, :].astype(np.float32)  # (1, 1, 384)
    std  = std[:,  np.newaxis, :].astype(np.float32)  # (1, 1, 384)

    return mean, std


def apply_normalization(X, mean, std):
    """
    Zero-mean unit-variance normalisation.
    X    : (N, 30, 384)
    mean : (1, 1, 384)
    std  : (1, 1, 384)
    """
    return ((X - mean) / std).astype(np.float32)


# ============================================================
# PROCESS ALL SESSIONS IN ONE SPLIT
# ============================================================

def process_split(session_dirs, split_name):
    """
    Run process_session() for every directory in the list.
    Skips sessions that fail with a warning.

    Returns
    -------
    X, Y_act, Y_cnt, Y_loc  — concatenated arrays, or None × 4
    summaries               — list of per-session info dicts
    """
    all_X   = []
    all_act = []
    all_cnt = []
    all_loc = []
    summaries = []

    for sd in session_dirs:
        try:
            X, Y_act, Y_cnt, Y_loc = process_session(sd)
        except Exception as e:
            print(f"  [SKIP] {sd.name}: {e}")
            continue

        all_X.append(X)
        all_act.append(Y_act)
        all_cnt.append(Y_cnt)
        all_loc.append(Y_loc)

        acts = [
            ACTIVITY_CLASSES[i]
            for i in range(len(ACTIVITY_CLASSES))
            if Y_act[0, i] > 0.5
        ]
        summaries.append({
            "session":    sd.name,
            "split":      split_name,
            "windows":    int(len(X)),
            "persons":    int(Y_cnt[0]),
            "activities": acts,
        })

        print(
            f"  [OK] {sd.name:20s}  "
            f"windows={len(X):4d}  "
            f"persons={Y_cnt[0]}  "
            f"acts={acts}"
        )

    if not all_X:
        return None, None, None, None, summaries

    return (
        np.concatenate(all_X,   axis=0),
        np.concatenate(all_act, axis=0),
        np.concatenate(all_cnt, axis=0),
        np.concatenate(all_loc, axis=0),
        summaries,
    )


# ============================================================
# SAVE ARRAYS
# ============================================================

def save_split(output_dir, split, X, Y_act, Y_cnt, Y_loc):
    np.save(output_dir / f"X_{split}.npy",            X)
    np.save(output_dir / f"Y_activity_{split}.npy",   Y_act)
    np.save(output_dir / f"Y_count_{split}.npy",      Y_cnt)
    np.save(output_dir / f"Y_location_{split}.npy",   Y_loc)


# ============================================================
# REPORT
# ============================================================

def print_report(X_train, X_val, Ya_train, Ya_val,
                 Yc_train, mean, std):

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  X_train  : {X_train.shape}")
    if X_val is not None:
        print(f"  X_val    : {X_val.shape}")
    print(f"  mean     : {mean.shape}   "
          f"range [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  std      : {std.shape}    "
          f"range [{std.min():.3f}, {std.max():.3f}]")
    print()
    print("  Activity window counts:")
    print(f"  {'class':12s}  {'idx':>3}  {'train':>8}  {'val':>8}")
    print("  " + "-" * 38)

    for i, act in enumerate(ACTIVITY_CLASSES):
        tr = int(Ya_train[:, i].sum())
        vl = int(Ya_val[:, i].sum()) if Ya_val is not None else 0
        print(f"  {act:12s}  {i:>3}  {tr:>8}  {vl:>8}")

    print()
    print("  Person count distribution (train):")
    for c in range(MAX_PERSONS + 1):
        n = int((Yc_train == c).sum())
        print(f"    {c} persons : {n:6d} windows")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-person CSI HAR preprocessing."
    )
    parser.add_argument(
        "--dataset", default="dataset",
        help="Root directory containing session_XXXX folders."
    )
    parser.add_argument(
        "--output", default="processed",
        help="Output directory for .npy and .json files."
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover session directories ─────────────────────
    session_dirs = sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir()]
    )
    if not session_dirs:
        raise RuntimeError(
            f"No session directories found inside {dataset_dir}. "
            f"Run the firmware and collector.py first to record data."
        )

    print()
    print("=" * 60)
    print("Wi-Fi CSI MULTI-PERSON HAR PREPROCESSING")
    print("=" * 60)
    print(f"  Dataset       : {dataset_dir.resolve()}")
    print(f"  Output        : {output_dir.resolve()}")
    print(f"  Sessions      : {len(session_dirs)}")
    print(f"  Window / Step : {WINDOW_SIZE} / {STEP_SIZE}")
    print(f"  Feature mode  : {FEATURE_MODE}")
    print(f"  Max persons   : {MAX_PERSONS}")
    print(f"  Val ratio     : {VALIDATION_RATIO}")
    print(f"  Seed          : {SEED}")
    print(f"  Activities    : {ACTIVITY_CLASSES}")
    print("=" * 60)
    print()

    # ── Session-level split ──────────────────────────────
    train_sessions, val_sessions = session_split(
        session_dirs, VALIDATION_RATIO
    )

    print(
        f"Session split — "
        f"train: {len(train_sessions)}  "
        f"val: {len(val_sessions)}"
    )
    print(f"  Train: {[s.name for s in train_sessions]}")
    print(f"  Val  : {[s.name for s in val_sessions]}")
    print()

    # ── Process training sessions ────────────────────────
    print("--- TRAINING SESSIONS ---")
    X_train, Ya_train, Yc_train, Yl_train, train_info = process_split(
        train_sessions, "train"
    )

    if X_train is None:
        raise RuntimeError(
            "No training data was successfully processed. "
            "Check that session directories contain valid CSV files."
        )

    # ── Process validation sessions ──────────────────────
    print()
    print("--- VALIDATION SESSIONS ---")
    X_val, Ya_val, Yc_val, Yl_val, val_info = process_split(
        val_sessions, "val"
    )

    # ── Normalisation (training stats only) ─────────────
    # BUG FIX: mean/std computed from X_train only.
    # Previously computed over the full dataset before splitting,
    # which contaminated validation accuracy.
    print()
    print("Computing normalisation from training data only ...")
    mean, std = compute_normalization(X_train)

    X_train = apply_normalization(X_train, mean, std)
    if X_val is not None:
        X_val = apply_normalization(X_val, mean, std)

    # ── Save training arrays ─────────────────────────────
    print()
    print("Saving arrays ...")
    save_split(output_dir, "train", X_train, Ya_train, Yc_train, Yl_train)

    if X_val is not None:
        save_split(output_dir, "val", X_val, Ya_val, Yc_val, Yl_val)
    else:
        print(
            "  [INFO] No validation sessions — "
            "X_val.npy not written. "
            "Training will run without validation."
        )

    # ── Normalisation stats ──────────────────────────────
    # Saved as (1, 1, 384) for broadcast compatibility
    # AND embedded into the model binary by weightloader.py
    np.save(output_dir / "feature_mean.npy", mean)   # (1, 1, 384)
    np.save(output_dir / "feature_std.npy",  std)    # (1, 1, 384)

    # weightloader.py reshapes to (384,) when embedding
    print(f"  feature_mean saved : {mean.shape}")
    print(f"  feature_std  saved : {std.shape}")

    # ── Label map ────────────────────────────────────────
    label_map = {act: i for i, act in enumerate(ACTIVITY_CLASSES)}
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)

    # ── Session summary ───────────────────────────────────
    all_info = train_info + (val_info if val_info else [])
    with open(output_dir / "session_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(all_info, f, indent=4)

    # ── Final report ─────────────────────────────────────
    print_report(
        X_train, X_val,
        Ya_train, Ya_val,
        Yc_train,
        mean, std
    )


if __name__ == "__main__":
    main()
