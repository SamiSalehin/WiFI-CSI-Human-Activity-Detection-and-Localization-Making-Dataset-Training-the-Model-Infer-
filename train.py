"""
train.py
========
Multi-Task TEDNet — Wi-Fi CSI Human Activity Recognition
and Localization

════════════════════════════════════════════════════════
INPUT  (from processed/)
════════════════════════════════════════════════════════

X_train.npy              (N, 30, 384)  float32
Y_activity_train.npy     (N, 7)         float32  multi-hot
Y_count_train.npy        (N,)           int64    0-4
Y_location_train.npy     (N, 4, 2)     float32  -1 = absent

X_val.npy                (N, 30, 384)  float32   optional
Y_activity_val.npy       ...
Y_count_val.npy          ...
Y_location_val.npy       ...

════════════════════════════════════════════════════════
MODEL ARCHITECTURE
════════════════════════════════════════════════════════

Shared backbone
    Conv1d(384 → 128, k=3) → BN → GELU
    Conv1d(128 → 128, k=3) → BN → GELU
    Learnable positional embedding  (30, 128)
    4 × TransformerEncoderLayer(d=128, heads=8, ff=512)
    Temporal mean pool  →  LayerNorm  →  128-dim vector

Three task heads  (each: Linear → GELU → Dropout → Linear)
    Activity head  : 128 → 64 → 7    BCEWithLogitsLoss
    Count head     : 128 → 64 → 5    CrossEntropyLoss
    Location head  : 128 → 64 → 8    Masked MSELoss
                     8 = MAX_PERSONS × 2 = 4 × (x, y)

Total loss = W_act × L_act  +  W_cnt × L_cnt  +  W_loc × L_loc

════════════════════════════════════════════════════════
OUTPUT  (models/)
════════════════════════════════════════════════════════

tednet_csi_har_best.pth
    model_state_dict
    activity_classes    list[str]
    label_map           dict[str, int]
    n_features          384
    window_size         30
    n_activities        7
    max_persons         4
    n_count_classes     5
    best_metric         float   (macro F1 on val)

════════════════════════════════════════════════════════
ACTIVITY CLASS ORDER  (must match preprocess + firmware)
════════════════════════════════════════════════════════

  0 empty    1 falling  2 lying   3 running
  4 sitting  5 standing 6 walking

════════════════════════════════════════════════════════
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR      = Path(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR    = BASE_DIR / "models"


# ============================================================
# HYPER-PARAMETERS
# ============================================================

# Architecture
N_FEATURES      = 384
WINDOW_SIZE     = 30
MAX_PERSONS     = 4
N_ACTIVITIES    = 7
N_COUNT_CLASSES = MAX_PERSONS + 1   # 0, 1, 2, 3, 4

CNN_CHANNELS       = 128
NHEAD              = 8
HEAD_DIM           = CNN_CHANNELS // NHEAD   # 16
TRANSFORMER_LAYERS = 4
TRANSFORMER_FF     = 512
DROPOUT            = 0.1
HEAD_HIDDEN        = 64

# Training
BATCH_SIZE    = 16
EPOCHS        = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
SEED          = 42

# Scheduler
LR_PATIENCE  = 6      # epochs without improvement before LR drop
LR_FACTOR    = 0.5    # multiply LR by this on plateau

# Loss weights
W_ACTIVITY = 1.0
W_COUNT    = 0.5
W_LOCATION = 0.3

# Location
LOCATION_SENTINEL = -1.0   # marks absent person slots

# Activity class names  (index = class ID)
ACTIVITY_CLASSES = [
    "empty",
    "falling",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class CSIDataset(Dataset):
    """
    Loads pre-processed arrays for one split (train or val).

    Arrays are produced by preprocess_csi_v2.py and live in
    PROCESSED_DIR.  All normalisation has already been applied.
    """

    def __init__(self, split: str):
        d = PROCESSED_DIR

        def _load(name):
            path = d / name
            if not path.exists():
                raise FileNotFoundError(
                    f"Expected file not found: {path}\n"
                    f"Run preprocess_csi_v2.py first."
                )
            return np.load(str(path))

        self.X     = _load(f"X_{split}.npy").astype(np.float32)
        self.Y_act = _load(f"Y_activity_{split}.npy").astype(np.float32)
        self.Y_cnt = _load(f"Y_count_{split}.npy").astype(np.int64)
        self.Y_loc = _load(f"Y_location_{split}.npy").astype(np.float32)

        # ── Shape checks ────────────────────────────────
        N = len(self.X)

        assert self.X.shape     == (N, WINDOW_SIZE, N_FEATURES), \
            f"[{split}] X shape mismatch: {self.X.shape}"
        assert self.Y_act.shape == (N, N_ACTIVITIES), \
            f"[{split}] Y_activity shape mismatch: {self.Y_act.shape}"
        assert self.Y_cnt.shape == (N,), \
            f"[{split}] Y_count shape mismatch: {self.Y_cnt.shape}"
        assert self.Y_loc.shape == (N, MAX_PERSONS, 2), \
            f"[{split}] Y_location shape mismatch: {self.Y_loc.shape}"

        print(
            f"  [{split:5s}]  X={self.X.shape}  "
            f"Y_act={self.Y_act.shape}  "
            f"Y_cnt={self.Y_cnt.shape}  "
            f"Y_loc={self.Y_loc.shape}"
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y_act[idx]),
            torch.tensor(self.Y_cnt[idx], dtype=torch.long),
            torch.from_numpy(self.Y_loc[idx]),
        )


# ============================================================
# MODEL
# ============================================================

class TEDNet(nn.Module):
    """
    Temporal Encoding Detector Network — multi-task edition.

    Shared backbone (CNN + Transformer) produces a 128-dim
    embedding.  Three independent heads produce:
        logits_act  (B, 7)       for activity detection
        logits_cnt  (B, 5)       for person count
        loc         (B, 4, 2)    for (x, y) per person slot
    """

    def __init__(self):
        super().__init__()

        # ── Shared backbone ──────────────────────────────

        # Conv1d expects (B, C_in, L)
        # input arrives as  (B, T, F) = (B, 30, 384)
        # we permute to     (B, F, T) = (B, 384, 30)
        self.cnn = nn.Sequential(
            nn.Conv1d(N_FEATURES, CNN_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm1d(CNN_CHANNELS),
            nn.GELU(),
            nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm1d(CNN_CHANNELS),
            nn.GELU(),
        )

        # Learnable positional embedding added before the Transformer
        self.pos_embed = nn.Parameter(
            torch.randn(1, WINDOW_SIZE, CNN_CHANNELS) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = CNN_CHANNELS,
            nhead          = NHEAD,
            dim_feedforward= TRANSFORMER_FF,
            dropout        = DROPOUT,
            activation     = "gelu",
            batch_first    = True,    # (B, T, C) throughout
            norm_first     = False,   # post-norm (same as original)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = TRANSFORMER_LAYERS,
        )

        # Applied to the temporal mean-pooled vector
        self.pool_norm = nn.LayerNorm(CNN_CHANNELS)

        # ── Activity head  (multi-label) ─────────────────
        # output[i] is an unbounded logit; sigmoid + 0.5 threshold
        self.head_activity = nn.Sequential(
            nn.Linear(CNN_CHANNELS, HEAD_HIDDEN),  # .0.weight / .0.bias
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, N_ACTIVITIES),  # .3.weight / .3.bias
        )

        # ── Count head  (single-label classification) ────
        self.head_count = nn.Sequential(
            nn.Linear(CNN_CHANNELS, HEAD_HIDDEN),  # .0.weight / .0.bias
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, N_COUNT_CLASSES),  # .3.weight / .3.bias
        )

        # ── Location head  (regression) ──────────────────
        # Outputs MAX_PERSONS * 2 = 8 values.
        # Reshaped to (B, MAX_PERSONS, 2) for the masked MSE loss.
        self.head_location = nn.Sequential(
            nn.Linear(CNN_CHANNELS, HEAD_HIDDEN),     # .0.weight / .0.bias
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, MAX_PERSONS * 2),  # .3.weight / .3.bias
        )

    def forward(self, x):
        """
        x : (B, 30, 384)

        Returns
        -------
        logits_act : (B, 7)
        logits_cnt : (B, 5)
        loc        : (B, 4, 2)
        """

        # CNN  (B, 30, 384) → (B, 128, 30) → (B, 30, 128)
        x = x.permute(0, 2, 1)          # (B, 384, 30)
        x = self.cnn(x)                  # (B, 128, 30)
        x = x.permute(0, 2, 1)          # (B, 30,  128)

        # Positional embedding + Transformer
        x = x + self.pos_embed           # broadcast over batch
        x = self.transformer(x)          # (B, 30, 128)

        # Temporal mean pool  →  LayerNorm  →  (B, 128)
        x = x.mean(dim=1)               # (B, 128)
        x = self.pool_norm(x)

        # Three heads
        logits_act = self.head_activity(x)                 # (B, 7)
        logits_cnt = self.head_count(x)                    # (B, 5)
        loc_flat   = self.head_location(x)                 # (B, 8)
        loc        = loc_flat.view(-1, MAX_PERSONS, 2)     # (B, 4, 2)

        return logits_act, logits_cnt, loc


# ============================================================
# LOSSES
# ============================================================

def masked_location_loss(pred_loc, true_loc):
    """
    Mean squared error computed only over VALID person slots.

    A slot is valid when true_loc[:, i, 0] != LOCATION_SENTINEL.
    Slots for absent persons (sentinel = -1.0) are ignored so
    the model is not penalised for predicting anything there.

    pred_loc : (B, MAX_PERSONS, 2)
    true_loc : (B, MAX_PERSONS, 2)  — sentinel = -1.0

    Returns scalar tensor.
    """
    # valid : (B, MAX_PERSONS)  — True for occupied slots
    valid = (true_loc[:, :, 0] != LOCATION_SENTINEL)

    if not valid.any():
        # No persons in the whole batch — return 0 with grad
        return (pred_loc * 0.0).sum()

    # Expand mask to cover both x and y
    valid_xy = valid.unsqueeze(-1).expand_as(pred_loc)   # (B, 4, 2)

    sq_err  = (pred_loc - true_loc) ** 2                 # (B, 4, 2)
    masked  = sq_err * valid_xy.float()
    n_valid = valid_xy.float().sum()

    return masked.sum() / (n_valid + 1e-8)


def total_loss(logits_act, logits_cnt, pred_loc,
               Y_act, Y_cnt, Y_loc,
               criterion_act, criterion_cnt):
    """
    Weighted sum of the three task losses.
    """
    l_act = criterion_act(logits_act, Y_act)
    l_cnt = criterion_cnt(logits_cnt, Y_cnt)
    l_loc = masked_location_loss(pred_loc, Y_loc)

    return (
        W_ACTIVITY * l_act
        + W_COUNT    * l_cnt
        + W_LOCATION * l_loc
    ), l_act.item(), l_cnt.item(), l_loc.item()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, criterion_act, criterion_cnt,
                    optimizer, scaler):
    """
    Run one full pass over the training DataLoader.
    Returns average total loss for the epoch.
    """
    model.train()

    total = 0.0
    n     = 0

    iterator = (
        tqdm(loader, desc="  train", leave=False)
        if USE_TQDM else loader
    )

    for X, Y_act, Y_cnt, Y_loc in iterator:
        X     = X.to(DEVICE)
        Y_act = Y_act.to(DEVICE)
        Y_cnt = Y_cnt.to(DEVICE)
        Y_loc = Y_loc.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        use_amp = DEVICE.type == "cuda"

        # CORRECT
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            logits_act, logits_cnt, pred_loc = model(X)
            loss, _, _, _ = total_loss(
                logits_act, logits_cnt, pred_loc,
                Y_act, Y_cnt, Y_loc,
                criterion_act, criterion_cnt,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item() * len(X)
        n     += len(X)

    return total / n


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion_act, criterion_cnt):
    """
    Run one full pass over a DataLoader in eval mode.

    Returns
    -------
    avg_loss   : float
    macro_f1   : float   macro-averaged F1 across 7 activity classes
    cnt_acc    : float   person-count classification accuracy
    loc_rmse   : float   location RMSE (metres) over valid slots
    per_class_f1 : ndarray (7,)  per-class F1
    """
    model.eval()

    total_l = 0.0
    n       = 0

    # Per-class TP / FP / FN for F1
    tp = np.zeros(N_ACTIVITIES, dtype=np.float64)
    fp = np.zeros(N_ACTIVITIES, dtype=np.float64)
    fn = np.zeros(N_ACTIVITIES, dtype=np.float64)

    cnt_correct = 0
    cnt_total   = 0

    loc_sq_sum = 0.0
    loc_n      = 0

    for X, Y_act, Y_cnt, Y_loc in loader:
        X     = X.to(DEVICE)
        Y_act = Y_act.to(DEVICE)
        Y_cnt = Y_cnt.to(DEVICE)
        Y_loc = Y_loc.to(DEVICE)

        logits_act, logits_cnt, pred_loc = model(X)

        loss, _, _, _ = total_loss(
            logits_act, logits_cnt, pred_loc,
            Y_act, Y_cnt, Y_loc,
            criterion_act, criterion_cnt,
        )
        total_l += loss.item() * len(X)
        n       += len(X)

        # ── Activity F1 ───────────────────────────────
        pred_act = (torch.sigmoid(logits_act) > 0.5).cpu().numpy()
        true_act = (Y_act.cpu().numpy() > 0.5)

        tp += ( pred_act &  true_act).sum(axis=0)
        fp += ( pred_act & ~true_act).sum(axis=0)
        fn += (~pred_act &  true_act).sum(axis=0)

        # ── Count accuracy ────────────────────────────
        pred_cnt     = logits_cnt.argmax(dim=1)
        cnt_correct += (pred_cnt == Y_cnt).sum().item()
        cnt_total   += len(Y_cnt)

        # ── Location RMSE (valid slots only) ──────────
        true_np  = Y_loc.cpu().numpy()                    # (B, 4, 2)
        pred_np  = pred_loc.cpu().numpy()                 # (B, 4, 2)
        valid_mask = (true_np[:, :, 0] != LOCATION_SENTINEL)  # (B, 4)

        dx = pred_np[:, :, 0] - true_np[:, :, 0]
        dy = pred_np[:, :, 1] - true_np[:, :, 1]
        sq = (dx ** 2 + dy ** 2) * valid_mask.astype(np.float64)

        loc_sq_sum += sq.sum()
        loc_n      += valid_mask.sum()

    avg_loss = total_l / n

    # Per-class F1  (avoid division by zero with +eps)
    per_class_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    macro_f1     = float(per_class_f1.mean())

    cnt_acc  = cnt_correct / cnt_total
    loc_rmse = float(np.sqrt(loc_sq_sum / (loc_n + 1e-8)))

    return avg_loss, macro_f1, cnt_acc, loc_rmse, per_class_f1


# ============================================================
# PRINT PER-CLASS DETAIL
# ============================================================

def print_per_class(per_class_f1):
    for i, (act, f1) in enumerate(zip(ACTIVITY_CLASSES, per_class_f1)):
        bar = "█" * int(f1 * 20)
        print(f"    {i} {act:10s}  F1={f1:.3f}  {bar}")


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(model, path, best_metric, label_map):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # Metadata read by weightloader.py and firmware
            "n_features":       N_FEATURES,
            "window_size":      WINDOW_SIZE,
            "n_activities":     N_ACTIVITIES,
            "max_persons":      MAX_PERSONS,
            "n_count_classes":  N_COUNT_CLASSES,
            "activity_classes": ACTIVITY_CLASSES,
            "label_map":        label_map,
            "best_metric":      best_metric,
        },
        str(path),
    )


# ============================================================
# MAIN
# ============================================================

def train():
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Header ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("TEDNET  MULTI-TASK  CSI  HAR  TRAINING")
    print("=" * 60)
    print(f"  Device           : {DEVICE}")
    print(f"  Activities       : {ACTIVITY_CLASSES}")
    print(f"  N_ACTIVITIES     : {N_ACTIVITIES}")
    print(f"  MAX_PERSONS      : {MAX_PERSONS}")
    print(f"  N_COUNT_CLASSES  : {N_COUNT_CLASSES}")
    print(f"  WINDOW_SIZE      : {WINDOW_SIZE}")
    print(f"  N_FEATURES       : {N_FEATURES}")
    print(f"  CNN_CHANNELS     : {CNN_CHANNELS}")
    print(f"  TRANSFORMER_LAYERS: {TRANSFORMER_LAYERS}")
    print(f"  NHEAD            : {NHEAD}")
    print(f"  TRANSFORMER_FF   : {TRANSFORMER_FF}")
    print(f"  HEAD_HIDDEN      : {HEAD_HIDDEN}")
    print(f"  DROPOUT          : {DROPOUT}")
    print(f"  EPOCHS           : {EPOCHS}")
    print(f"  BATCH_SIZE       : {BATCH_SIZE}")
    print(f"  LEARNING_RATE    : {LEARNING_RATE}")
    print(f"  WEIGHT_DECAY     : {WEIGHT_DECAY}")
    print(f"  W_ACTIVITY       : {W_ACTIVITY}")
    print(f"  W_COUNT          : {W_COUNT}")
    print(f"  W_LOCATION       : {W_LOCATION}")
    print("=" * 60)

    # ── Label map ────────────────────────────────────────
    label_map_path = PROCESSED_DIR / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, encoding="utf-8") as f:
            label_map = json.load(f)
        print(f"\nLabel map loaded: {label_map}")
    else:
        label_map = {act: i for i, act in enumerate(ACTIVITY_CLASSES)}
        print(f"\nLabel map (default): {label_map}")

    # ── Datasets ─────────────────────────────────────────
    print("\nLoading datasets ...")
    train_ds = CSIDataset("train")

    has_val = (PROCESSED_DIR / "X_val.npy").exists()
    val_ds  = CSIDataset("val") if has_val else None

    if not has_val:
        print(
            "  [INFO] X_val.npy not found — "
            "training without validation. "
            "Model selection will use training loss."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = DEVICE.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size  = BATCH_SIZE,
            shuffle     = False,
            num_workers = 0,
            pin_memory  = DEVICE.type == "cuda",
        )
        if val_ds else None
    )

    # ── Model ────────────────────────────────────────────
    model   = TEDNet().to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters()
                  if p.requires_grad)
    print(f"\nParameters : {n_param:,}  (trainable: {n_train:,})")

    # ── Loss functions ────────────────────────────────────
    criterion_act = nn.BCEWithLogitsLoss()   # multi-label
    criterion_cnt = nn.CrossEntropyLoss()    # single-label

    # ── Optimiser + scheduler ────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "max",     # maximise macro F1
        factor   = LR_FACTOR,
        patience = LR_PATIENCE,
    )

    # AMP scaler (no-op on CPU)
# CORRECT - safe on both CPU and GPU
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    # ── Training loop ────────────────────────────────────
    best_path   = MODELS_DIR / "tednet_csi_har_best.pth"
    best_metric = -1.0        # macro F1 (or -train_loss if no val)

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):

        # ── Train ────────────────────────────────────────
        train_loss = train_one_epoch(
            model, train_loader,
            criterion_act, criterion_cnt,
            optimizer, scaler,
        )

        # ── Validate ─────────────────────────────────────
        if val_loader is not None:
            val_loss, macro_f1, cnt_acc, loc_rmse, per_f1 = evaluate(
                model, val_loader,
                criterion_act, criterion_cnt,
            )
            metric = macro_f1
            lr     = optimizer.param_groups[0]["lr"]

            print(
                f"Ep {epoch:03d}/{EPOCHS}  "
                f"train={train_loss:.4f}  "
                f"val={val_loss:.4f}  "
                f"F1={macro_f1:.3f}  "
                f"cnt={cnt_acc:.3f}  "
                f"rmse={loc_rmse:.3f}  "
                f"lr={lr:.2e}"
            )

            scheduler.step(metric)

        else:
            # No validation — select on lowest training loss
            metric = -train_loss
            print(
                f"Ep {epoch:03d}/{EPOCHS}  "
                f"train={train_loss:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        # ── Checkpoint if best ────────────────────────────
        if metric > best_metric:
            best_metric = metric
            save_checkpoint(model, best_path, best_metric, label_map)

            if val_loader is not None:
                print(
                    f"  → BEST  "
                    f"(F1={macro_f1:.3f}  "
                    f"cnt={cnt_acc:.3f}  "
                    f"rmse={loc_rmse:.3f})"
                )
                # Print per-class F1 every time we get a new best
                print_per_class(per_f1)
            else:
                print(f"  → BEST  (train_loss={train_loss:.4f})")

    # ── Final evaluation ──────────────────────────────────
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model : {best_path}")
    print(f"  Best metric: {best_metric:.4f}")

    if val_loader is not None:
        print()
        print("Final evaluation on validation set ...")

        # Reload best weights for final report
        ckpt = torch.load(str(best_path), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

        val_loss, macro_f1, cnt_acc, loc_rmse, per_f1 = evaluate(
            model, val_loader,
            criterion_act, criterion_cnt,
        )

        print(f"  val_loss   : {val_loss:.4f}")
        print(f"  macro_F1   : {macro_f1:.4f}")
        print(f"  count_acc  : {cnt_acc:.4f}")
        print(f"  loc_RMSE   : {loc_rmse:.4f} m")
        print()
        print("  Per-class F1:")
        print_per_class(per_f1)

    print("=" * 60)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()
    time.sleep(4)  # allow stdout to flush before exit
