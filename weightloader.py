"""
weightloader.py
===============
Export TEDNet weights as a float32 binary (CSI2 v3)
and upload to RX2 via serial.

════════════════════════════════════════════════════════
WHAT THIS SCRIPT DOES
════════════════════════════════════════════════════════

1.  Finds the latest .pth checkpoint in models/.
2.  Loads all weight tensors as float32.
3.  Loads feature_mean.npy and feature_std.npy from
    processed/ and adds them as named tensors so the
    firmware can normalise inputs at inference time.
    (Bug 1 fix — these were previously missing.)
4.  Encodes activity class metadata as a tensor so the
    firmware knows class names without hard-coding them.
5.  Writes a self-contained binary file (CSI2 v3).
6.  Optionally uploads the binary to RX2 via serial
    using the MODEL_BEGIN / MODEL_READY / MODEL_END
    protocol that the firmware already implements.

════════════════════════════════════════════════════════
BINARY FORMAT  (CSI2 version 3 — float32 only)
════════════════════════════════════════════════════════

Offset  Size   Field
──────  ─────  ─────────────────────────────────────────
0       4      Magic bytes  "CSI2"
4       4      Version      3  (uint32 LE)
8       4      Tensor count N  (uint32 LE)

For each of the N tensors:
  0     2      name_len          (uint16 LE)
  2     name_len  name           (UTF-8, NOT null-terminated)
  +0    1      dtype             (uint8, always 1 = float32)
  +1    4      ndim              (uint32 LE)
  +5    ndim×4 dims[i]           (uint32 LE each)
  +x    8      data_size         (uint64 LE, bytes = elems × 4)
  +x+8  data_size  data         (raw float32 LE)

Special tensors embedded after the model weights:
  "feature_mean"   shape (384,)  — per-feature training mean
  "feature_std"    shape (384,)  — per-feature training std
  "__label_info__" shape (K,)    — ASCII bytes of a JSON string

════════════════════════════════════════════════════════
UPLOAD PROTOCOL  (matches RX2 firmware exactly)
════════════════════════════════════════════════════════

  PC  →  "MODEL_BEGIN,<size>,3\\n"
  RX2 →  "MODEL_READY"
  PC  →  raw binary in CHUNK_SIZE chunks
  PC  →  "MODEL_END\\n"
  RX2 →  "MODEL_LOADED"   (after parsing succeeds)

════════════════════════════════════════════════════════
BUGS FIXED vs ORIGINAL
════════════════════════════════════════════════════════

  Bug 1 — feature_mean / feature_std were never included
           in the binary.  The firmware set
           modelHasNormalization = false and ran raw
           amplitude values through a model trained on
           zero-mean unit-variance data.
           Fixed: both tensors are now embedded.

  Bug 5 — BatchNorm running_mean / running_var were
           INT8-quantised, corrupting the normalisation
           inside every CNN layer.
           Fixed: all tensors are written as float32;
           no quantisation is performed anywhere.

════════════════════════════════════════════════════════
"""

import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed.")
    print("       pip install torch")
    sys.exit(1)

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed.")
    print("       pip install pyserial")
    sys.exit(1)


# ============================================================
# PATHS
# ============================================================

BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = BASE_DIR / "models"
PROCESSED  = BASE_DIR / "processed"
BIN_FILE   = MODEL_DIR / "model_fp32.bin"


# ============================================================
# SERIAL CONFIGURATION
# ============================================================

SERIAL_PORT = "COM11"
BAUD_RATE   = 115200
CHUNK_SIZE  = 256        # bytes per serial.write() call


# ============================================================
# BINARY FORMAT CONSTANTS
# ============================================================

MAGIC          = b"CSI2"
FORMAT_VERSION = 3
DTYPE_FLOAT32  = 1       # only dtype used in v3

MAX_MODEL_BYTES = 8 * 1024 * 1024   # 8 MB PSRAM hard limit


# ============================================================
# STEP 1  —  FIND LATEST CHECKPOINT
# ============================================================

def find_latest_model():
    """
    Return the most recently modified .pth file in MODEL_DIR.
    Raises FileNotFoundError if none exist.
    """
    candidates = sorted(
        MODEL_DIR.glob("*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .pth checkpoint found in {MODEL_DIR}.\n"
            f"Run train.py first."
        )
    return candidates[0]


# ============================================================
# STEP 2  —  LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model_path):
    """
    Load a PyTorch checkpoint and extract the state_dict.

    Supports three checkpoint structures:
      { "model_state_dict": {...}, ... }   ← train.py default
      { "state_dict":       {...}, ... }   ← older convention
      { <layer>: tensor, ... }             ← bare state dict

    Returns
    -------
    tensors : dict[str, torch.Tensor]   float32, CPU, contiguous
    ckpt    : dict                       full checkpoint dict
    """
    print(f"[1/5] Loading checkpoint: {model_path.name}")

    ckpt = torch.load(str(model_path), map_location="cpu")

    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"Unsupported checkpoint type: {type(ckpt)}. "
            f"Expected a dict produced by torch.save()."
        )

    # Extract state dict
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # Assume the checkpoint IS the state dict
        state_dict = ckpt

    # ── Print checkpoint metadata ────────────────────────
    print(f"       activity_classes : {ckpt.get('activity_classes', '?')}")
    print(f"       n_activities     : {ckpt.get('n_activities', '?')}")
    print(f"       max_persons      : {ckpt.get('max_persons', '?')}")
    print(f"       n_count_classes  : {ckpt.get('n_count_classes', '?')}")
    print(f"       window_size      : {ckpt.get('window_size', '?')}")
    print(f"       n_features       : {ckpt.get('n_features', '?')}")
    print(f"       best_metric      : {ckpt.get('best_metric', '?')}")

    # ── Convert every tensor to float32 ─────────────────
    # Note: this includes BatchNorm running_mean,
    # running_var, and num_batches_tracked.
    # All are kept as float32.  num_batches_tracked is a
    # scalar int64 in PyTorch; .float() converts it but
    # the firmware never looks it up so it is harmless.
    tensors = {}
    for name, val in state_dict.items():
        if torch.is_tensor(val):
            tensors[name] = (
                val.detach()
                   .cpu()
                   .float()       # always float32
                   .contiguous()
            )

    print(f"       State dict tensors: {len(tensors)}")
    return tensors, ckpt


# ============================================================
# STEP 3A  —  EMBED NORMALISATION TENSORS
# ============================================================

def embed_normalization(tensors):
    """
    Load feature_mean.npy and feature_std.npy from processed/
    and add them to the tensor dict as "feature_mean" and
    "feature_std" with shape (384,).

    The firmware's parseModelContainer() looks for these by
    name.  If they are absent modelHasNormalization stays
    false and inference receives raw amplitude values instead
    of normalised ones — causing completely wrong predictions.

    This was Bug 1 in the original codebase.
    """
    mean_path = PROCESSED / "feature_mean.npy"
    std_path  = PROCESSED / "feature_std.npy"

    for p in [mean_path, std_path]:
        if not p.exists():
            print(
                f"\n  [FATAL] {p.name} not found in {PROCESSED}.\n"
                f"          Run preprocess_csi_v2.py before "
                f"weightloader.py.\n"
                f"          Without normalisation stats the model "
                f"will give wrong predictions on the device."
            )
            sys.exit(1)

    # preprocess saves shape (1, 1, 384) for numpy broadcast.
    # The firmware expects a flat (384,) array.
    mean = np.load(str(mean_path)).reshape(-1).astype(np.float32)
    std  = np.load(str(std_path )).reshape(-1).astype(np.float32)

    if mean.shape[0] != 384 or std.shape[0] != 384:
        raise ValueError(
            f"feature_mean/std must have 384 elements. "
            f"Got mean={mean.shape} std={std.shape}."
        )

    # Guard against zero-std entries (should not happen after
    # preprocess but be safe)
    std[std < 1e-8] = 1.0

    tensors["feature_mean"] = torch.from_numpy(mean)
    tensors["feature_std"]  = torch.from_numpy(std)

    print(
        f"       feature_mean : shape={mean.shape}  "
        f"range=[{mean.min():.4f}, {mean.max():.4f}]"
    )
    print(
        f"       feature_std  : shape={std.shape}  "
        f"range=[{std.min():.4f}, {std.max():.4f}]"
    )
    return tensors


# ============================================================
# STEP 3B  —  EMBED LABEL INFO
# ============================================================

def embed_label_info(tensors, ckpt):
    """
    Serialise activity class metadata as a JSON string and
    store its bytes as a float32 tensor named "__label_info__".

    The firmware can decode this by reading the float32 values
    as uint8 bytes and parsing the ASCII JSON.  This makes the
    binary self-describing — no hard-coding of class names
    in the firmware is required for future model updates.

    JSON fields stored:
        activity_classes  list[str]
        label_map         dict[str, int]
        max_persons       int
        n_activities      int
        n_count_classes   int
    """
    info = {
        "activity_classes": ckpt.get("activity_classes", []),
        "label_map":        ckpt.get("label_map",        {}),
        "max_persons":      ckpt.get("max_persons",       4),
        "n_activities":     ckpt.get("n_activities",      7),
        "n_count_classes":  ckpt.get("n_count_classes",   5),
    }

    json_bytes = json.dumps(
        info, separators=(",", ":")
    ).encode("utf-8")

    # Store each byte as a float32 value (0-255).
    arr = np.frombuffer(json_bytes, dtype=np.uint8).astype(np.float32)
    tensors["__label_info__"] = torch.from_numpy(arr.copy())

    print(
        f"       __label_info__ : {len(json_bytes)} bytes  "
        f"→  {json_bytes.decode()}"
    )
    return tensors


# ============================================================
# STEP 4  —  WRITE BINARY
# ============================================================

def write_binary(tensors, output_path):
    """
    Serialise all tensors into the CSI2 v3 binary format.

    Layout per tensor:
        uint16  name_len
        bytes   name  (UTF-8, no null terminator)
        uint8   dtype (1 = float32)
        uint32  ndim
        uint32  dims[0..ndim-1]
        uint64  data_size  (bytes)
        bytes   data       (raw float32 LE)

    All integers are little-endian.
    All floating-point data is IEEE 754 single precision LE.
    """
    print(f"\n[4/5] Writing float32 binary")
    print(f"       Output: {output_path}")
    print()
    print(
        f"  {'Tensor name':<55}  "
        f"{'Shape':<22}  "
        f"{'Bytes':>10}"
    )
    print("  " + "-" * 93)

    with open(str(output_path), "wb") as f:

        # ── File header ──────────────────────────────────
        f.write(MAGIC)                                   # 4 bytes
        f.write(struct.pack("<I", FORMAT_VERSION))       # 4 bytes
        f.write(struct.pack("<I", len(tensors)))         # 4 bytes

        # ── Tensor records ────────────────────────────────
        for name, tensor in tensors.items():

            name_bytes = name.encode("utf-8")
            data_np    = tensor.numpy().astype(np.float32)
            data_bytes = data_np.tobytes()             # raw float32 LE

            # name_len (uint16)
            f.write(struct.pack("<H", len(name_bytes)))

            # name
            f.write(name_bytes)

            # dtype (uint8)
            f.write(struct.pack("<B", DTYPE_FLOAT32))

            # ndim (uint32)
            ndim = tensor.dim()
            f.write(struct.pack("<I", ndim))

            # dims (uint32 each)
            for d in tensor.shape:
                f.write(struct.pack("<I", int(d)))

            # data_size (uint64)
            f.write(struct.pack("<Q", len(data_bytes)))

            # data
            f.write(data_bytes)

            shape_str = str(tuple(tensor.shape)) if ndim > 0 else "()"
            print(
                f"  {name:<55}  "
                f"{shape_str:<22}  "
                f"{len(data_bytes):>10,} B"
            )

    # ── File size check ──────────────────────────────────
    file_size = output_path.stat().st_size

    print()
    print(f"  Total binary size : {file_size:,} bytes  "
          f"({file_size / 1024 / 1024:.3f} MB)")

    if file_size > MAX_MODEL_BYTES:
        print()
        print(
            f"  [FATAL] Binary ({file_size / 1024 / 1024:.2f} MB) "
            f"exceeds PSRAM limit "
            f"({MAX_MODEL_BYTES / 1024 / 1024:.0f} MB)."
        )
        print(
            f"          Reduce model size (fewer layers, smaller "
            f"hidden dims) before uploading."
        )
        sys.exit(1)

    return file_size


# ============================================================
# SHA256
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# STEP 5  —  UPLOAD TO RX2
# ============================================================

def wait_for_response(ser, expected, timeout_s=15):
    """
    Read lines from ser until one contains `expected` or
    until timeout_s seconds have elapsed.

    Prints every non-empty line received.
    Returns True if expected was found, False on timeout.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if line:
            print(f"         RX2: {line}")
        if expected in line:
            return True
        if "ERROR" in line:
            return False
    return False


def upload_to_rx2(bin_path, file_size):
    """
    Send the binary to RX2 using the serial protocol:

        PC  →  MODEL_BEGIN,<size>,3\\n
        RX2 →  MODEL_READY
        PC  →  <raw bytes in chunks>
        PC  →  MODEL_END\\n
        RX2 →  MODEL_LOADED
    """
    print(f"\n[5/5] Uploading to RX2 on {SERIAL_PORT}")
    print(f"       Size  : {file_size:,} bytes "
          f"({file_size / 1024 / 1024:.3f} MB)")
    print()
    print("       ⚠  Make sure Arduino Serial Monitor is CLOSED.")
    print("       ⚠  Make sure collector.py is NOT running.")
    print()

    # ── Open serial port ─────────────────────────────────
    try:
        ser = serial.Serial()
        ser.port          = SERIAL_PORT
        ser.baudrate      = BAUD_RATE
        ser.timeout       = 2
        ser.write_timeout = 10
        ser.dtr = False
        ser.rts = False
        ser.open()
        time.sleep(0.3)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"  [ERROR] Cannot open {SERIAL_PORT}: {e}")
        print(
            f"          Make sure the port is not held by another "
            f"process (Arduino IDE, collector.py, etc.)"
        )
        return False

    try:
        # Give RX2 time to settle after port open
        time.sleep(2.0)
        ser.reset_input_buffer()

        # ── Send MODEL_BEGIN ─────────────────────────────
        cmd = f"MODEL_BEGIN,{file_size},{FORMAT_VERSION}\n"
        print(f"       → {cmd.strip()}")
        ser.write(cmd.encode("ascii"))
        ser.flush()

        # ── Wait for MODEL_READY ─────────────────────────
        print("       Waiting for MODEL_READY ...")
        if not wait_for_response(ser, "MODEL_READY", timeout_s=20):
            print(
                "\n  [ERROR] RX2 did not respond with MODEL_READY.\n"
                "          Check that RX2 is powered and running "
                "the correct firmware."
            )
            return False

        print("       RX2 ready — sending binary ...")
        print()

        # ── Send raw binary ──────────────────────────────
        sent       = 0
        start_time = time.time()

        with open(str(bin_path), "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break

                ser.write(chunk)
                sent += len(chunk)

                elapsed = time.time() - start_time
                speed   = sent / max(elapsed, 1e-6) / 1024   # KB/s
                pct     = 100.0 * sent / file_size

                print(
                    f"\r       {pct:6.2f}%  "
                    f"{sent:>10,} / {file_size:,} B  "
                    f"{speed:7.1f} KB/s",
                    end="",
                    flush=True,
                )

        elapsed_total = time.time() - start_time
        avg_speed     = file_size / max(elapsed_total, 1e-6) / 1024

        print()
        print()
        print(
            f"       Transfer complete — "
            f"{file_size:,} bytes in "
            f"{elapsed_total:.1f} s  "
            f"({avg_speed:.1f} KB/s avg)"
        )

        # ── Send MODEL_END ───────────────────────────────
        print("       → MODEL_END")
        ser.write(b"MODEL_END\n")
        ser.flush()

        # ── Wait for MODEL_LOADED ────────────────────────
        print("       Waiting for MODEL_LOADED "
              "(may take up to 90 s while RX2 parses) ...")

        if wait_for_response(ser, "MODEL_LOADED", timeout_s=90):
            print()
            print("       ✓  Model uploaded and parsed successfully.")
            return True

        print()
        print(
            "  [ERROR] RX2 did not confirm MODEL_LOADED.\n"
            "          The binary may be corrupt or too large "
            "for PSRAM."
        )
        return False

    finally:
        ser.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("TEDNET  FLOAT32  WEIGHT LOADER  —  CSI2 v3")
    print("=" * 60)
    print(f"  Model dir  : {MODEL_DIR}")
    print(f"  Processed  : {PROCESSED}")
    print(f"  Output     : {BIN_FILE}")
    print(f"  Serial     : {SERIAL_PORT}  @  {BAUD_RATE}")
    print(f"  PSRAM max  : {MAX_MODEL_BYTES / 1024 / 1024:.0f} MB")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Find checkpoint ──────────────────────────
    model_path = find_latest_model()
    print(f"\n       Using: {model_path.name}")

    # ── Step 2: Load weights ─────────────────────────────
    print(f"\n[2/5] Extracting float32 tensors ...")
    tensors, ckpt = load_checkpoint(model_path)

    # ── Step 3a: Embed normalisation ─────────────────────
    print(f"\n[3/5] Embedding normalisation tensors ...")
    tensors = embed_normalization(tensors)

    # ── Step 3b: Embed label info ─────────────────────────
    print(f"\n       Embedding label info ...")
    tensors = embed_label_info(tensors, ckpt)

    # ── Step 4: Write binary ──────────────────────────────
    file_size = write_binary(tensors, BIN_FILE)

    # ── SHA256 ────────────────────────────────────────────
    digest = sha256_file(BIN_FILE)
    print(f"  SHA256 : {digest}")

    # ── PSRAM usage summary ───────────────────────────────
    pct = 100.0 * file_size / MAX_MODEL_BYTES
    print(f"  PSRAM  : {file_size:,} / {MAX_MODEL_BYTES:,} B "
          f"({pct:.1f}% used  —  "
          f"{(MAX_MODEL_BYTES - file_size) / 1024 / 1024:.2f} MB free)")

    # ── Step 5: Upload ────────────────────────────────────
        # ── Step 5: Upload ────────────────────────────────────
    print()
    print("─" * 60)
    print("  Auto-uploading to RX2 ...")

    ok = upload_to_rx2(BIN_FILE, file_size)
    print()
    if ok:
        print("=" * 60)
        print("MODEL TRANSFER COMPLETE")
        print("=" * 60)
        print()
        print("  Next steps:")
        print("  1. Open the CSI_CONTROL Wi-Fi hotspot.")
        print("  2. Navigate to http://192.168.4.1")
        print("  3. Press START TESTING to run live inference.")
    else:
        print("=" * 60)
        print("MODEL TRANSFER FAILED")
        print("=" * 60)
        print()
        print(f"  Binary is saved at:")
        print(f"    {BIN_FILE}")
        print()
        print("  To retry upload, run weightloader.py again.")

import time
if __name__ == "__main__":
    main()
    time.sleep(3)

