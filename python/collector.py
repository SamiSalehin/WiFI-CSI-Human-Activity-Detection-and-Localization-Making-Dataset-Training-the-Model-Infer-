"""
collector.py
============
Wi-Fi CSI HAR  —  Dataset Collector + Training Manager

HARDWARE
    RX1  ESP32-S3  →  COM9   (CSI only, no control)
    RX2  ESP32-S3  →  COM11  (CSI + HTML interface + training trigger)

════════════════════════════════════════════════════════
DATASET COLLECTION
════════════════════════════════════════════════════════

RX2 sends a SESSION_START line when the HTML Start button
is pressed.  collector.py creates:

    dataset/
        session_XXXX/
            rx1.csv
            rx2.csv
            metadata.json   

and saves rows until SESSION_STOP is received.

Multi-person SESSION_START format (new):

    SESSION_START,num_persons=2,
    p0_activity=standing,p0_x=1.5,p0_y=1.0,
    p1_activity=sitting,p1_x=3.0,p1_y=2.0,
    subject=S01,note=

Single-person (backward compatible):

    SESSION_START,num_persons=1,
    p0_activity=standing,p0_x=1.5,p0_y=1.0,
    subject=S01,note=

Empty room:

    SESSION_START,num_persons=0,subject=S01,note=

metadata.json produced (new format):

    {
      "session_id"  : "session_0001",
      "created_utc" : "...",
      "num_persons" : 2,
      "persons": [
        {"activity": "standing", "x": 1.5, "y": 1.0},
        {"activity": "sitting",  "x": 3.0, "y": 2.0}
      ],
      "subject" : "S01",
      "note"    : ""
    }

════════════════════════════════════════════════════════
TRAINING PIPELINE
════════════════════════════════════════════════════════

When RX2 sends TRAINING_START (HTML Train button):

    preprocess_csi_v2.py
            |
            v
        train.py
            |
            v
    models/tednet_csi_har_best.pth
            |
            v
      weightloader.py  ← releases COM11 first, re-opens after

════════════════════════════════════════════════════════
SERIAL LINES RECOGNISED FROM RX2
════════════════════════════════════════════════════════

    SESSION_START,...    → start dataset collection
    SESSION_STOP         → stop  dataset collection
    TRAINING_START       → launch training pipeline
    RX2,...              → CSI row  (saved when session active)
    MODEL_*              → printed to console
    PSRAM_*              → printed to console
    TEDNET_*             → printed to console
    RX2_READY            → printed to console
    TEST_RESULT:*        → printed to console

════════════════════════════════════════════════════════
"""

import csv
import json
import os
import re
import sys
import time
import threading
import subprocess
from datetime import datetime, timezone

import serial
import io
from collections import deque


# ============================================================
# SERIAL CONFIGURATION
# ============================================================

RX1_PORT = "COM9"
RX2_PORT = "COM11"
BAUD     = 115200

# ============================================================
# PATHS
# ============================================================

# collector.py lives in:  project/python/collector.py
# So PROJECT_ROOT is one level up.

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATASET_ROOT        = os.path.join(PROJECT_ROOT, "dataset")
PROCESSED_DIR       = os.path.join(PROJECT_ROOT, "processed")
MODELS_DIR          = os.path.join(PROJECT_ROOT, "models")
PREPROCESS_SCRIPT   = os.path.join(PROJECT_ROOT, "preprocess_csi_v2.py")
TRAIN_SCRIPT        = os.path.join(PROJECT_ROOT, "train.py")
WEIGHTLOADER_SCRIPT = os.path.join(PROJECT_ROOT, "weightloader.py")
BEST_MODEL_PATH     = os.path.join(MODELS_DIR,   "tednet_csi_har_best.pth")
TEST_ON_SERVER_SCRIPT = os.path.join(PROJECT_ROOT, "test_on_server.py")


MAX_PERSONS = 4

# ============================================================
# THREAD / SESSION STATE
# ============================================================

state_lock = threading.Lock()

# Dataset collection
active         = False
session_dir    = None
metadata       = None
session_number = 0
files          = {}
writers        = {}

# Training pipeline
training_pipeline_running = False
training_lock = threading.Lock()
# Serial handles
rx1_serial  = None
rx2_serial  = None
serial_lock = threading.Lock()          # ← MUST BE HERE

# Signal reader threads to pause
serial_pause_event = threading.Event()  # ← MUST BE HERE

# ============================================================
# SERVER INFERENCE STATE                 ← server block AFTER
# ============================================================
server_test_active    = False
server_test_lock      = threading.Lock()
server_rx1_lines      = []
server_rx2_lines      = []
server_buffer_lock    = threading.Lock()
server_collecting_now = False
last_server_result    = None
TEMP_RX1               = os.path.join(PROJECT_ROOT, "temp_server_rx1.csv")
TEMP_RX2               = os.path.join(PROJECT_ROOT, "temp_server_rx2.csv")
SERVER_COLLECT_SECONDS = 5
serial_pause_event = threading.Event()  

# Signal reader threads to pause (during weightloader upload)
serial_pause_event = threading.Event()


# ============================================================
# CSV COLUMN HELPERS
# ============================================================

BASE_CSV_FIELDS = [
    "host_timestamp_iso",
    "rx_timestamp_ms",
    "rssi",
    "sequence",
    "channel",
    "bandwidth",
    "csi_length",
]


def make_csi_fields(n_values):
    """
    Convert n_values I/Q values into column names:
        SC0_I, SC0_Q, SC1_I, SC1_Q, ...
    """
    fields = []
    for sc in range(n_values // 2):
        fields.append(f"SC{sc}_I")
        fields.append(f"SC{sc}_Q")
    return fields


# ============================================================
# NEXT SESSION DIRECTORY
# ============================================================

def next_session_dir():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    numbers = []
    for name in os.listdir(DATASET_ROOT):
        m = re.fullmatch(r"session_(\d+)", name)
        if m:
            numbers.append(int(m.group(1)))
    next_num  = (max(numbers) if numbers else 0) + 1
    directory = os.path.join(DATASET_ROOT, f"session_{next_num:04d}")
    return next_num, directory


# ============================================================
# PARSE SESSION_START METADATA
# ============================================================

def parse_session_metadata(raw_string):
    """
    Parse the comma-separated key=value string that follows
    "SESSION_START," into a structured dict.

    Input example:
        "num_persons=2,p0_activity=standing,p0_x=1.5,p0_y=1.0,
         p1_activity=sitting,p1_x=3.0,p1_y=2.0,subject=S01,note="

    Returns:
        {
            "num_persons": 2,
            "persons": [
                {"activity": "standing", "x": 1.5, "y": 1.0},
                {"activity": "sitting",  "x": 3.0, "y": 2.0}
            ],
            "subject": "S01",
            "note":    ""
        }
    """

    # Split on commas, parse every key=value pair
    kv = {}
    for part in raw_string.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        kv[key.strip()] = value.strip()

    # Number of persons
    try:
        num_persons = int(kv.get("num_persons", 1))
    except (ValueError, TypeError):
        num_persons = 1

    num_persons = max(0, min(num_persons, MAX_PERSONS))

    # Per-person fields
    persons = []
    for i in range(num_persons):
        activity = kv.get(f"p{i}_activity", "empty").lower().strip()

        try:
            x = float(kv.get(f"p{i}_x", 0.0))
        except (ValueError, TypeError):
            x = 0.0

        try:
            y = float(kv.get(f"p{i}_y", 0.0))
        except (ValueError, TypeError):
            y = 0.0

        persons.append({"activity": activity, "x": x, "y": y})

    return {
        "num_persons": num_persons,
        "persons":     persons,
        "subject":     kv.get("subject", "").strip(),
        "note":        kv.get("note",    "").strip(),
    }


# ============================================================
# START DATASET SESSION
# ============================================================

def start_session(parsed_meta):
    """
    Open rx1.csv, rx2.csv, write metadata.json,
    and set active = True.
    """
    global active, session_dir, metadata
    global files, writers, session_number

    with state_lock:

        if active:
            print("[WARNING] Session already active — ignoring duplicate START.")
            return

        # ── Create directory ────────────────────────────
        session_number, session_dir = next_session_dir()
        os.makedirs(session_dir, exist_ok=True)

        # ── Build metadata dict ─────────────────────────
        metadata = {
            "session_id":   f"session_{session_number:04d}",
            "created_utc":  datetime.now(timezone.utc).isoformat(),
            "num_persons":  parsed_meta["num_persons"],
            "persons":      parsed_meta["persons"],
            "subject":      parsed_meta["subject"],
            "note":         parsed_meta["note"],
            "rx1_port":     RX1_PORT,
            "rx2_port":     RX2_PORT,
            "baud":         BAUD,
            "data_format":  "Raw CSI I/Q",
            "csi_format":   "I0,Q0,I1,Q1,...",
            "csv_structure":"One row per received CSI packet",
            "files":        ["rx1.csv", "rx2.csv", "metadata.json"],
        }

        # ── Open CSV files ──────────────────────────────
        rx1_file = open(
            os.path.join(session_dir, "rx1.csv"),
            "w", newline="", encoding="utf-8"
        )
        rx2_file = open(
            os.path.join(session_dir, "rx2.csv"),
            "w", newline="", encoding="utf-8"
        )

        files   = {"rx1": rx1_file, "rx2": rx2_file}
        writers = {"rx1": None,     "rx2": None}

        # CSV writers are created lazily on the first CSI packet
        # because we don't know the column count until then.

        # ── Write metadata ──────────────────────────────
        meta_path = os.path.join(session_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2)

        # ── Activate ────────────────────────────────────
        active = True

        # ── Console summary ─────────────────────────────
        print()
        print("=" * 56)
        print(f"[SESSION START]  {metadata['session_id']}")
        print(f"  Persons  : {parsed_meta['num_persons']}")
        for i, p in enumerate(parsed_meta["persons"]):
            print(
                f"  Person {i} : {p['activity']}"
                f"  @  ({p['x']}, {p['y']})"
            )
        print(f"  Subject  : {parsed_meta['subject']}")
        print(f"  Note     : {parsed_meta['note']}")
        print(f"  Dir      : {session_dir}")
        print("=" * 56)
        print()


# ============================================================
# ENSURE CSV WRITER EXISTS
# ============================================================

def ensure_writer(rx_name, csi_value_count):
    """
    Create the DictWriter for rx_name on the first CSI packet.
    We defer this because csi_value_count is not known until
    the first packet arrives.
    """
    global writers

    if writers.get(rx_name) is not None:
        return   # already initialised

    fh         = files[rx_name]
    csi_fields = make_csi_fields(csi_value_count)
    fieldnames = BASE_CSV_FIELDS + csi_fields

    writer = csv.DictWriter(
        fh,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )
    writer.writeheader()
    fh.flush()

    writers[rx_name] = writer

    print(
        f"  [{rx_name.upper()}] CSV header written "
        f"({csi_value_count} CSI values, "
        f"{csi_value_count // 2} subcarriers)"
    )


# ============================================================
# WRITE ONE CSI LINE
# ============================================================

def write_rx_line(line):
    """
    Parse and write one CSI serial line from RX1 or RX2.
    Only runs when a dataset session is active.

    Expected format:
        RX1,<ts_ms>,<rssi>,<seq>,<ch>,<bw>,<len>,I0,Q0,I1,Q1,...
        RX2,<ts_ms>,<rssi>,<seq>,<ch>,<bw>,<len>,I0,Q0,I1,Q1,...
    """
    # ── Server inference buffer ─────────────────────────────
    # Runs regardless of whether a dataset session is active.
    if server_collecting_now:
        with server_buffer_lock:
            host_ts = datetime.now(timezone.utc).isoformat()
            if line.startswith("RX1,"):
                server_rx1_lines.append((host_ts, line))
            elif line.startswith("RX2,"):
                server_rx2_lines.append((host_ts, line))

    with state_lock:

        if not active:
            return

        parts = line.rstrip("\r\n").split(",")
        if len(parts) < 8:
            return

        # ── Receiver name ───────────────────────────────
        rx_name = parts[0].lower()
        if rx_name not in ("rx1", "rx2"):
            return

        # ── Header fields ───────────────────────────────
        timestamp        = parts[1]
        rssi             = parts[2]
        sequence         = parts[3]
        channel          = parts[4]
        bandwidth        = parts[5]
        csi_length_text  = parts[6]

        # ── CSI values ──────────────────────────────────
        raw_csi    = parts[7:]
        csi_values = []
        for v in raw_csi:
            try:
                csi_values.append(int(v))
            except ValueError:
                continue

        try:
            csi_length = int(csi_length_text)
        except ValueError:
            csi_length = len(csi_values)

        if not csi_values:
            return

        # ── Lazy writer init ────────────────────────────
        ensure_writer(rx_name, len(csi_values))
        writer = writers[rx_name]

        # ── Build row dict ──────────────────────────────
        row = {
            "host_timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "rx_timestamp_ms":    timestamp,
            "rssi":               rssi,
            "sequence":           sequence,
            "channel":            channel,
            "bandwidth":          bandwidth,
            "csi_length":         csi_length,
        }

        for idx, val in enumerate(csi_values):
            sc  = idx // 2
            col = f"SC{sc}_{'I' if idx % 2 == 0 else 'Q'}"
            row[col] = val

        # ── Write + flush ───────────────────────────────
        writer.writerow(row)
        files[rx_name].flush()


# ============================================================
# STOP DATASET SESSION
# ============================================================

def stop_session():
    global active, session_dir, metadata, files, writers

    with state_lock:

        if not active:
            return

        # ── Flush and close CSV files ───────────────────
        for fh in files.values():
            try:
                fh.flush()
            except Exception:
                pass
        for fh in files.values():
            try:
                fh.close()
            except Exception:
                pass

        # ── Update metadata with stop timestamp ─────────
        if metadata is not None:
            metadata["stopped_utc"] = datetime.now(timezone.utc).isoformat()
            meta_path = os.path.join(session_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(metadata, mf, indent=2)

        # ── Console summary ─────────────────────────────
        print()
        print("=" * 56)
        if metadata:
            print(f"[SESSION STOP]   {metadata['session_id']}")
        print(f"  Saved to : {session_dir}")
        print("=" * 56)
        print()

        # ── Reset state ─────────────────────────────────
        active       = False
        session_dir  = None
        metadata     = None
        files        = {}
        writers      = {}


# ============================================================
# RUN EXTERNAL PYTHON SCRIPT
# ============================================================

def run_script(script_path, label):
    """
    Launch a Python script in a NEW console window (Windows).
    Waits for the script to finish before returning.
    Returns True if exit code is 0.
    """
    print()
    print("=" * 56)
    print(f"[PIPELINE] {label}")
    print(f"           {script_path}")
    print("=" * 56)
    print()

    if not os.path.exists(script_path):
        print(f"  [ERROR] File not found: {script_path}")
        return False

    # On Windows open a new console so the script's output is
    # visible without competing with collector.py's output.
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0

    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            creationflags=flags,
        )
    except Exception as e:
        print(f"  [ERROR] Could not start process: {e}")
        return False

    return_code = proc.wait()

    print()
    print(f"  [PIPELINE] {label} → return code {return_code}")

    return return_code == 0


# ============================================================
# SERIAL PAUSE / RESUME
# ============================================================

def close_serial_connections():
    """
    Release COM9 and COM11 before weightloader.py opens COM11.
    weightloader.py needs exclusive access to send the model binary.
    """
    global rx1_serial, rx2_serial

    print()
    print("[SERIAL] Releasing COM ports for weightloader.py ...")

    serial_pause_event.set()
    time.sleep(0.5)   # give reader threads time to notice

    with serial_lock:
        for port_name, ser in [("COM9", rx1_serial), ("COM11", rx2_serial)]:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
        rx1_serial = None
        rx2_serial = None

    print("[SERIAL] COM9 and COM11 released.")


def reopen_serial_connections():
    """
    Signal reader threads to reconnect after weightloader finishes.
    The threads themselves handle re-opening the port.
    """
    serial_pause_event.clear()
    print("[SERIAL] Serial readers will reconnect automatically.")


# ============================================================
# TRAINING PIPELINE
# ============================================================

def run_training_pipeline():
    """
    Runs on a background daemon thread.
    Sequence:
        1. preprocess_csi_v2.py   (new session-aware preprocessing)
        2. train.py               (multi-task TEDNet)
        3. weightloader.py        (float32 export + upload to RX2)
    """
    global training_pipeline_running

    with training_lock:
        if training_pipeline_running:
            print("[TRAINING] Pipeline already running.")
            return
        training_pipeline_running = True

    try:

        print()
        print("#" * 56)
        print("#         AUTOMATIC TRAINING PIPELINE          #")
        print("#" * 56)
        print()

        # ── Guard: no active collection ─────────────────
        with state_lock:
            if active:
                print("[TRAINING ERROR] Stop dataset collection first.")
                return

        # ── Step 1: Preprocess ──────────────────────────
        print("[PIPELINE] STEP 1 / 3 — preprocess_csi_v2.py")

        ok = run_script(PREPROCESS_SCRIPT, "CSI PREPROCESSING (multi-person)")

        if not ok:
            print("[TRAINING ERROR] Preprocessing failed — aborting.")
            return

        x_train_path = os.path.join(PROCESSED_DIR, "X_train.npy")
        if not os.path.exists(x_train_path):
            print(
                f"[TRAINING ERROR] X_train.npy not found at:\n"
                f"  {x_train_path}\n"
                f"  Preprocessing must have failed silently."
            )
            return

        print("[PIPELINE] Preprocessed data confirmed.")

        # ── Step 2: Train ───────────────────────────────
        print("[PIPELINE] STEP 2 / 3 — train.py")

        run_script(TRAIN_SCRIPT, "PYTORCH MULTI-TASK TRAINING")

        # Even a non-zero exit code is recoverable if a
        # checkpoint was written before training crashed.
        if not os.path.exists(BEST_MODEL_PATH):
            print(
                f"[TRAINING ERROR] No model checkpoint found at:\n"
                f"  {BEST_MODEL_PATH}"
            )
            return

        print(f"[PIPELINE] Best model confirmed: {BEST_MODEL_PATH}")

        # ── Step 3: Weight loader ───────────────────────
        print("[PIPELINE] STEP 3 / 3 — weightloader.py")
        print("[PIPELINE] Releasing serial ports ...")

        close_serial_connections()
        time.sleep(1.0)   # ensure Windows has fully released the ports

        ok = run_script(WEIGHTLOADER_SCRIPT, "FLOAT32 WEIGHT LOADER → RX2")

        # Always reopen serial regardless of upload result
        reopen_serial_connections()
        time.sleep(1.0)

        # ── Result ──────────────────────────────────────
        print()
        if ok:
            print("#" * 56)
            print("#         TRAINING PIPELINE COMPLETE         #")
            print("#" * 56)
            print()
            print("  Model uploaded to RX2.")
        else:
            print("#" * 56)
            print("#    WEIGHT UPLOAD FAILED / INCOMPLETE        #")
            print("#" * 56)
            print()
            print(f"  Binary remains on laptop: {BEST_MODEL_PATH}")
            print("  Re-run weightloader.py manually to retry upload.")

    except Exception as e:
        print()
        print(f"[PIPELINE EXCEPTION] {e}")
        # Make sure serial is always restored
        try:
            reopen_serial_connections()
        except Exception:
            pass

    finally:
        with training_lock:
            training_pipeline_running = False
        print()
        print("[TRAINING] Manager is ready for next pipeline run.")


def start_training_pipeline():
    with training_lock:
        if training_pipeline_running:
            print("[TRAINING] Already running — ignoring duplicate trigger.")
            return

    t = threading.Thread(
        target=run_training_pipeline,
        daemon=True
    )
    t.start()




# ============================================================
# SERVER INFERENCE
# ============================================================

def send_server_result_to_rx2(json_str):
    """Write SERVER_RESULT:<json> to COM11 so RX2 can display it."""
    global rx2_serial
    try:
        with serial_lock:
            ser = rx2_serial
        if ser is None:
            print("[SERVER] Cannot send result — COM11 not open.")
            return
        msg = f"SERVER_RESULT:{json_str}\n"
        ser.write(msg.encode("utf-8"))
        ser.flush()
        print(f"[SERVER] Result sent to RX2.")
    except Exception as e:
        print(f"[SERVER] Failed to send result: {e}")


def server_inference_loop():
    """
    Daemon thread. Repeatedly:
      1. Collects SERVER_COLLECT_SECONDS of RX1+RX2 lines
      2. Writes them to temp CSV files
      3. Runs test_on_server.py as subprocess
      4. Sends JSON result back to RX2 via COM11
    """
    global server_test_active, server_collecting_now
    global server_rx1_lines, server_rx2_lines, last_server_result

    print("[SERVER] Inference loop started.")

    while server_test_active:

        # ── Collect window ───────────────────────────────
        with server_buffer_lock:
            server_rx1_lines = []
            server_rx2_lines = []
        server_collecting_now = True

        print(f"[SERVER] Collecting {SERVER_COLLECT_SECONDS}s of CSI...")
        time.sleep(SERVER_COLLECT_SECONDS)
        server_collecting_now = False

        if not server_test_active:
            break

        # ── Snapshot ─────────────────────────────────────
        with server_buffer_lock:
            rx1_snap = list(server_rx1_lines)
            rx2_snap = list(server_rx2_lines)

        print(f"[SERVER] RX1={len(rx1_snap)} lines  RX2={len(rx2_snap)} lines")

        if len(rx1_snap) < 30 or len(rx2_snap) < 30:
            print("[SERVER] Not enough frames — skipping cycle.")
            continue

        # ── Write temp CSVs ──────────────────────────────
        def write_temp(path, lines, prefix):
            rows = []
            for item in lines:
                host_ts, ln = item   # unpack (timestamp, line) tuple
                parts = ln.rstrip("\r\n").split(",")
                if len(parts) < 8:
                    continue
                csi_values = []
                for v in parts[7:]:
                    try:
                        csi_values.append(int(v))
                    except ValueError:
                        continue
                if not csi_values:
                    continue
                row = {
                    "host_timestamp_iso": host_ts,   # ← actual receipt time
                    "rx_timestamp_ms":    parts[1],
                    "rssi":               parts[2],
                    "sequence":           parts[3],
                    "channel":            parts[4],
                    "bandwidth":          parts[5],
                    "csi_length":         len(csi_values),
                }
                for idx, val in enumerate(csi_values):
                    sc  = idx // 2
                    col = f"SC{sc}_{'I' if idx % 2 == 0 else 'Q'}"
                    row[col] = val
                rows.append(row)

            if not rows:
                return False

            import pandas as pd
            pd.DataFrame(rows).to_csv(path, index=False)
            return True

        ok1 = write_temp(TEMP_RX1, rx1_snap, "RX1")
        ok2 = write_temp(TEMP_RX2, rx2_snap, "RX2")

        if not ok1 or not ok2:
            print("[SERVER] Failed to write temp CSVs.")
            continue

        # ── Run test_on_server.py ────────────────────────
        if not os.path.exists(TEST_ON_SERVER_SCRIPT):
            print(f"[SERVER] test_on_server.py not found.")
            continue

        print("[SERVER] Running test_on_server.py ...")
        try:
            proc = subprocess.Popen(
                [sys.executable, TEST_ON_SERVER_SCRIPT,
                 TEMP_RX1, TEMP_RX2],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=PROJECT_ROOT,
            )
            stdout, stderr = proc.communicate(timeout=60)

            if stderr:
                err_txt = stderr.decode(errors="ignore").strip()
                if err_txt:
                    print(f"[SERVER] stderr: {err_txt[:200]}")

            result_line = stdout.decode(errors="ignore").strip()
            if not result_line:
                print("[SERVER] No output from test_on_server.py")
                continue

            # Validate JSON
            json.loads(result_line)

            last_server_result = result_line
            print(f"[SERVER] Result: {result_line[:120]}")

            send_server_result_to_rx2(result_line)

        except subprocess.TimeoutExpired:
            proc.kill()
            print("[SERVER] test_on_server.py timed out (60s).")
        except json.JSONDecodeError:
            print(f"[SERVER] Invalid JSON from test_on_server.py: {result_line[:100]}")
        except Exception as e:
            print(f"[SERVER] Error: {e}")

    print("[SERVER] Inference loop stopped.")


def start_server_test():
    global server_test_active
    with server_test_lock:
        if server_test_active:
            print("[SERVER] Already running.")
            return
        server_test_active = True
    t = threading.Thread(target=server_inference_loop, daemon=True)
    t.start()
    print("[SERVER] Started.")


def stop_server_test():
    global server_test_active, server_collecting_now
    with server_test_lock:
        server_test_active    = False
        server_collecting_now = False
    print("[SERVER] Stopped.")


# ============================================================
# RX2 SERIAL READER THREAD
# ============================================================
# ============================================================
# RX2 SERIAL READER THREAD
# ============================================================

def rx2_reader_thread():
    global rx2_serial

    print(f"[RX2] Opening {RX2_PORT} ...")

    while True:

        # ── Pause during weightloader upload ────────────
        if serial_pause_event.is_set():
            time.sleep(0.2)
            continue

        # ── Open COM11 ───────────────────────────────────
        try:
            with serial_lock:
                if rx2_serial is None:
                    rx2_serial = serial.Serial()
                    rx2_serial.port     = RX2_PORT
                    rx2_serial.baudrate = BAUD
                    rx2_serial.timeout  = 1
                    rx2_serial.dtr = False
                    rx2_serial.rts = False
                    rx2_serial.open()
                    print(f"[RX2] Connected to {RX2_PORT}.")
        except Exception as e:
            print(f"[RX2 ERROR] Cannot open {RX2_PORT}: {e}")
            time.sleep(2)
            continue

        # ── Read one line ────────────────────────────────
        try:
            with serial_lock:
                ser = rx2_serial
            if ser is None:
                continue

            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

                        # ── SERVER_TEST_START ─────────────────────────
            if line == "SERVER_TEST_START":
                print("[RX2] SERVER_TEST_START received.")
                start_server_test()

            # ── SERVER_TEST_STOP ──────────────────────────
            elif line == "SERVER_TEST_STOP":
                print("[RX2] SERVER_TEST_STOP received.")
                stop_server_test()

            # ── SESSION_START ────────────────────────────
            elif line.startswith("SESSION_START,"):
                raw    = line[len("SESSION_START,"):]
                parsed = parse_session_metadata(raw)
                start_session(parsed)

            # ── SESSION_STOP ─────────────────────────────
            elif line == "SESSION_STOP":
                stop_session()

            # ── TRAINING_START ───────────────────────────
            elif line == "TRAINING_START":
                print()
                print("[RX2] TRAINING_START received.")
                print("[RX2] Launching training pipeline ...")
                start_training_pipeline()

            # ── RX2 CSI data ─────────────────────────────
            elif line.startswith("RX2,"):
                write_rx_line(line)

            # ── Status / model messages ───────────────────
            elif any(line.startswith(prefix) for prefix in (
                "MODEL_",
                "PSRAM_",
                "TEDNET_",
                "NORMALIZATION:",
                "FLASH_",
                "PARSER_",
                "INFERENCE_",
                "RX2_READY",
                "RX1_READY",
                "CSI_",
                "HTTP_",
                "LittleFS",
                "TRAINING_STATUS:",
                "TEST_RESULT:",
            )):
                print(f"[RX2] {line}")

        except Exception as e:
            if serial_pause_event.is_set():
                time.sleep(0.2)
                continue

            print(f"[RX2 READ ERROR] {e}")

            with serial_lock:
                try:
                    if rx2_serial is not None:
                        rx2_serial.close()
                except Exception:
                    pass
                rx2_serial = None

            time.sleep(1)


# ============================================================
# RX1 SERIAL READER THREAD
# ============================================================

def rx1_reader_thread():
    global rx1_serial

    print(f"[RX1] Opening {RX1_PORT} ...")

    while True:

        # ── Pause during weightloader upload ────────────
        if serial_pause_event.is_set():
            time.sleep(0.2)
            continue

        # ── Open COM9 ────────────────────────────────────
        try:
            with serial_lock:
                if rx1_serial is None:
                    rx1_serial = serial.Serial()
                    rx1_serial.port     = RX1_PORT
                    rx1_serial.baudrate = BAUD
                    rx1_serial.timeout  = 1
                    rx1_serial.dtr = False
                    rx1_serial.rts = False
                    rx1_serial.open()
                    print(f"[RX1] Connected to {RX1_PORT}.")
        except Exception as e:
            print(f"[RX1 ERROR] Cannot open {RX1_PORT}: {e}")
            time.sleep(2)
            continue

        # ── Read one line ────────────────────────────────
        try:
            with serial_lock:
                ser = rx1_serial
            if ser is None:
                continue

            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            # ── RX1 CSI data ──────────────────────────────
            if line.startswith("RX1,"):
                write_rx_line(line)

            # ── RX1 status messages ───────────────────────
            elif any(line.startswith(prefix) for prefix in (
                "RX1_READY",
                "CSI_",
                "WIFI_",
                "STATIC_IP",
                "UDP_",
                "--- RX1",
            )):
                print(f"[RX1] {line}")

        except Exception as e:
            if serial_pause_event.is_set():
                time.sleep(0.2)
                continue

            print(f"[RX1 READ ERROR] {e}")

            with serial_lock:
                try:
                    if rx1_serial is not None:
                        rx1_serial.close()
                except Exception:
                    pass
                rx1_serial = None

            time.sleep(1)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 56)
    print("  Wi-Fi CSI HAR  COLLECTOR + TRAINING MANAGER")
    print("=" * 56)
    print(f"  RX1 port        : {RX1_PORT}")
    print(f"  RX2 port        : {RX2_PORT}")
    print(f"  Baud            : {BAUD}")
    print(f"  Project root    : {PROJECT_ROOT}")
    print(f"  Dataset         : {DATASET_ROOT}")
    print(f"  Preprocess      : {PREPROCESS_SCRIPT}")
    print(f"  Train           : {TRAIN_SCRIPT}")
    print(f"  Weight loader   : {WEIGHTLOADER_SCRIPT}")
    print(f"  Best model      : {BEST_MODEL_PATH}")
    print("=" * 56)
    print()

    # ── Create required directories ──────────────────────
    for d in [DATASET_ROOT, PROCESSED_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Start serial reader threads ──────────────────────
    t_rx1 = threading.Thread(target=rx1_reader_thread, daemon=True)
    t_rx2 = threading.Thread(target=rx2_reader_thread, daemon=True)

    t_rx1.start()
    t_rx2.start()

    print("[MANAGER] RX1 reader thread started.")
    print("[MANAGER] RX2 reader thread started.")
    print()
    print("Waiting for RX2 ...")
    print()

    # ── Keep main thread alive ───────────────────────────
    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print()
        print("[MANAGER] Ctrl+C — stopping collector ...")

        stop_session()

        serial_pause_event.set()

        with serial_lock:
            for ser in [rx1_serial, rx2_serial]:
                try:
                    if ser is not None:
                        ser.close()
                except Exception:
                    pass

        print("[MANAGER] Stopped.")
