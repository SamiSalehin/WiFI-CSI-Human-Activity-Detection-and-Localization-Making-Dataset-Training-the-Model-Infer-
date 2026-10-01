# WiFI-CSI-Human-Activity-Detection-and-Localization-Making-Dataset-Training-the-Model-Infer-




> **Contactless human activity recognition and indoor localization using Wi-Fi Channel State Information — no cameras, no wearables, no cloud.**

A complete embedded ML system that detects what people are doing and where they are in a room by measuring how their bodies distort Wi-Fi radio signals. The neural network runs entirely on an ESP32-S3 microcontroller with no internet connection required during inference.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Data Collection](#data-collection)
- [Training](#training)
- [On-Device Deployment](#on-device-deployment)
- [Inference Modes](#inference-modes)
- [Model Architecture](#model-architecture)
- [Configuration Reference](#configuration-reference)
- [Activity Classes](#activity-classes)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Team](#team)

---

## Overview

This system uses three ESP32 boards and a laptop to build a real-time human activity recognition (HAR) and localization pipeline. Two receiver boards capture Channel State Information (CSI) from a transmitter board firing UDP packets at 50 Hz. The raw complex I/Q measurements from 192 OFDM subcarriers per receiver encode how the room's radio environment changes when people move through it.

A multi-task neural network — **TEDNet** (Temporal Encoder with Dual-receiver Network) — processes 0.6-second windows of combined CSI amplitude from both receivers and simultaneously predicts:

- **Activity** — which of 7 activities are happening in the scene (multi-hot, supports multiple simultaneous persons)
- **Count** — how many persons are present (0–4)
- **Location** — estimated (X, Y) coordinates in metres for each detected person

The trained model is serialized to a custom float32 binary format and uploaded to the ESP32-S3 over USB serial where it runs entirely in PSRAM, producing inference results every \~1 second.

---

## System Architecture

```
┌─────────────────┐     802.11n HT      ┌─────────────────┐
│   TX WROOM      │ ──────────────────► │     RX1 S3      │
│  (ESP32-WROOM)  │                     │   (ESP32-S3)    │
│                 │                     │                 │
│  AP: CSI_TX_    │ ──────────────────► │  CSI → COM9 →  │
│  NETWORK ch.6   │  50 Hz UDP frames   │  collector.py  │
│  192.168.4.1    │                     └────────┬────────┘
└─────────────────┘                              │ UDP amplitudes
                                                 │ (wireless, port 4444)
┌──────────────────────────────┐      ┌──────────▼────────┐
│           Laptop             │      │     RX2 S3        │
│                              │      │   (ESP32-S3)      │
│  collector.py  ◄──── COM11 ──┤◄─────│                   │
│  preprocess.py               │      │  CSI capture      │
│  train.py                    │      │  HTML control     │
│  weightloader.py ──── COM11 ►├─────►│  TEDNet inference │
│  test_on_server.py           │      │  192.168.4.3      │
└──────────────────────────────┘      └───────────────────┘
          ▲
          │ HTTP (Wi-Fi: CSI_CONTROL)
┌─────────┴──────┐
│  Phone/Tablet  │
│  Browser UI    │
│  192.168.4.1   │
└────────────────┘
```

### Wi-Fi Networks

| Network | SSID | Purpose | AP |
| --- | --- | --- | --- |
| CSI capture | `CSI_TX_NETWORK` | TX fires frames; RX1/RX2 capture CSI | TX WROOM |
| Control | `CSI_CONTROL` | Browser UI for session control and live results | RX2 S3 |

### IP Addresses

| Board | IP | Role |
| --- | --- | --- |
| TX WROOM | 192.168.4.1 | AP gateway |
| RX1 S3 | 192.168.4.2 | CSI receiver 1 |
| RX2 S3 | 192.168.4.3 | CSI receiver 2 + inference engine |
| RX2 S3 (AP) | 192.168.4.1 | HTML control interface |

---

## Hardware Requirements

| Component | Quantity | Notes |
| --- | --- | --- |
| ESP32-WROOM-32 DevKit | 1 | Transmitter (TX) |
| ESP32-S3 DevKitC | 2 | Receivers (RX1, RX2) — must have OPI PSRAM (8 MB) |
| USB cables | 3 | Data-capable (not charge-only) |
| PC / Laptop | 1 | Windows/Linux/macOS, Python 3.9+ |
| Android/iOS device | 1 | For browser control UI (optional, laptop browser works too) |

> **Important:** RX2 requires an ESP32-S3 variant with **8 MB OPI PSRAM**. The \~4 MB float32 model cannot fit in standard SRAM. Verify your board's PSRAM in the Arduino IDE board settings.

### Physical Setup

- Fix all three boards in a **triangle formation** with measured positions
- Mark TX as origin (0, 0)
- Measure RX1 and RX2 positions in metres from TX
- Mark grid points on the floor where subjects will stand/sit — measure and record (X, Y) coordinates for each
- Keep boards stationary throughout all data collection and testing

---

## Software Requirements

### Arduino IDE

| Library / Board | Version | Notes |
| --- | --- | --- |
| ESP32 Arduino Core | ≥ 2.0.14 | Espressif board package |
| Arduino IDE | ≥ 2.x | Or PlatformIO |

### Python (Laptop)

```bash
pip install torch torchvision numpy pandas pyserial tqdm scikit-learn
```

| Package | Purpose |
| --- | --- |
| `torch` | Model training and server inference |
| `numpy`, `pandas` | Data processing |
| `pyserial` | Serial communication with ESP32 boards |
| `tqdm` | Training progress bar |
| `scikit-learn` | Metrics (F1, confusion matrix) |

---

## Repository Structure

```
project_root/
│
├── firmware/
│   ├── TX_WROOM/
│   │   └── TX_WROOM.ino          ← Upload to ESP32-WROOM (transmitter)
│   ├── RX1_S3/
│   │   └── RX1_S3.ino            ← Upload to ESP32-S3 #1
│   └── RX2_S3_HOTSPOT/
│       └── RX2_S3_HOTSPOT.ino    ← Upload to ESP32-S3 #2
│
├── python/
│   └── collector.py              ← Run this on laptop during collection
│
├── preprocess_csi_v2.py          ← Step 1: raw CSV → training arrays
├── train.py                      ← Step 2: train TEDNet
├── weightloader.py               ← Step 3: export + upload model to RX2
├── test_on_server.py             ← Server-side inference (laptop GPU/CPU)
│
├── dataset/                      ← Created by collector.py
│   └── session_XXXX/
│       ├── rx1.csv
│       ├── rx2.csv
│       └── metadata.json
│
├── processed/                    ← Created by preprocess_csi_v2.py
│   ├── X_train.npy
│   ├── Y_activity_train.npy
│   ├── Y_count_train.npy
│   ├── Y_location_train.npy
│   ├── X_val.npy
│   ├── ...
│   ├── feature_mean.npy
│   ├── feature_std.npy
│   └── label_map.json
│
├── models/                       ← Created by train.py / weightloader.py
│   ├── tednet_csi_har_best.pth   ← PyTorch checkpoint
│   └── model_fp32.bin            ← Binary for RX2 (CSI2 v3 format)
│
└── README.md
```

---

## Quick Start

### 1 — Flash Firmware

Open each `.ino` file in Arduino IDE and upload to the corresponding board:

| File | Board | COM port |
| --- | --- | --- |
| `TX_WROOM.ino` | ESP32-WROOM | any |
| `RX1_S3.ino` | ESP32-S3 #1 | COM9 (adjust in collector.py if different) |
| `RX2_S3_HOTSPOT.ino` | ESP32-S3 #2 | COM11 (adjust in collector.py if different) |

> For RX2, set the Arduino partition scheme to one that includes **LittleFS with ≥ 4 MB flash** (e.g. "16MB Flash (3MB APP/9MB FATFS)" or similar, depending on your board).

### 2 — Verify Hardware

Power all three boards. Open Serial Monitor on RX2 (COM11, 115200 baud). You should see:

```
RX2 ESP32-S3 CSI + INFERENCE
PSRAM_TOTAL:8388608
PSRAM_MODEL_BUFFER:5242880
LittleFS_OK
TX_WIFI_CONNECTED
RX2_STA_IP:192.168.4.3
HTTP_SERVER_READY
CSI_ENABLED
RX2_READY
```

### 3 — Start collector.py

```bash
python python/collector.py
```

Leave this running for the entire data collection and training session.

### 4 — Connect to Control UI

On your phone or laptop, connect to Wi-Fi network `CSI_CONTROL` (password: `csi_control_123`).

Open browser: `http://192.168.4.1`

---

## Data Collection

### Session Workflow

1. In the browser UI, select:
   - Number of persons (0–4)
   - Activity and (X, Y) position for each person
   - Subject ID
2. Press **START SESSION** — wait 2 seconds before subjects move into position
3. Hold the activity for 20–30 seconds
4. Press **STOP SESSION** — wait 1 second before subjects move

Each session is saved to `dataset/session_XXXX/` with `rx1.csv`, `rx2.csv`, and `metadata.json`.

### Recommended Data Collection per Class

| Activity | Minimum sessions | Notes |
| --- | --- | --- |
| empty | 5 | Leave room completely |
| sitting | 8 per position | Stay still on chair |
| standing | 8 per position | Stand still, arms at sides |
| walking | 15 | Fixed 2m path, consistent pace |
| running | 10 | Jogging in place or short path |
| lying | 8 per position | Still on mat |
| falling | 12 | Slow controlled fall onto crash mat |

**Collect from at least 3 different subjects** across different days and positions. Body size significantly affects CSI amplitude.

### Pre-Session Checklist

```
□ TX WROOM powered (CSI_TX_NETWORK visible)
□ collector.py running, COM9 + COM11 connected
□ No active session in status bar
□ Subject in position at marked grid point
□ Door closed, no one else in room
□ Activity and X/Y entered correctly
□ Waited 2 seconds after START before subject moves
```

---

## Training

### Step 1 — Preprocess

```bash
python preprocess_csi_v2.py
```

Reads all sessions from `dataset/`, synchronizes RX1 and RX2 frames by laptop timestamp, extracts amplitudes, creates sliding windows (size=30, step=3), performs session-level 80/20 train/val split, normalizes, and saves NumPy arrays to `processed/`.

### Step 2 — Train

```bash
python train.py
```

Trains TEDNet for 60 epochs. Per-epoch output when validation data exists:

```
Ep 001/060  train=1.2341  val=1.1892  act_F1=0.124  cnt_acc=0.412  loc_RMSE=1.832  lr=1.00e-03
  → BEST SAVED  (metric=0.1240)
```

Best checkpoint saved to `models/tednet_csi_har_best.pth` based on validation macro F1.

### Step 3 — Upload Model to RX2

```bash
python weightloader.py
```

Exports all tensors from the checkpoint to `models/model_fp32.bin` (CSI2 v3 binary format), then uploads to RX2 over COM11 at 115200 baud (\~6 minutes). The model is parsed and saved to LittleFS flash — it survives reboots automatically.

### Automated Pipeline

All three steps can be triggered from the browser UI by pressing the **TRAIN** button. collector.py runs them sequentially in separate console windows.

---

## Inference Modes

### On-Device Testing (RX2 ESP32-S3)

Press **START TESTING** in the browser UI.

- RX2 collects 30 synchronized frames from RX1 (via UDP) and its own CSI
- Runs full TEDNet forward pass in PSRAM (\~300–500 ms on ESP32-S3 @ 240 MHz)
- Results displayed on browser every \~1 second:

```
PERSONS   : 1
ACTIVITIES: standing
Person 0  : X=1.50 m  Y=1.00 m
CONFIDENCE: 87.4%
RX1 packets: 1250   rejects: 12
```

### Server Testing (Laptop)

Press **START TEST ON SERVER** (or trigger via HTTP endpoint).

- collector.py buffers 5 seconds of CSI from both RX1 and RX2
- Runs `test_on_server.py` as a subprocess — full PyTorch inference on laptop CPU/GPU
- Result transmitted back to RX2 and displayed on browser
- Cycle time \~15 seconds; best used during model development when architecture changes frequently

---

## Model Architecture

**TEDNet** — Temporal Encoder with Dual-receiver Network

```
Input: (Batch, 30, 384)
  30 = time steps (0.6 seconds at 50 Hz)
 384 = features (192 RX1 amplitudes ++ 192 RX2 amplitudes)

Shared Backbone:
  Conv1d(384→128, k=3, pad=1) → BatchNorm → GELU
  Conv1d(128→128, k=3, pad=1) → BatchNorm → GELU
  + Learnable positional embedding (30 × 128)
  4× TransformerEncoderLayer(d=128, heads=8, ff=512, GELU)
  Temporal mean pool → LayerNorm(128)
  Output: (Batch, 128)

Three Heads (parallel, shared backbone):
  Activity  : Linear(128→64) → GELU → Linear(64→7)   BCEWithLogitsLoss
  Count     : Linear(128→64) → GELU → Linear(64→5)   CrossEntropyLoss
  Location  : Linear(128→64) → GELU → Linear(64→8)   Masked MSELoss

Total parameters : ~1.02 million
Model size (fp32): ~4.1 MB
```

### Loss Function

```
L_total = 1.0 × L_activity + 0.5 × L_count + 0.3 × L_location
```

Location loss is masked — only occupied person slots contribute to the gradient.

### Hyperparameters

| Parameter | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Epochs | 60 |
| LR schedule | ReduceLROnPlateau (patience=6, factor=0.5) |
| Dropout | 0.1 |
| Window size | 30 frames |
| Window step | 3 frames |

---

## Configuration Reference

### Network (must match across all firmware files)

| Parameter | Value |
| --- | --- |
| TX AP SSID | `CSI_TX_NETWORK` |
| TX AP Password | `csi_tx_12345678` |
| TX AP Channel | 6 (2.437 GHz) |
| TX WROOM MAC | `84:0D:8E:E8:31:29` |
| RX1 static IP | 192.168.4.2 |
| RX2 static IP | 192.168.4.3 |
| TX UDP port | 3333 |
| RX1→RX2 UDP port | 4444 |
| Control AP SSID | `CSI_CONTROL` |
| Control AP Password | `csi_control_123` |
| Serial COM (RX1) | COM9 |
| Serial COM (RX2) | COM11 |
| Baud rate | 115200 |

> **If your WROOM's MAC address is different**, update `WROOM_MAC` in both `RX1_S3.ino` and `RX2_S3_HOTSPOT.ino`. The MAC is printed to Serial on boot.

### CSI Signal

| Parameter | Value |
| --- | --- |
| TX rate | 50 Hz (20 ms interval) |
| TX power | \~20 dBm (default) |
| Band | 2.4 GHz HT20 |
| Subcarriers per receiver | 192 |
| Combined features | 384 |
| CSI fields captured | LLTF + HTLTF + STBC-HTLTF2 (merged) |

### PSRAM (RX2)

| Parameter | Value |
| --- | --- |
| Total PSRAM | 8 MB OPI |
| Model buffer | 5 MB (pre-allocated at boot, never freed) |
| Inference workspace peak | \~230 KB |

---

## Activity Classes

| Index | Name | Description |
| --- | --- | --- |
| 0 | empty | No person in the scene |
| 1 | falling | Person falling (use crash mat for safety) |
| 2 | lying | Person lying still on floor |
| 3 | running | Person running or jogging in place |
| 4 | sitting | Person sitting still on chair |
| 5 | standing | Person standing still |
| 6 | walking | Person walking through the room |

Activity output is **multi-hot** — multiple classes can be active simultaneously when multiple persons are present. The model detects which activities are happening in the scene, not which person is doing which activity.

---

## Results

Current results with sitting and standing data only (other classes pending data collection):

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| sitting | — | — | 0.90 |
| standing | — | — | 0.66 |
| empty / falling / lying / running / walking | — | — | 0.00 (no training data) |

**Count accuracy:** Preliminary (1-person scenes only)

**Location RMSE:** Not yet evaluated (insufficient position diversity)

> Results will be updated as more training data is collected across all 7 classes and multiple subjects.

---

## Known Limitations

### Hardware

- **Single antenna (SISO):** ESP32-WROOM has one antenna. MIMO spatial diversity is not available.
- **2.4 GHz only:** ESP32-WROOM does not support 5 GHz.
- **AP/STA channel coupling:** RX2 in WIFI_AP_STA mode forces the CSI_CONTROL AP to the same channel as TX_NETWORK (channel 6). This is an ESP32 hardware constraint.
- **LittleFS partition:** Must use a partition scheme allocating ≥ 4 MB for LittleFS. Check your board's flash size.

### Model

- **Scene-level activity only:** The model detects which activities are present but cannot associate a specific activity to a specific person when multiple persons are present.
- **Environment-specific:** The model learns the spatial CSI fingerprint of your specific room with your specific board placement. Changing board positions requires retraining from scratch.
- **Static activities generalize better than dynamic:** Walking, running, and falling require more data diversity (multiple subjects, directions, speeds) to achieve good accuracy.

### Software

- **COM port hardcoded:** `collector.py` and `weightloader.py` default to COM9 (RX1) and COM11 (RX2). Change `RX1_PORT` and `RX2_PORT` constants if your ports differ.
- **Windows paths:** Tested primarily on Windows. On Linux/macOS, COM ports will be `/dev/ttyUSB0` style.

---

## Future Work

- [ ] Collect training data for all 7 activity classes across ≥ 3 subjects

- [ ] Per-person activity head — link specific activity to specific person slot

- [ ] Increase window size from 30 to 50 frames for better dynamic activity detection

- [ ] TFLite Micro or ESP-DL deployment for architecture-independent on-device inference

- [ ] INT8 quantization to reduce model size to \~1 MB and eliminate PSRAM dependency

- [ ] Multi-room generalization study

- [ ] Real-time visualization dashboard for location tracking

---

## Serial Protocol Reference

### RX2 → Laptop (COM11)

| Message | When |
| --- | --- |
| `SESSION_START,num_persons=N,p0_activity=X,...` | User presses START SESSION |
| `SESSION_STOP` | User presses STOP SESSION |
| `TRAINING_START` | User presses TRAIN |
| `RX2,ts,rssi,seq,ch,bw,len,I0,Q0,...` | During data collection (one line per CSI frame) |
| `MODEL_READY` | Ready to receive model binary |
| `MODEL_LOADED` | Model parsed and saved successfully |
| `PERSONS:N` / `ACTIVITIES:...` | After each inference window |

### Laptop → RX2 (COM11, model upload)

```
MODEL_BEGIN,<size_bytes>,3\n
<raw binary bytes>
MODEL_END\n
```

### RX1 → RX2 (UDP port 4444, during testing)

```
RX1TEST,<seq>,<timestamp_ms>,<amp0>,<amp1>,...,<amp191>
```

---

## Team

**Group 02 — BUET EEE 416**

| Name | Student ID |
| --- | --- |
| Ekhlas Ibn Ali | 2106068 |
| MD. Sami Salehin Diep | 2106069 |
| Khalid Al Masfiq | 2106070 |
| Mohammad Rayhan Firoz | 2106071 |

---

## License

This project is released for academic purposes. Contact the authors for any other use.

---

**Wi-Fi CSI · ESP32-S3 · TEDNet · Multi-Task Learning · Embedded ML**
