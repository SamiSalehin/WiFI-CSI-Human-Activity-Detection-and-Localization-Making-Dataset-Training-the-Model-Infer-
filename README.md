# WiFI-CSI-Human-Activity-Detection-and-Localization-Making-Dataset-Training-the-Model-Infer-




 
> Contactless human activity recognition and indoor localization using Wi-Fi Channel State Information — no cameras, no wearables, no cloud.
 
[![License: Academic](https://img.shields.io/badge/License-Academic-blue.svg)]()
[![Platform: ESP32-S3](https://img.shields.io/badge/Platform-ESP32--S3-red.svg)]()
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)]()
 
---
 
## What This Does
 
This system detects what people are doing and where they are in a room by measuring how their bodies distort Wi-Fi radio signals — using only off-the-shelf ESP32 microcontrollers and a laptop.
 
No cameras. No wearables. No internet connection needed during inference.
 
The trained neural network runs entirely on an embedded microcontroller, producing real-time predictions every second.
 
---
 
## Capabilities
 
- **Activity Recognition** — detects 7 activities simultaneously across multiple people
- **Person Counting** — estimates how many people are in the room (0–4)
- **Indoor Localization** — estimates (X, Y) position in metres for each detected person
- **On-Device Inference** — full neural network runs on an ESP32-S3 microcontroller
- **Wireless Control** — browser-based UI for data collection, training triggers, and live results
---
 
## Activity Classes
 
| Index | Activity |
|---|---|
| 0 | Empty room |
| 1 | Falling |
| 2 | Lying |
| 3 | Running |
| 4 | Sitting |
| 5 | Standing |
| 6 | Walking |
 
Multiple activities can be detected simultaneously when multiple persons are present.
 
---
 
## System Overview
 
The system uses three ESP32 boards placed in a triangle formation inside a room.
 
A **transmitter board** fires Wi-Fi frames at 50 Hz. Two **receiver boards** measure how the room's radio environment changes as people move — this measurement is called Channel State Information (CSI). A laptop collects, processes, and trains on this data. The trained model is then deployed directly onto the second receiver board.
 
```
Transmitter ──── Wi-Fi ────► Receiver 1 ──┐
     │                                     │ (both feed into)
     └────── Wi-Fi ────────► Receiver 2 ◄─┘
                                  │
                             Neural Network
                             runs here on
                             the device itself
```
 
A phone or laptop connects to the device's Wi-Fi hotspot to control sessions and view live predictions through a browser.
 
---
 
## Model — TEDNet
 
**TEDNet** (Temporal Encoder with Dual-receiver Network) is a custom multi-task neural network designed for this system.
 
It processes 0.6-second windows of CSI amplitude from both receivers and produces three simultaneous predictions through a shared backbone with three task-specific output heads.
 
```
CSI Window (0.6 seconds, both receivers)
            ↓
    CNN Feature Extraction
            ↓
  Transformer Encoder (4 layers)
            ↓
       ┌────┴────┬──────────┐
       ↓         ↓          ↓
   Activity   Person     Location
  Detection   Count     Estimation
```
 
- ~1 million parameters
- ~4 MB float32 model
- ~1 second per inference on ESP32-S3
---
 
## Results
 
> Results reflect current data collection status. More classes are being collected.
 
| Activity | F1 Score |
|---|---|
| Sitting | 0.90 |
| Standing | 0.66 |
| Others | Pending data collection |
 
Static activities (sitting, standing) perform well. Dynamic activities (walking, running, falling) require more diverse training data across multiple subjects and directions.
 
---
 
## Repository Structure
 
```
├── firmware/          ← Arduino sketches for all three ESP32 boards
├── python/            ← Data collection script (runs on laptop)
├── preprocess_csi_v2.py
├── train.py
├── weightloader.py
├── test_on_server.py
├── dataset/           ← Created during data collection
├── processed/         ← Created during preprocessing
└── models/            ← Saved model checkpoints and deployment binary
```
 
---
 
## Requirements
 
### Hardware
- 1× ESP32-WROOM-32 (transmitter)
- 2× ESP32-S3 with 8 MB OPI PSRAM (receivers)
- 3× USB cables
- Laptop / PC
### Software
- Arduino IDE 2.x with ESP32 board package
- Python 3.9+
- PyTorch, NumPy, pandas, pyserial, scikit-learn, tqdm
---
 
## Getting Started
 
### 1. Flash the firmware
Upload each sketch in the `firmware/` folder to its corresponding board.
 
### 2. Start data collection
```bash
python python/collector.py
```
Connect to the device's Wi-Fi hotspot and open the browser control panel to label and record sessions.
 
### 3. Train the model
```bash
python preprocess_csi_v2.py
python train.py
python weightloader.py
```
Or trigger all three steps from the browser UI with one button press.
 
### 4. Run inference
Open the browser UI and press **START TESTING**. Live predictions appear within seconds.
 
---
 
## How It Works
 
Wi-Fi signals bounce off walls, furniture, and human bodies. When a person moves, these reflections change in a way that is unique to both the activity and the position of the person. By measuring the signal across 192 frequency subcarriers from two receivers simultaneously, the system captures a rich spatial and temporal fingerprint of the room.
 
The neural network learns to associate patterns in these fingerprints with specific activities and positions from labeled training data collected in the same room.
 
---
 
## Limitations
 
- The model is environment-specific — moving the boards requires retraining
- Activity detection is scene-level: the system knows *what* is happening but cannot yet link a specific activity to a specific person's location
- Dynamic activities (walking, falling) require significantly more training data diversity than static ones
---
 
## Future Work
 
- [ ] Collect training data for all 7 classes across multiple subjects
- [ ] Per-person activity-location association
- [ ] INT8 quantization for faster on-device inference
- [ ] TFLite Micro / ESP-DL deployment for architecture-independent inference
- [ ] Larger temporal window for improved dynamic activity recognition
---
 
## Team
 
**Group 02 — BUET EEE 416**
 
| Name | Student ID |
|---|---|
| Ekhlas Ibn Ali | 2106068 |
| MD. Sami Salehin Diep | 2106069 |
| Khalid Al Masfiq | 2106070 |
| Mohammad Rayhan Firoz | 2106071 |
 
---
 
<p align="center">
  <em>Built with ESP32-S3 · PyTorch · Wi-Fi CSI · Embedded ML</em>
</p>
