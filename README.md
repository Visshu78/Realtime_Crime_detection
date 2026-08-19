# 🚨 Intelligent Real-Time Crime Detection System Using CCTV Footage

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.19%2B-orange.svg)](https://pytorch.org/vision/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20H100%20PCIe-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An **AI-powered end-to-end intelligent surveillance system** designed to automatically detect, classify, and report criminal activity from live CCTV streams in real time. The system converts passive CCTV surveillance into an active threat detection, suspect identification, and automated incident reporting mechanism.

---

## 📌 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Key Features](#-key-features)
- [Model Benchmark & Performance](#-model-benchmark--performance)
- [Project Pipeline](#-project-pipeline)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Training](#-training)
- [Inference & Video Analysis](#-inference--video-analysis)
- [Live CCTV RTSP Integration](#-live-cctv-rtsp-integration)
- [Roadmap & Future Phases](#-roadmap--future-phases)

---

## 🧠 Architecture Overview

The system utilizes a **multi-stage hierarchical deep learning pipeline** to maximize detection accuracy while minimizing computational load:

```
                            ┌────────────────────────┐
                            │ CCTV Stream / Video    │ (15–30 FPS)
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │   Stage 1: VideoViT    │ ◄── Fast Binary Gate (>94.5% Acc)
                            │ (Spatial-Temporal ViT) │     (Filters normal footage continuously)
                            └───────────┬────────────┘
                                        │ (If Crime Detected: "YES")
                                        ▼
                            ┌────────────────────────┐
                            │  Stage 2: SlowFast Net │ ◄── Fine-Grained Crime Classification
                            │  (Multi-Class Actions) │     (Fight, Assault, Burglary, Arson, etc.)
                            └───────────┬────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
          ┌─────────────────────┐               ┌─────────────────────┐
          │  Stage 3A: YOLOv5   │               │ Stage 3B: RetinaFace│
          │ (Weapon & Object)   │               │     + ArcFace       │
          │ (Gun, Knife, Tools) │               │ (Suspect Face Match)│
          └──────────┬──────────┘               └──────────┬──────────┘
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        ▼
                            ┌────────────────────────┐
                            │  Stage 4: Auto-Report  │ ◄── Structured Evidence Dossier
                            │ (WHAT, HOW, WHO, TIME) │     (Instant Law Enforcement Alert)
                            └────────────────────────┘
```

---

## ⚙️ Key Features

- ✅ **Hierarchical Two-Stage Gating**: Lightweight Transformer gate filters 95%+ of normal footage, reserving heavy compute only for confirmed threats.
- ✅ **Spatial-Temporal Vision Transformer (`VideoViT`)**: Leverages pre-trained spatial visual embeddings and 8-Head Temporal Self-Attention across 24-frame clips.
- ✅ **Multi-Stream Feature Blending**: Combines RGB appearance tokens with dynamic frame-to-frame motion difference vectors.
- ✅ **Real-Time Video Clip Analytics**: Complete whole-video evaluation with clip-level confidence scoring and aggregate violence metrics.
- ✅ **Automated Incident Dossier**: Structured breakdown detailing **WHAT** happened (crime type), **HOW** it was done (objects/weapons), and **WHO** was involved (suspect face recognition).

---

## 📊 Model Benchmark & Performance

Tested on the **Real Life Violence Situations Benchmark Dataset** (2,000 CCTV videos, 80/10/10 Stratified Split):

| Model Architecture | Total Params | Model Size | Best Val Acc | **Final Test Acc** | **Violent Recall** | **Violent Precision** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 3D CNN** | ~1.2M | 4.8 MB | 88.00% | `90.00%` | `90.00%` | `90.00%` |
| **3D Res-SE CNN** | ~3.6M | 14.5 MB | 89.00% | `91.50%` | `90.00%` | `92.78%` |
| **VideoViT v1** | ~6.7M | 26.8 MB | 96.00% | `94.50%` | `95.00%` | `94.06%` |
| **Optimized VideoViT v2** 🏆 | **~9.8M** | **38.4 MB** | **`95.50%`** | **`95.50%`** | **`96.00%`** | **`95.05%`** |

### 📈 Phase 1 VideoViT v2 Classification Report (200 Test Videos)

```text
                 Precision    Recall    F1-Score   Support

  Non-Violent     0.9596      0.9500     0.9548      100
      Violent     0.9505      0.9600     0.9552      100

     Accuracy                            0.9550      200
    Macro Avg     0.9550      0.9550     0.9550      200
 Weighted Avg     0.9550      0.9550     0.9550      200
```

---

## 📂 Project Structure

```text
CrimeDetectionCCTV/
├── Architecture.jpeg          # Master Multi-Stage System Architecture
├── pipeline_inference.py      # End-to-End Hierarchical Pipeline (Phase 1 + Phase 2)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excluded datasets & checkpoints
├── README.md                  # Comprehensive Documentation
│
├── Phase 1/                   # 🚨 PHASE 1: BINARY CRIME DETECTION GATE
│   ├── Model.py               # Video Vision Transformer (VideoViT v2, 95.5% Acc)
│   ├── test_Model.py          # Whole-video multi-clip Phase 1 evaluation script
│   ├── datasetLoader.py       # Automated Real Life Violence dataset downloader
│   └── video_check.py         # Video corruption & frame integrity validation
│
├── Phase 2/                   # 🥊 PHASE 2: MULTI-CLASS FINE-GRAINED CLASSIFICATION
│   ├── MultiClassModel.py     # SlowFast Network (14-Class Action Classifier)
│   ├── test_MultiClass.py     # Standalone Stage 2 action evaluation script
│   └── download_ucf.py        # Automated UCF-Crime dataset downloader
│
├── Dataset/                   # Phase 1 Dataset (Violence vs NonViolence)
├── UCF Dataset/               # Phase 2 Dataset (14 UCF-Crime Classes)
└── Optimisedmodel/            # Trained weights & checkpoints
    ├── best_3dcnn_crime_detector.pth        # Phase 1 Best Weights (95.5% Acc)
    └── best_slowfast_crime_classifier.pth   # Phase 2 SlowFast Best Weights
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Visshu78/Crime_Detection.git
cd Crime_Detection
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install torchvision pillow scikit-learn tqdm opencv-python
```

### 4. Download Dataset (Optional for training)
```bash
python Dataset/datasetLoader.py
```

---

## 🏋️ Training

To train the **Spatial-Temporal Vision Transformer (`VideoViT`)** with automatic GPU MIG acceleration and early stopping:

### Foreground Training
```bash
python Model.py
```

### Background Training (Survives SSH Disconnect)
```bash
nohup python Model.py > models1/training.log 2>&1 &
```

To monitor training progress in real-time:
```bash
tail -f models1/training.log
```

---

## 🔍 Inference & Video Analysis

### 1. Analyze an Entire Video File
```bash
python test_Model.py
```
*Prompt:* Enter the path to your video file (e.g. `Dataset/Violence/V_1.mp4`).

**Sample Output:**
```text
Video Information
----------------------------------------
FPS          : 30.0
Total Frames : 120
Duration     : 4.00 seconds
----------------------------------------
Analyzing video...
==================================================
Clip 001/005 | Violence Probability:  96.42% | VIOLENT
Clip 002/005 | Violence Probability:  98.15% | VIOLENT
Clip 003/005 | Violence Probability:  97.80% | VIOLENT
Clip 004/005 | Violence Probability:  94.20% | VIOLENT
Clip 005/005 | Violence Probability:  91.50% | VIOLENT

============================================================
              FINAL VIDEO VERDICT
============================================================
Total Clips              : 5
Violent Clips            : 5/5
Violent Clip Percentage  : 100.00%
Average Violence Score   : 95.61%
------------------------------------------------------------
FINAL PREDICTION         : VIOLENT
============================================================
```

### 2. Single Clip Quick Prediction
```bash
python Model.py --predict path/to/video.mp4
```

---

## 📡 Live CCTV RTSP Integration

To attach to a live IP camera stream via RTSP:

```python
import cv2, torch, numpy as np
from collections import deque
from Model import VideoViT, Config, compute_motion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VideoViT().to(device)
model.load_state_dict(torch.load("models1/best_3dcnn_crime_detector.pth", map_location=device))
model.eval()

# Connect to CCTV RTSP stream or local webcam (0)
CCTV_URL = "rtsp://username:password@192.168.1.100:554/stream1"
cap = cv2.VideoCapture(CCTV_URL)
frame_buffer = deque(maxlen=Config.MAX_FRAMES)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    resized = cv2.resize(frame, Config.TARGET_SIZE).astype("float32") / 255.0
    frame_buffer.append(resized)
    
    if len(frame_buffer) == Config.MAX_FRAMES:
        blended = compute_motion(np.array(frame_buffer))
        tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        
        status = f"🚨 CRIME DETECTED ({prob*100:.1f}%)" if prob > 0.5 else f"✅ NORMAL ({ (1-prob)*100:.1f}%)"
        color = (0, 0, 255) if prob > 0.5 else (0, 255, 0)
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("CCTV Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
```

---

## 🗺️ Roadmap & Future Phases

- [x] **Stage 1 (Complete)**: High-accuracy real-time binary Crime Detector using Spatial-Temporal Vision Transformer (`VideoViT`) (94.5% Test Acc, 96% Val Acc).
- [ ] **Stage 2**: Multi-class Action Classification using SlowFast Network on UCF-Crime / XD-Violence (*Fighting, Burglary, Arson, Robbery, Assault*).
- [ ] **Stage 3A**: Weapon and suspicious object detection integration using YOLOv8 / YOLOv5.
- [ ] **Stage 3B**: Facial feature extraction & suspect matching via RetinaFace and ArcFace.
- [ ] **Stage 4**: Automated PDF/JSON Incident Dossier generation with law enforcement webhook dispatcher.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
