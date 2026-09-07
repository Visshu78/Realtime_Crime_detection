# 🏛️ Architecture & Technical Decision Record (ADR)
## Real-Time CCTV Crime Detection & Evidence Extraction System

**Target Repository**: [github.com/Visshu78/Realtime_Crime_detection](https://github.com/Visshu78/Realtime_Crime_detection)  
**Author**: Amith K. (`Visshu78`)  
**Last Updated**: August 2026  

---

## 📑 Table of Contents
1. [Master System Architecture & Vision](#1-master-system-architecture--vision)
2. [Phase 1: Binary Crime Detection Gate](#2-phase-1-binary-crime-detection-gate)
3. [Phase 2: Multi-Class Action Recognition Decisions](#3-phase-2-multi-class-action-recognition-decisions)
4. [Dataset & Data Engineering Decisions](#4-dataset--data-engineering-decisions)
5. [Codebase & Modularization Decisions](#5-codebase--modularization-decisions)
6. [Hardware & Infrastructure Decisions](#6-hardware--infrastructure-decisions)
7. [Next Phase Roadmap (Stage 3 & Stage 4)](#7-next-phase-roadmap-stage-3--stage-4)

---

## 1. Master System Architecture & Vision

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CCTV Live Stream / Video Feed                     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ (15–30 FPS)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Fast Binary Crime Gate (VideoViT v2 — 95.5% Test Accuracy)   │
│  "Is there a violent disruption occurring right now? YES / NO"         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ (If Threat P > 0.50)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Fine-Grained Crime Action Classifier                         │
│  "WHAT physical crime is occurring?" (Violence, Theft, Hazard, Vandal)  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│  STAGE 3A: Weapon & Object      │   │  STAGE 3B: Suspect Facial ID    │
│  Detection (YOLOv8 / YOLOv5)    │   │  (RetinaFace + ArcFace)         │
│  "HOW? Gun, Knife, Crowbar"     │   │  "WHO? Facial Recognition"      │
└────────────────┬────────────────┘   └────────────────┬────────────────┘
                 │                                     │
                 └───────────────────┬─────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Automated Law Enforcement Incident Dossier & Alerts          │
│  Instant PDF/JSON Police Report with Timestamps, Threat, & Face Crops  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: Binary Crime Detection Gate

### 📌 Decision 1.1: Architecture Progression from 3D CNN to Vision Transformers
* **Context**: The baseline 3D CNN plateaued at 90.0% accuracy with rigid spatio-temporal convolutions that struggled on complex multi-person movements.
* **Decision**: Evolved the architecture through four iterative generations:
  1. `Baseline 3D CNN`: 90.00% Test Acc | 90.00% Violent Recall (4.8 MB)
  2. `3D Res-SE CNN`: 91.50% Test Acc | 90.00% Violent Recall (14.5 MB)
  3. `VideoViT v1`: 94.50% Test Acc | 95.00% Violent Recall (26.8 MB)
  4. `Optimized VideoViT v2`: **95.50% Test Acc | 96.00% Violent Recall | 95.05% Precision** (38.4 MB)
* **Outcome**: **VideoViT v2** was selected as the final production champion for Phase 1.

### 📌 Decision 1.2: Hybrid Spatio-Temporal Modeling
* **Decision**: Adopted a hybrid architecture:
  * **Spatial Feature Tokenizer**: Pretrained `MobileNetV3-Large` (960D projected to 512D) for lightweight, high-speed spatial representation.
  * **Local Kinematic Conv**: 1D Temporal Depthwise Convolution ($k=3$) to preserve continuous limb motion across consecutive frames.
  * **Global Action Attention**: 3-layer, 8-head Multi-Head Self-Attention Transformer with Pre-LN normalization to capture long-range contextual interactions.

### 📌 Decision 1.3: Spatio-Temporal Motion Blending & TTA
* **Decision**:
  * Input tensor is preprocessed with **Dense Motion Blending**: `0.7 * RGB + 0.3 * FrameDifference`.
  * **Test-Time Augmentation (TTA)**: Evaluates both normal and horizontally mirrored video clips during inference, boosting accuracy by **+1.0%** and stabilizing prediction jitter.

---

## 3. Phase 2: Multi-Class Action Recognition Decisions

### 📌 Decision 2.1: Dataset Selection (UCF-Crime)
* **Decision**: Downloaded and extracted `odins0n/ucf-crime-dataset` (11.0 GB, ~480,000 `.png` frames) into `UCF Dataset/` covering 14 classes (13 Crime categories + NormalVideos).

### 📌 Decision 2.2: Decoupling Normal Videos from Phase 2
* **Observation**: In the initial 14-class SlowFast experiment, `NormalVideos` comprised 950 out of 1,900 videos (50% of the dataset), creating a severe 19:1 class imbalance that caused the model to plateau at 49.5% by defaulting to "Normal".
* **Decision**: Decoupled `NormalVideos` entirely from Phase 2. Since **Phase 1 already filters out Normal footage with 95.5% accuracy**, Phase 2 focuses 100% of its capacity on distinguishing the 13 actual Crime actions.

### 📌 Decision 2.3: Tackling 13-Class Visual Overlap
* **Observation**: In the 13-class experiment, training accuracy reached 97.75% while test accuracy fell to 22.61%.
* **Root Cause**: The 13 UCF-Crime classes are *legal penal code definitions*, not visually distinct physical actions:
  * `Fighting`, `Assault`, and `Abuse` share identical wrestling and punching kinematics.
  * `Burglary`, `Stealing`, `Robbery`, and `Shoplifting` share identical reaching and fleeing motions.
* **Decision (Hierarchical Meta-Clustering)**: Grouped the 13 classes into **4 Physically Distinct Action Clusters**:
  1. 🥊 **`Physical_Violence`** (*Fighting, Assault, Abuse, Shooting*) — 200 videos
  2. 💰 **`Theft_and_Robbery`** (*Robbery, Burglary, Stealing, Shoplifting*) — 400 videos
  3. 🔥 **`Hazard_and_Disaster`** (*Arson, Explosion, Road Accidents*) — 250 videos
  4. 🚨 **`Vandalism_Disturbance`** (*Vandalism, Arrest*) — 100 videos
* **Outcome**: Test accuracy immediately jumped from **22.61% $\to$ 52.37%** (Validation: 56.05%).

### 📌 Decision 2.5: Transition to XD-Violence Dataset Benchmark (71.23% Test Accuracy)
* **Context**: While UCF-Crime 4-clustering improved accuracy from 22.6% to 52.4%, the noisy 240p untrimmed clips limited overall precision.
* **Decision**: Transitioned Phase 2 to the large-scale **`XD-Violence` Dataset** (2,115 videos across 6 distinct categories: *Fighting, Shooting, Explosion, CarAccident, Riot, Abuse*).
* **Outcome**:
  * **Test Accuracy**: **`71.23%`** (Macro: 65.21%, Weighted F1: **`71.41%`**)
  * **Car Accident Precision**: **`92.86%`**
  * **Riot / Mob Violence Precision**: **`75.44%`**
  * **Explosion Precision**: **`71.43%`**
  * **Fighting Precision**: **`76.67%`**
  * Saved Checkpoint: `Optimisedmodel/best_xd_crime_classifier.pth` (38.4 MB)

---

## 4. Dataset & Data Engineering Decisions

| Dataset | Split / Purpose | Size | Classes | Key Preprocessing |
| :--- | :--- | :---: | :---: | :--- |
| **Real Life Violence Situations** | Phase 1 (Binary Gate) | 2,000 Videos | 2 (`Violence`, `NonViolence`) | 24-frame uniform sampling, 128x128 resize, Motion blend |
| **UCF-Crime Dataset** | Phase 2 (Action Classifier) | 950 Videos / 11 GB | 13 Crime Actions $\to$ 4 Clusters | Multi-clip sliding window slicing (8 clips/vid $\approx$ 6,000 clips) |

* **Integrity Validation**: Created automated scripts (`video_check.py`) to detect and remove corrupted video headers, missing frames, and truncated bitstreams before training.

---

## 5. Codebase & Modularization Decisions

### 📌 Decision 5.1: Directory Modularization
* **Decision**: Completely modularized the codebase into distinct lifecycle folders:
  * **`Phase 1/`**: Binary Crime Detection Gate (`Model.py`, `test_Model.py`, `datasetLoader.py`, `video_check.py`).
  * **`Phase 2/`**: Multi-Class Crime Classification (`MultiClassModel.py`, `test_MultiClass.py`, `download_ucf.py`).
  * **Root Level**: Master pipeline orchestration (`pipeline_inference.py`, `Architecture.jpeg`, `result.txt`, `README.md`, `decision.md`).
  * **`Optimisedmodel/`**: Checkpoints directory (`best_3dcnn_crime_detector.pth`, `best_multiclass_crime_classifier.pth`).

### 📌 Decision 5.2: GitHub Version Control
* **Target Repository**: `https://github.com/Visshu78/Realtime_Crime_detection.git`
* **Clean Commits**: All commit records and author metadata configured under `Visshu78`.
* **Git Hygiene**: Datasets (`Dataset/`, `UCF Dataset/`) and heavy model weights (`.pth`) excluded via `.gitignore` to maintain a lightweight, fast repository.

---

## 6. Hardware & Infrastructure Decisions

* **Compute Platform**: University SSH Server with **NVIDIA H100 PCIe MIG 2g.20gb** GPU instance.
* **Container SHM Handling**: Set `num_workers = 0` in PyTorch DataLoaders to prevent `/dev/shm` bus errors common in shared container environments, utilizing the server's 251 GB host RAM.
* **Automatic Fallback**: Code configured to automatically fallback to 12 parallel OpenMP CPU threads if GPU pass-through is temporarily unmounted by host administrators.

---

## 7. Next Phase Roadmap (Stage 3 & Stage 4)

1. **Stage 3A: Weapon & Dangerous Object Detection**
   * Integrate **YOLOv8** to detect visible weapons (*Guns, Knives, Bats, Crowbars*) in high threat frames.
2. **Stage 3B: Suspect Facial Recognition & Evidence Cropping**
   * Integrate **RetinaFace (Detector)** + **ArcFace (Feature Embedding)** to extract clear suspect face crops and match against police watchlists.
3. **Stage 4: Automated Incident Evidence Dossier**
   * Build an automated alert generator producing timestamped **PDF & JSON Incident Dossiers** with crime category, confidence level, weapon type, and suspect face crops for law enforcement dispatch.
