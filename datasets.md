# 📦 Dataset Repositories & Download Sources
## Real-Time CCTV Crime Detection & Evidence Extraction System

This document catalogs all datasets used, tested, and referenced across each phase of the project, including Kaggle URLs, `kagglehub` identifiers, and descriptions.

---

## 🚨 Phase 1: Binary Crime Detection Gate

### 1. Real Life Violence Situations Dataset
* **Purpose**: Primary training dataset for Phase 1 Binary Crime Gating (*Violence* vs *Non-Violence*).
* **Kaggle URL**: [kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
* **Kagglehub Identifier**: `mohamedmustafa/real-life-violence-situations-dataset`
* **Size**: ~3.6 GB (2,000 CCTV and real-life surveillance video clips: 1,000 Violent / 1,000 Non-Violent)
* **Status**: ✅ **Fully trained & verified** — Achieved **95.50% Test Accuracy** and **96.00% Crime Recall** with `VideoViT v2`.
* **Download Code**:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset")
  ```

---

## 🥊 Phase 2: Multi-Class Crime Action Recognition

### 2. UCF-Crime Dataset (Pre-Extracted Frames Edition)
* **Purpose**: Initial dataset evaluated for 13-Class / 4-Cluster Fine-Grained Crime Classification.
* **Kaggle URL**: [kaggle.com/datasets/odins0n/ucf-crime-dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset)
* **Kagglehub Identifier**: `odins0n/ucf-crime-dataset`
* **Official Academic URL**: [crcv.ucf.edu/projects/real-world/](https://www.crcv.ucf.edu/projects/real-world/)
* **Size**: ~11.0 GB (~480,000 PNG image frames across 13 Crime categories + NormalVideos)
* **Classes**: *Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism, NormalVideos*
* **Benchmark Result**: Achieved **52.37% Test Accuracy** using the 4-Cluster Hierarchical Action Model.
* **Download Code**:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("odins0n/ucf-crime-dataset")
  ```

---

### 3. XD-Violence Dataset (Large-Scale Multi-Class CCTV Benchmark)
* **Purpose**: Large-scale dataset for 6-Class High-Accuracy Crime & Action Recognition.
* **Kaggle URL**: [kaggle.com/datasets/bypktt/xd-violence](https://www.kaggle.com/datasets/bypktt/xd-violence)
* **Kagglehub Identifier**: `bypktt/xd-violence`
* **Official Project Page**: [roc-ng.github.io/XD-Violence/](https://roc-ng.github.io/XD-Violence/)
* **Size**: ~75.9 GB (4,754 surveillance videos across 217 hours of multi-camera CCTV footage)
* **6 Clean Action Classes**:
  1. 🥊 **`Fighting`** *(Physical brawls & assaults)*
  2. 🔫 **`Shooting`** *(Active gunfire & firearms)*
  3. 💥 **`Explosion`** *(Detonations & fire hazards)*
  4. 🚗 **`Car Accident`** *(High-speed vehicular collisions)*
  5. 👥 **`Riot`** *(Mass mob violence & street unrest)*
  6. 🛑 **`Abuse`** *(One-on-one domestic abuse)*
* **Download Code**:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("bypktt/xd-violence")
  ```

---

## 🎯 Phase 3: Weapon Detection & Face Recognition (Future Stages)

### 4. Guns & Knives Threat Object Detection Dataset (YOLO Format)
* **Purpose**: Training and testing YOLOv8 for lethal weapon detection (*Guns, Knives, Bats, Crowbars*).
* **Kaggle URL (Weapons)**: [kaggle.com/datasets/issaisas/guns-knives-weapons-detection](https://www.kaggle.com/datasets/issaisas/guns-knives-weapons-detection)
* **Kaggle URL (Knife Detection)**: [kaggle.com/datasets/vbookshelf/knife-detection-dataset](https://www.kaggle.com/datasets/vbookshelf/knife-detection-dataset)
* **Kagglehub Identifier**: `issaisas/guns-knives-weapons-detection`

### 5. InsightFace Pretrained Models (RetinaFace + ArcFace)
* **Purpose**: Suspect facial detection, alignment, and 512-D identity embedding extraction for police watchlist matching.
* **GitHub Repository**: [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
* **Pretrained Models**: `buffalo_l` (RetinaFace ResNet50 + ArcFace ResNet100)
