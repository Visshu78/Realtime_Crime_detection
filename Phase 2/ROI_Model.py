# ==============================================================================
# 🚨 PHASE 2: OBJECT & ACTION-CENTRIC 200x200 ROI VIDEO VISION TRANSFORMER
# ==============================================================================

import os
import sys
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO

# Configure PyTorch flags
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# ==============================================================================
# GPU & SEED CONFIGURATION
# ==============================================================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(12)
        print(f"Using CPU for execution ({torch.get_num_threads()} parallel OpenMP CPU threads).")
except Exception:
    device = torch.device("cpu")
    torch.set_num_threads(12)
    print(f"Using CPU for execution ({torch.get_num_threads()} parallel OpenMP CPU threads).")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "Phase" in os.path.dirname(os.path.abspath(__file__)) else os.path.dirname(os.path.abspath(__file__))

class Config:
    DATASET_DIR = os.path.join(PROJECT_ROOT, "XD_Violence_Dataset")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Optimisedmodel")
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_roi_crime_classifier.pth")

    MAX_FRAMES = 24
    TARGET_SIZE = (200, 200)  # 200x200 Object-Centric Region of Interest (RoI)
    BATCH_SIZE = 16
    EPOCHS = 30
    BASE_LR = 3e-4
    BACKBONE_LR = 3e-5
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.35
    LABEL_SMOOTHING = 0.05
    USE_TTA = True
    EARLY_STOP_PATIENCE = 8

XD_CRIME_CLASSES = [
    "Fighting",       # Brawls & physical violence
    "Shooting",       # Firearms & active shooter
    "Explosion",      # Detonations, fireballs, bomb blasts
    "CarAccident",    # Vehicular collisions & crashes
    "Riot",           # Mass crowd unrest & mob violence
    "Abuse"           # One-on-one assault / domestic abuse
]

NUM_CLASSES = len(XD_CRIME_CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(XD_CRIME_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(XD_CRIME_CLASSES)}

# ==============================================================================
# 200x200 OBJECT-CENTRIC ROI EXTRACTOR
# ==============================================================================
class RoIExtractor:
    """
    Extracts a focused 200x200 bounding box centered on detected weapons,
    interacting humans, or active motion regions.
    """
    def __init__(self):
        self.yolo = YOLO("yolov8n.pt")

    def crop_roi(self, frame, target_size=(200, 200)):
        H, W, _ = frame.shape
        crop_w, crop_h = target_size

        # Run YOLO to find weapons, persons, or cars
        results = self.yolo.predict(source=frame, conf=0.25, verbose=False)
        center_x, center_y = W // 2, H // 2

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            threat_box = None
            person_box = None

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.yolo.names[cls_id].lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if any(w in cls_name for w in ["knife", "gun", "pistol", "bat", "scissors"]):
                    threat_box = ((x1 + x2) // 2, (y1 + y2) // 2)
                    break
                elif "person" in cls_name and person_box is None:
                    # Focus on upper torso/arms
                    person_box = ((x1 + x2) // 2, int(y1 + 0.35 * (y2 - y1)))

            if threat_box:
                center_x, center_y = threat_box
            elif person_box:
                center_x, center_y = person_box

        # Calculate 200x200 crop boundaries clamped within frame
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(W, x1 + crop_w)
        y2 = min(H, y1 + crop_h)

        # Adjust for edge borders
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, target_size)
        return crop

# ==============================================================================
# DATASET LOADER WITH 200x200 ROI CROPPING
# ==============================================================================
def compute_motion(frames):
    motion = np.abs(np.diff(frames, axis=0))
    first_diff = motion[0:1]
    motion = np.concatenate([first_diff, motion], axis=0)
    blended = 0.70 * frames + 0.30 * motion
    return blended

class ROIXDDataset(Dataset):
    def __init__(self, video_items, max_frames=Config.MAX_FRAMES, target_size=Config.TARGET_SIZE, augment=False):
        self.video_items = video_items
        self.max_frames = max_frames
        self.target_size = target_size
        self.augment = augment
        self.roi_extractor = RoIExtractor()

    def __len__(self):
        return len(self.video_items)

    def __getitem__(self, idx):
        video_path, class_idx = self.video_items[idx]
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        raw_frames = []
        if total_frames <= self.max_frames:
            while True:
                ret, frame = cap.read()
                if not ret: break
                raw_frames.append(frame)
            cap.release()
            while len(raw_frames) < self.max_frames:
                raw_frames.append(raw_frames[-1] if len(raw_frames) > 0 else np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8))
        else:
            if self.augment:
                start_frame = random.randint(0, total_frames - self.max_frames)
            else:
                start_frame = (total_frames - self.max_frames) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(self.max_frames):
                ret, frame = cap.read()
                if not ret: break
                raw_frames.append(frame)
            cap.release()
            while len(raw_frames) < self.max_frames:
                raw_frames.append(raw_frames[-1] if len(raw_frames) > 0 else np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8))

        # Extract 200x200 RoI Crop on each frame
        roi_frames = []
        for frame in raw_frames[:self.max_frames]:
            crop = self.roi_extractor.crop_roi(frame, target_size=self.target_size)
            crop_norm = crop.astype("float32") / 255.0
            roi_frames.append(crop_norm)

        frames = np.array(roi_frames)

        if self.augment:
            if random.random() > 0.5:
                frames = np.flip(frames, axis=2).copy()
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-0.10, 0.10)
                frames = np.clip(frames * alpha + beta, 0.0, 1.0)

        blended = compute_motion(frames)
        tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2)
        label = torch.tensor(class_idx, dtype=torch.long)
        return tensor, label

# ==============================================================================
# MODEL ARCHITECTURE (200x200 VideoViT)
# ==============================================================================
class ROICrimeClassifierViT(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_frames=Config.MAX_FRAMES, d_model=512, num_layers=3, num_heads=8):
        super(ROICrimeClassifierViT, self).__init__()
        
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.spatial_backbone = backbone.features
        
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(960, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(p=Config.DROPOUT)
        )
        
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, d_model))
        self.pos_drop = nn.Dropout(p=Config.DROPOUT)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=Config.DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, num_classes)
        )
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_frames = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        
        feat = self.spatial_backbone(x_frames)
        tokens = self.proj(feat).view(B, T, -1)
        tokens = tokens + self.temporal_conv(tokens.permute(0, 2, 1)).permute(0, 2, 1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tokens, tokens), dim=1)
        x_tokens = self.pos_drop(x_tokens + self.pos_embed)
        
        trans_out = self.norm(self.temporal_transformer(x_tokens))
        cls_rep = trans_out[:, 0]
        logits = self.head(cls_rep)
        return logits

# ==============================================================================
# DATASET DISCOVERY
# ==============================================================================
def discover_xd_dataset(dataset_dir=Config.DATASET_DIR):
    video_items = []
    if not os.path.exists(dataset_dir):
        return video_items

    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                full_path = os.path.join(root, f)
                lower_path = full_path.lower()
                if "normal" in lower_path or "nonviolence" in lower_path:
                    continue
                
                matched_class = None
                for cls_name in XD_CRIME_CLASSES:
                    if cls_name.lower() in lower_path:
                        matched_class = cls_name
                        break
                    elif cls_name == "CarAccident" and ("accident" in lower_path or "crash" in lower_path):
                        matched_class = "CarAccident"
                        break
                    elif cls_name == "Fighting" and ("fight" in lower_path or "brawl" in lower_path):
                        matched_class = "Fighting"
                        break
                    elif cls_name == "Shooting" and ("shoot" in lower_path or "gun" in lower_path):
                        matched_class = "Shooting"
                        break

                if matched_class:
                    video_items.append((full_path, CLASS_TO_IDX[matched_class]))

    return video_items

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    if "--dry-run" in sys.argv:
        print("\n--- DRY RUN: 200x200 ROI VIDEO VISION TRANSFORMER ---")
        model = ROICrimeClassifierViT(num_classes=NUM_CLASSES).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Initialized: {total_params:,} parameters across {NUM_CLASSES} classes.")
        dummy = torch.randn(2, 3, Config.MAX_FRAMES, 200, 200).to(device)
        out = model(dummy)
        print(f"Forward Pass Shape: {out.shape} (Expected: [2, {NUM_CLASSES}])")
        print("Dry run completed successfully.")
        return

    video_items = discover_xd_dataset()
    print(f"\nDiscovered {len(video_items)} total Crime Videos for 200x200 RoI Training.")

    labels = [vi[1] for vi in video_items]
    train_items, temp_items = train_test_split(video_items, test_size=0.20, stratify=labels, random_state=SEED)
    temp_labels = [ti[1] for ti in temp_items]
    val_items, test_items = train_test_split(temp_items, test_size=0.50, stratify=temp_labels, random_state=SEED)

    train_dataset = ROIXDDataset(train_items, augment=True)
    val_dataset   = ROIXDDataset(val_items, augment=False)
    test_dataset  = ROIXDDataset(test_items, augment=False)

    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)

    model = ROICrimeClassifierViT(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    optimizer = optim.AdamW([
        {'params': model.spatial_backbone.parameters(), 'lr': Config.BACKBONE_LR},
        {'params': model.proj.parameters(), 'lr': Config.BASE_LR},
        {'params': model.temporal_conv.parameters(), 'lr': Config.BASE_LR},
        {'params': model.temporal_transformer.parameters(), 'lr': Config.BASE_LR},
        {'params': model.head.parameters(), 'lr': Config.BASE_LR},
        {'params': [model.cls_token, model.pos_embed], 'lr': Config.BASE_LR}
    ], weight_decay=Config.WEIGHT_DECAY)

    print("\nStarting 200x200 Object-Centric RoI VideoViT Training...")
    print("=" * 65)

    best_val_acc = 0.0
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{Config.EPOCHS:02d}] RoI", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            pbar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})

        train_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch [{epoch+1:02d}/{Config.EPOCHS:02d}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  --> Saved Best RoI Model Checkpoint! ({best_val_acc:.2f}%)")

if __name__ == "__main__":
    main()
