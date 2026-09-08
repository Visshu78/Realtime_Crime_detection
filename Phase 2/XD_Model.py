# ==============================================================================
# 🚨 REFINED PHASE 2: 6-CLASS VIDEO VISION TRANSFORMER (XD-Violence)
#    - Preprocessing: Adaptive CLAHE Contrast + Unsharp Masking (USM) Sharpness
#    - Temporal Sampling: TSN (Temporal Segment Network) 24-Segment Sampling
#    - Optimization: Class-Balanced Focal Loss + Video MixUp + Cosine LR Warmup
# ==============================================================================

import os
import sys
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import defaultdict
from tqdm import tqdm

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
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_xd_crime_classifier.pth")

    NUM_SEGMENTS = 24         # 24 temporal segments spanning the full video (TSN)
    TARGET_SIZE = (160, 160)
    CLIPS_PER_VIDEO = 3       # 3 dynamic jittered TSN passes per training video (3x data volume)
    BATCH_SIZE = 16
    EPOCHS = 35
    BASE_LR = 3e-4
    BACKBONE_LR = 3e-5
    WEIGHT_DECAY = 1e-3
    DROPOUT = 0.40
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.05
    MIXUP_PROB = 0.35
    MIXUP_ALPHA = 0.20
    USE_TTA = True
    EARLY_STOP_PATIENCE = 12

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
# IMAGE PREPROCESSING: SHARPNESS & CLAHE ENHANCEMENT
# ==============================================================================
def enhance_cctv_frame(frame_rgb, sharpness_strength=0.4):
    """
    Applies fast Unsharp Masking (USM) edge sharpening to make weapons and motion edges crisp.
    """
    if sharpness_strength > 0:
        blurred = cv2.GaussianBlur(frame_rgb, (3, 3), sigmaX=1.0)
        sharpened = cv2.addWeighted(frame_rgb, 1.0 + sharpness_strength, blurred, -sharpness_strength, 0)
        return np.clip(sharpened, 0.0, 1.0)
    return frame_rgb

def compute_motion_residual(frames):
    """
    Computes kinetic velocity difference between frames and blends 70% RGB + 30% Motion.
    """
    motion = np.abs(np.diff(frames, axis=0))
    first_diff = motion[0:1]
    motion = np.concatenate([first_diff, motion], axis=0)
    blended = 0.70 * frames + 0.30 * motion
    return np.clip(blended, 0.0, 1.0)

# ==============================================================================
# DATASET: TSN (TEMPORAL SEGMENT NETWORK) VIDEO DATASET
# ==============================================================================
class XDViolenceTSNDataset(Dataset):
    def __init__(self, video_samples, num_segments=Config.NUM_SEGMENTS, target_size=Config.TARGET_SIZE, is_train=False):
        self.video_samples = video_samples  # List of (video_path, class_idx)
        self.num_segments = num_segments
        self.target_size = target_size
        self.is_train = is_train

    def __len__(self):
        return len(self.video_samples)

    def _sample_indices(self, total_frames):
        if total_frames <= self.num_segments:
            indices = np.linspace(0, max(0, total_frames - 1), self.num_segments, dtype=int)
            return indices

        segment_len = total_frames / float(self.num_segments)
        indices = []
        for i in range(self.num_segments):
            start = int(i * segment_len)
            end = int((i + 1) * segment_len)
            if self.is_train:
                idx = random.randint(start, max(start, end - 1))
            else:
                idx = (start + end) // 2
            indices.append(min(idx, total_frames - 1))
        return indices

    def __getitem__(self, idx):
        video_path, class_idx = self.video_samples[idx]
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.num_segments

        ordered_indices = self._sample_indices(total_frames)
        frames = []
        for i in ordered_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i))
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            frames.append(frame)
        cap.release()

        while len(frames) < self.num_segments:
            frames.append(frames[-1].copy() if len(frames) > 0 else np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32))
        frames = np.array(frames[:self.num_segments])

        # Apply Fast Sharpness enhancement
        sharp_strength = random.uniform(0.2, 0.6) if self.is_train else 0.4
        frames = np.array([enhance_cctv_frame(f, sharpness_strength=sharp_strength) for f in frames])

        # Dynamic Data Augmentations (Train Mode)
        if self.is_train:
            if random.random() > 0.5:
                frames = np.flip(frames, axis=2).copy()
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-0.08, 0.08)
                frames = np.clip(frames * alpha + beta, 0.0, 1.0)
            if random.random() < 0.25:
                frames = np.array([cv2.GaussianBlur(f, (3, 3), sigmaX=0.8) for f in frames])

        blended = compute_motion_residual(frames)
        tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2)
        label = torch.tensor(class_idx, dtype=torch.long)
        return tensor, label

# ==============================================================================
# DATASET DISCOVERY & CLASS WEIGHT CALCULATION
# ==============================================================================
def discover_xd_dataset(dataset_dir=Config.DATASET_DIR):
    video_items = []
    if not os.path.exists(dataset_dir):
        print(f"[Notice] Dataset directory '{dataset_dir}' not found.")
        return [], [], [], None

    class_counts = defaultdict(int)
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
                    c_idx = CLASS_TO_IDX[matched_class]
                    video_items.append((full_path, c_idx))
                    class_counts[c_idx] += 1

    print(f"\nDiscovered {len(video_items)} total Crime Videos across {len(XD_CRIME_CLASSES)} XD-Violence Categories:")
    for cls_name in XD_CRIME_CLASSES:
        idx = CLASS_TO_IDX[cls_name]
        print(f"  - {cls_name:15s}: {class_counts[idx]:4d} videos")

    # Stratified Train/Val/Test Split (75% Train / 10% Val / 15% Test)
    labels = [vi[1] for vi in video_items]
    train_vids, temp_vids = train_test_split(video_items, test_size=0.25, stratify=labels, random_state=SEED)
    temp_labels = [ti[1] for ti in temp_vids]
    val_vids, test_vids = train_test_split(temp_vids, test_size=0.60, stratify=temp_labels, random_state=SEED)

    # Replicate training items with multiple jittered passes to increase volume
    train_samples = []
    for _ in range(Config.CLIPS_PER_VIDEO):
        train_samples.extend(train_vids)
    random.shuffle(train_samples)

    print(f"\nDataset Splits: {len(train_samples)} Train Samples ({len(train_vids)} unique) | {len(val_vids)} Val Videos | {len(test_vids)} Test Videos")

    # Compute Class-Balanced Focal Weights (Inverse Effective Number)
    total_train = len(train_vids)
    beta = 0.999
    effective_num = [1.0 - np.power(beta, max(1, class_counts[i])) for i in range(NUM_CLASSES)]
    weights = [(1.0 - beta) / np.array(effective_num[i]) for i in range(NUM_CLASSES)]
    weights = np.array(weights) / np.sum(weights) * NUM_CLASSES
    class_weights_tensor = torch.FloatTensor(weights).to(device)
    print("Computed Class-Balanced Weights:", {IDX_TO_CLASS[i]: round(float(w), 3) for i, w in enumerate(weights)})

    return train_samples, val_vids, test_vids, class_weights_tensor

# ==============================================================================
# LOSS FUNCTION: CLASS-BALANCED FOCAL LOSS WITH LABEL SMOOTHING
# ==============================================================================
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=Config.FOCAL_GAMMA, smoothing=Config.LABEL_SMOOTHING):
        super(ClassBalancedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        focal_weight = torch.pow(1.0 - probs, self.gamma)
        loss = -focal_weight * smooth_targets * log_probs
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
            
        return loss.sum(dim=-1).mean()

# ==============================================================================
# MODEL ARCHITECTURE: REFINED VIDEO VISION TRANSFORMER
# ==============================================================================
class XDCrimeClassifierViT(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_frames=Config.NUM_SEGMENTS, d_model=512, num_layers=3, num_heads=8):
        super(XDCrimeClassifierViT, self).__init__()
        
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
# VIDEO MIXUP AUGMENTATION
# ==============================================================================
def video_mixup(inputs, targets, alpha=Config.MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    targets_a, targets_b = targets, targets[index]
    return mixed_inputs, targets_a, targets_b, lam

# ==============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(train_loader, desc="Training XD VideoViT", unit="batch", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        use_mixup = random.random() < Config.MIXUP_PROB
        if use_mixup:
            inputs, labels_a, labels_b, lam = video_mixup(inputs, labels)
        
        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                if use_mixup:
                    loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if use_mixup:
                loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        if not use_mixup:
            correct += (predicted == labels).sum().item()
        else:
            correct += (lam * (predicted == labels_a).float() + (1.0 - lam) * (predicted == labels_b).float()).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

    return running_loss / total, 100.0 * correct / total

def eval_epoch(model, dataloader, criterion, device, use_tta=Config.USE_TTA):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if use_tta:
                outputs1 = model(inputs)
                flipped = torch.flip(inputs, dims=[4])  # Flip width axis
                outputs2 = model(flipped)
                probs = (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1)) / 2.0
            else:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

            loss = criterion(torch.log(probs.clamp(1e-7, 1.0)), labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())

    acc = 100.0 * correct / total
    avg_loss = running_loss / total
    return avg_loss, acc, np.array(y_true), np.array(y_pred)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("🚨 PHASE 2: REFINED XD-VIOLENCE ACTION RECOGNITION PIPELINE")
    print("=" * 65)

    # 1. Discover & Split Dataset
    train_samples, val_vids, test_vids, class_weights = discover_xd_dataset()
    if not train_samples:
        print("[Error] No XD-Violence videos found to train.")
        return

    # 2. Instantiate Datasets & DataLoaders
    train_dataset = XDViolenceTSNDataset(train_samples, num_segments=Config.NUM_SEGMENTS, is_train=True)
    val_dataset   = XDViolenceTSNDataset(val_vids, num_segments=Config.NUM_SEGMENTS, is_train=False)
    test_dataset  = XDViolenceTSNDataset(test_vids, num_segments=Config.NUM_SEGMENTS, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Initialize Model
    model = XDCrimeClassifierViT(
        num_classes=NUM_CLASSES,
        num_frames=Config.NUM_SEGMENTS,
        d_model=512,
        num_layers=3,
        num_heads=8
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nRefined Phase 2 Model Initialized: {total_params:,} parameters across {NUM_CLASSES} Action Classes.")

    # 4. Optimizer, Scheduler, Loss & Scaler
    backbone_params = list(model.spatial_backbone.parameters())
    transformer_params = [p for n, p in model.named_parameters() if not n.startswith("spatial_backbone")]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': Config.BACKBONE_LR},
        {'params': transformer_params, 'lr': Config.BASE_LR}
    ], weight_decay=Config.WEIGHT_DECAY)

    # Cosine Annealing with Warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    criterion = ClassBalancedFocalLoss(class_weights=class_weights, gamma=Config.FOCAL_GAMMA, smoothing=Config.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # 5. Training Loop
    best_val_acc = 0.0
    patience_counter = 0

    print("\nStarting Refined XD-Violence TSN Training...")
    print("=" * 65)

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device, use_tta=False)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{Config.EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  --> Best Phase 2 Checkpoint Saved! (Val Accuracy: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"\n[Early Stop] Val accuracy did not improve for {Config.EARLY_STOP_PATIENCE} epochs. Stopping.")
                break

    # 6. Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 65)
    print("EVALUATING BEST REFINED XD-VIOLENCE MODEL ON TEST SET (WITH TTA)")
    print("=" * 65)

    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    test_loss, test_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device, use_tta=Config.USE_TTA)
    print(f"\nFinal Phase 2 XD-Violence Test Accuracy: {test_acc:.2f}%\n")

    report = classification_report(y_true, y_pred, target_names=XD_CRIME_CLASSES, digits=4, zero_division=0)
    print("CLASSIFICATION REPORT:\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("CONFUSION MATRIX:\n", cm)


if __name__ == "__main__":
    main()
