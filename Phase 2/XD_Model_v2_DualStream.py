# ==============================================================================
# 🚨 PHASE 2 (v2): DUAL-STREAM MOTION-HOTSPOT VIDEO VISION TRANSFORMER
#    - Stream 1: Global Wide-Angle Context (Full Scene: Crowds, Traffic, Blasts)
#    - Stream 2: Dynamic Motion-Centric Hotspot Zoom (3x Resolution on Action/Weapons)
#    - Fusion: Cross-Attention Multi-Head Temporal Transformer
#    - Preprocessing: USM Edge Sharpness + Fast TSN 24-Segment Sampling
#    - Loss: Class-Balanced Focal Loss (gamma=2.0) + Label Smoothing
# ==============================================================================

import os
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

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
        device = torch.device("cuda:0")
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        num_cores = os.cpu_count() or 12
        train_threads = min(32, num_cores)
        torch.set_num_threads(train_threads)
        try:
            torch.set_num_interop_threads(min(16, num_cores // 2))
        except Exception:
            pass
        print(f"🚀 Maximizing CPU Engine: Allocated {train_threads} PyTorch matrix computation threads across {num_cores} cores.")
except Exception:
    device = torch.device("cpu")
    torch.set_num_threads(24)
    print("Using CPU engine with 24 threads.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "Phase" in os.path.dirname(os.path.abspath(__file__)) else os.path.dirname(os.path.abspath(__file__))

class Config:
    DATASET_DIR = os.path.join(PROJECT_ROOT, "XD_Violence_Dataset")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Optimisedmodel")
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_xd_crime_classifier_v2_dualstream.pth")

    NUM_SEGMENTS = 24         # 24 temporal segments (TSN)
    TARGET_SIZE = (160, 160)
    CLIPS_PER_VIDEO = 3       # 3 dynamic jittered TSN passes per training video
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
# PREPROCESSING: USM SHARPNESS & MOTION-CENTRIC DYNAMIC HOTSPOT CROPPING
# ==============================================================================
def enhance_cctv_frame(frame_rgb, sharpness_strength=0.4):
    """Applies fast Unsharp Masking (USM) edge sharpening."""
    if sharpness_strength > 0:
        blurred = cv2.GaussianBlur(frame_rgb, (3, 3), sigmaX=1.0)
        sharpened = cv2.addWeighted(frame_rgb, 1.0 + sharpness_strength, blurred, -sharpness_strength, 0)
        return np.clip(sharpened, 0.0, 1.0)
    return frame_rgb

def extract_motion_hotspot_box(raw_frames_rgb, target_size=Config.TARGET_SIZE):
    """
    Ultra-fast dynamic motion hotspot crop (<0.5ms).
    Locates peak kinetic movement on a downsampled grid and extracts high-res 60% center crop.
    """
    if len(raw_frames_rgb) < 2:
        return [cv2.resize(f, target_size) for f in raw_frames_rgb]

    mid_idx = len(raw_frames_rgb) // 2
    f_start = cv2.resize(raw_frames_rgb[0], (40, 40))
    f_mid   = cv2.resize(raw_frames_rgb[mid_idx], (40, 40))
    f_end   = cv2.resize(raw_frames_rgb[-1], (40, 40))
    
    diff = np.abs(f_mid - f_start) + np.abs(f_end - f_mid)
    diff_gray = np.mean(diff, axis=2)
    
    my, mx = np.unravel_index(np.argmax(diff_gray), (40, 40))
    cx, cy = mx / 40.0, my / 40.0

    h, w = raw_frames_rgb[0].shape[:2]
    box_w, box_h = int(0.60 * w), int(0.60 * h)
    center_x, center_y = int(cx * w), int(cy * h)

    x1 = max(0, min(w - box_w, center_x - box_w // 2))
    y1 = max(0, min(h - box_h, center_y - box_h // 2))
    x2 = min(w, x1 + box_w)
    y2 = min(h, y1 + box_h)

    hotspot_frames = [cv2.resize(f[y1:y2, x1:x2], target_size) for f in raw_frames_rgb]
    return hotspot_frames

def compute_motion_residual(frames):
    """Computes kinetic velocity difference between frames and blends 70% RGB + 30% Motion."""
    motion = np.abs(np.diff(frames, axis=0))
    first_diff = motion[0:1]
    motion = np.concatenate([first_diff, motion], axis=0)
    blended = 0.70 * frames + 0.30 * motion
    return np.clip(blended, 0.0, 1.0)

# ==============================================================================
# DATASET: DUAL-STREAM TSN VIDEO DATASET
# ==============================================================================
class XDViolenceDualStreamDataset(Dataset):
    def __init__(self, video_samples, num_segments=Config.NUM_SEGMENTS, target_size=Config.TARGET_SIZE, is_train=False):
        self.video_samples = video_samples
        self.num_segments = num_segments
        self.target_size = target_size
        self.is_train = is_train

    def __len__(self):
        return len(self.video_samples)

    def _sample_indices(self, total_frames):
        if total_frames <= self.num_segments:
            return np.linspace(0, max(0, total_frames - 1), self.num_segments, dtype=int)

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
        raw_frames = []
        for i in ordered_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i))
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            raw_frames.append(frame)
        cap.release()

        while len(raw_frames) < self.num_segments:
            raw_frames.append(raw_frames[-1].copy() if len(raw_frames) > 0 else np.zeros((160, 160, 3), dtype=np.float32))
        raw_frames = raw_frames[:self.num_segments]

        # 1. Global Stream (Full Wide-Angle Context)
        global_frames = [cv2.resize(f, self.target_size) for f in raw_frames]
        
        # 2. Hotspot Zoom Stream (Dynamic Kinetic RoI Focus)
        hotspot_frames = extract_motion_hotspot_box(raw_frames, target_size=self.target_size)

        # Apply Sharpness Enhancement
        sharp_strength = random.uniform(0.2, 0.6) if self.is_train else 0.4
        global_frames = np.array([enhance_cctv_frame(f, sharpness_strength=sharp_strength) for f in global_frames])
        hotspot_frames = np.array([enhance_cctv_frame(f, sharpness_strength=sharp_strength) for f in hotspot_frames])

        # Dynamic Augmentation (Train Mode)
        if self.is_train:
            if random.random() > 0.5:
                global_frames = np.flip(global_frames, axis=2).copy()
                hotspot_frames = np.flip(hotspot_frames, axis=2).copy()
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-0.08, 0.08)
                global_frames = np.clip(global_frames * alpha + beta, 0.0, 1.0)
                hotspot_frames = np.clip(hotspot_frames * alpha + beta, 0.0, 1.0)

        # Motion Residual Blending
        blended_global = compute_motion_residual(global_frames)
        blended_hotspot = compute_motion_residual(hotspot_frames)

        tensor_global = torch.FloatTensor(blended_global).permute(3, 0, 1, 2)   # [C, T, H, W]
        tensor_hotspot = torch.FloatTensor(blended_hotspot).permute(3, 0, 1, 2) # [C, T, H, W]
        label = torch.tensor(class_idx, dtype=torch.long)
        
        return tensor_global, tensor_hotspot, label

# ==============================================================================
# DATASET DISCOVERY & SPLIT
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

    labels = [vi[1] for vi in video_items]
    train_vids, temp_vids = train_test_split(video_items, test_size=0.25, stratify=labels, random_state=SEED)
    temp_labels = [ti[1] for ti in temp_vids]
    val_vids, test_vids = train_test_split(temp_vids, test_size=0.60, stratify=temp_labels, random_state=SEED)

    train_samples = []
    for _ in range(Config.CLIPS_PER_VIDEO):
        train_samples.extend(train_vids)
    random.shuffle(train_samples)

    print(f"\nDataset Splits: {len(train_samples)} Train Samples ({len(train_vids)} unique) | {len(val_vids)} Val Videos | {len(test_vids)} Test Videos")

    total_train = len(train_vids)
    beta = 0.999
    effective_num = [1.0 - np.power(beta, max(1, class_counts[i])) for i in range(NUM_CLASSES)]
    weights = [(1.0 - beta) / np.array(effective_num[i]) for i in range(NUM_CLASSES)]
    weights = np.array(weights) / np.sum(weights) * NUM_CLASSES
    class_weights_tensor = torch.FloatTensor(weights).to(device)

    return train_samples, val_vids, test_vids, class_weights_tensor

# ==============================================================================
# LOSS: CLASS-BALANCED FOCAL LOSS
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
# MODEL: DUAL-STREAM CROSS-ATTENTION VIDEO VISION TRANSFORMER
# ==============================================================================
class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.35):
        super(CrossAttentionFusionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x_global, x_hotspot):
        # Global tokens query Hotspot tokens
        attn_out, _ = self.cross_attn(query=self.norm1(x_global), key=self.norm1(x_hotspot), value=self.norm1(x_hotspot))
        x = x_global + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class DualStreamXDCrimeClassifierViT(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_frames=Config.NUM_SEGMENTS, d_model=512, num_layers=3, num_heads=8):
        super(DualStreamXDCrimeClassifierViT, self).__init__()
        
        # Spatial Feature Extractor (Shared Pretrained MobileNetV3-Large Backbone)
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.spatial_backbone = backbone.features
        
        self.proj_global = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(960, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(p=Config.DROPOUT)
        )
        
        self.proj_hotspot = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(960, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(p=Config.DROPOUT)
        )
        
        # Cross-Attention Stream Fusion
        self.cross_fusion = CrossAttentionFusionBlock(d_model=d_model, nhead=num_heads, dropout=Config.DROPOUT)
        
        # Learnable CLS Token and Temporal Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, d_model))
        self.pos_drop = nn.Dropout(p=Config.DROPOUT)
        
        # Temporal Self-Attention Transformer
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
        
        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, num_classes)
        )
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_global, x_hotspot):
        B, C, T, H, W = x_global.shape
        
        # 1. Global Stream Features
        g_frames = x_global.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        g_feat = self.spatial_backbone(g_frames)
        tokens_global = self.proj_global(g_feat).view(B, T, -1)
        
        # 2. Hotspot Zoom Stream Features
        h_frames = x_hotspot.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        h_feat = self.spatial_backbone(h_frames)
        tokens_hotspot = self.proj_hotspot(h_feat).view(B, T, -1)
        
        # 3. Cross-Attention Stream Fusion
        fused_tokens = self.cross_fusion(tokens_global, tokens_hotspot)
        
        # 4. Temporal Transformer
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tokens, fused_tokens), dim=1)
        x_tokens = self.pos_drop(x_tokens + self.pos_embed)
        
        trans_out = self.norm(self.temporal_transformer(x_tokens))
        cls_rep = trans_out[:, 0]
        logits = self.head(cls_rep)
        return logits

# ==============================================================================
# DUAL-STREAM VIDEO MIXUP
# ==============================================================================
def dual_stream_mixup(g_in, h_in, targets, alpha=Config.MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = g_in.size(0)
    index = torch.randperm(batch_size).to(g_in.device)
    mixed_g = lam * g_in + (1.0 - lam) * g_in[index]
    mixed_h = lam * h_in + (1.0 - lam) * h_in[index]
    targets_a, targets_b = targets, targets[index]
    return mixed_g, mixed_h, targets_a, targets_b, lam

# ==============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(train_loader, desc="Training Dual-Stream VideoViT", unit="batch", leave=False)
    for g_in, h_in, labels in pbar:
        g_in, h_in, labels = g_in.to(device), h_in.to(device), labels.to(device)
        
        use_mixup = random.random() < Config.MIXUP_PROB
        if use_mixup:
            g_in, h_in, labels_a, labels_b, lam = dual_stream_mixup(g_in, h_in, labels)
        
        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(g_in, h_in)
                if use_mixup:
                    loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(g_in, h_in)
            if use_mixup:
                loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * g_in.size(0)
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
        for g_in, h_in, labels in dataloader:
            g_in, h_in, labels = g_in.to(device), h_in.to(device), labels.to(device)
            
            if use_tta:
                out1 = model(g_in, h_in)
                g_flip = torch.flip(g_in, dims=[4])
                h_flip = torch.flip(h_in, dims=[4])
                out2 = model(g_flip, h_flip)
                probs = (torch.softmax(out1, dim=1) + torch.softmax(out2, dim=1)) / 2.0
            else:
                out = model(g_in, h_in)
                probs = torch.softmax(out, dim=1)

            loss = criterion(torch.log(probs.clamp(1e-7, 1.0)), labels)
            running_loss += loss.item() * g_in.size(0)
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

    print("\n" + "=" * 70)
    print("🚨 PHASE 2 (v2): DUAL-STREAM MOTION-HOTSPOT ACTION RECOGNITION")
    print("=" * 70)

    train_samples, val_vids, test_vids, class_weights = discover_xd_dataset()
    if not train_samples:
        print("[Error] No XD-Violence videos found to train.")
        return

    train_dataset = XDViolenceDualStreamDataset(train_samples, num_segments=Config.NUM_SEGMENTS, is_train=True)
    val_dataset   = XDViolenceDualStreamDataset(val_vids, num_segments=Config.NUM_SEGMENTS, is_train=False)
    test_dataset  = XDViolenceDualStreamDataset(test_vids, num_segments=Config.NUM_SEGMENTS, is_train=False)

    def worker_init_fn(worker_id):
        cv2.setNumThreads(1)
        np.random.seed(SEED + worker_id)

    num_workers = 6 if device.type == "cpu" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = DualStreamXDCrimeClassifierViT(
        num_classes=NUM_CLASSES,
        num_frames=Config.NUM_SEGMENTS,
        d_model=512,
        num_layers=3,
        num_heads=8
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nDual-Stream Phase 2 Model Initialized: {total_params:,} parameters across {NUM_CLASSES} Action Classes.")

    backbone_params = list(model.spatial_backbone.parameters())
    transformer_params = [p for n, p in model.named_parameters() if not n.startswith("spatial_backbone")]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': Config.BACKBONE_LR},
        {'params': transformer_params, 'lr': Config.BASE_LR}
    ], weight_decay=Config.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    criterion = ClassBalancedFocalLoss(class_weights=class_weights, gamma=Config.FOCAL_GAMMA, smoothing=Config.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    best_val_acc = 0.0
    patience_counter = 0

    print("\nStarting Dual-Stream XD-Violence Training on GPU...")
    print("=" * 70)

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device, use_tta=False)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{Config.EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  --> Best Dual-Stream Phase 2 Checkpoint Saved! (Val Accuracy: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"\n[Early Stop] Val accuracy did not improve for {Config.EARLY_STOP_PATIENCE} epochs. Stopping.")
                break

    print("\n" + "=" * 70)
    print("EVALUATING BEST DUAL-STREAM MODEL ON TEST SET (WITH TTA)")
    print("=" * 70)

    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    test_loss, test_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device, use_tta=Config.USE_TTA)
    print(f"\nFinal Dual-Stream Phase 2 Test Accuracy: {test_acc:.2f}%\n")

    report = classification_report(y_true, y_pred, target_names=XD_CRIME_CLASSES, digits=4, zero_division=0)
    print("CLASSIFICATION REPORT:\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("CONFUSION MATRIX:\n", cm)


if __name__ == "__main__":
    main()
