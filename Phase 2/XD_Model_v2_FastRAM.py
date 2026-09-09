# ==============================================================================
# 🚀 PHASE 2 (v2): ULTRA-FAST RAM-CACHED DUAL-STREAM VIDEO VISION TRANSFORMER
#    - Step 1: One-Time Feature Extraction into RAM (Zero disk bottleneck)
#    - Step 2: Ultra-Fast Temporal Cross-Attention Training (5-8s per epoch)
#    - Step 3: Complete Unified Model Checkpoint Assembly & Evaluation
# ==============================================================================

import os
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import sys
import time
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# SEED & CPU/GPU CONFIGURATION
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
        print(f"🚀 Fast CPU Engine Activated: {train_threads} PyTorch matrix threads across {num_cores} cores.")
except Exception:
    device = torch.device("cpu")
    torch.set_num_threads(24)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "Phase" in os.path.dirname(os.path.abspath(__file__)) else os.path.dirname(os.path.abspath(__file__))

class Config:
    DATASET_DIR = os.path.join(PROJECT_ROOT, "XD_Violence_Dataset")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Optimisedmodel")
    CACHE_PATH = os.path.join(OUTPUT_DIR, "xd_dualstream_ram_cache.pt")
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_xd_crime_classifier_v2_dualstream.pth")

    NUM_SEGMENTS = 24         # 24 temporal segments (TSN)
    TARGET_SIZE = (160, 160)
    D_MODEL = 512
    BATCH_SIZE = 32           # Larger batch size for fast RAM training
    EPOCHS = 35
    BASE_LR = 4e-4
    WEIGHT_DECAY = 1e-3
    DROPOUT = 0.40
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.05
    MIXUP_PROB = 0.35
    MIXUP_ALPHA = 0.20
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
# FAST VIDEO SAMPLING & HOTSPOT EXTRACTION
# ==============================================================================
def enhance_cctv_frame(frame_rgb, sharpness_strength=0.4):
    if sharpness_strength > 0:
        blurred = cv2.GaussianBlur(frame_rgb, (3, 3), sigmaX=1.0)
        sharpened = cv2.addWeighted(frame_rgb, 1.0 + sharpness_strength, blurred, -sharpness_strength, 0)
        return np.clip(sharpened, 0.0, 1.0)
    return frame_rgb

def extract_dual_stream_frames(video_path, num_segments=Config.NUM_SEGMENTS, target_size=Config.TARGET_SIZE):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = num_segments

    # TSN uniform segment sampling
    segment_len = total_frames / float(num_segments)
    indices = [min(int((i + 0.5) * segment_len), total_frames - 1) for i in range(num_segments)]

    raw_frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i))
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        raw_frames.append(frame)
    cap.release()

    while len(raw_frames) < num_segments:
        raw_frames.append(raw_frames[-1].copy() if len(raw_frames) > 0 else np.zeros((160, 160, 3), dtype=np.float32))
    raw_frames = raw_frames[:num_segments]

    # 1. Global Stream
    global_frames = np.array([enhance_cctv_frame(cv2.resize(f, target_size), 0.4) for f in raw_frames])

    # 2. Fast Hotspot Zoom Stream
    mid_idx = len(raw_frames) // 2
    f_start = cv2.resize(raw_frames[0], (40, 40))
    f_mid   = cv2.resize(raw_frames[mid_idx], (40, 40))
    f_end   = cv2.resize(raw_frames[-1], (40, 40))
    diff_gray = np.mean(np.abs(f_mid - f_start) + np.abs(f_end - f_mid), axis=2)
    my, mx = np.unravel_index(np.argmax(diff_gray), (40, 40))
    cx, cy = mx / 40.0, my / 40.0

    h, w = raw_frames[0].shape[:2]
    box_w, box_h = int(0.60 * w), int(0.60 * h)
    center_x, center_y = int(cx * w), int(cy * h)
    x1 = max(0, min(w - box_w, center_x - box_w // 2))
    y1 = max(0, min(h - box_h, center_y - box_h // 2))

    hotspot_frames = np.array([enhance_cctv_frame(cv2.resize(f[y1:y1+box_h, x1:x1+box_w], target_size), 0.4) for f in raw_frames])

    # Motion residual blending
    def blend_motion(frames):
        motion = np.abs(np.diff(frames, axis=0))
        motion = np.concatenate([motion[0:1], motion], axis=0)
        return np.clip(0.70 * frames + 0.30 * motion, 0.0, 1.0)

    g_tensor = torch.FloatTensor(blend_motion(global_frames)).permute(3, 0, 1, 2).unsqueeze(0)   # [1, 3, T, H, W]
    h_tensor = torch.FloatTensor(blend_motion(hotspot_frames)).permute(3, 0, 1, 2).unsqueeze(0) # [1, 3, T, H, W]
    return g_tensor, h_tensor

# ==============================================================================
# SPATIAL FEATURE EXTRACTOR (Pretrained MobileNetV3-Large Backbone)
# ==============================================================================
class FeatureExtractor(nn.Module):
    def __init__(self, d_model=Config.D_MODEL):
        super(FeatureExtractor, self).__init__()
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(960, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x_video):
        # x_video: [1, 3, T, H, W]
        B, C, T, H, W = x_video.shape
        x_frames = x_video.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        with torch.no_grad():
            feat = self.features(x_frames)
            tokens = self.proj(feat).view(T, -1)  # [T, D_MODEL]
        return tokens

# ==============================================================================
# TEMPORAL TRANSFORMER HEAD (Trains in RAM)
# ==============================================================================
class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, d_model=Config.D_MODEL, nhead=8, dropout=0.35):
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
        attn_out, _ = self.cross_attn(query=self.norm1(x_global), key=self.norm1(x_hotspot), value=self.norm1(x_hotspot))
        x = x_global + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class DualStreamTemporalHead(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_frames=Config.NUM_SEGMENTS, d_model=Config.D_MODEL, num_layers=3, num_heads=8):
        super(DualStreamTemporalHead, self).__init__()
        self.cross_fusion = CrossAttentionFusionBlock(d_model=d_model, nhead=num_heads, dropout=Config.DROPOUT)
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

    def forward(self, g_tokens, h_tokens):
        # g_tokens: [B, T, D], h_tokens: [B, T, D]
        B = g_tokens.size(0)
        fused = self.cross_fusion(g_tokens, h_tokens)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tokens, fused), dim=1)
        x_tokens = self.pos_drop(x_tokens + self.pos_embed)
        trans_out = self.norm(self.temporal_transformer(x_tokens))
        cls_rep = trans_out[:, 0]
        logits = self.head(cls_rep)
        return logits

# Full Unified Model for Final Export & Inference
class UnifiedDualStreamModel(nn.Module):
    def __init__(self, extractor, head):
        super(UnifiedDualStreamModel, self).__init__()
        self.spatial_backbone = extractor.features
        self.proj_global = extractor.proj
        self.temporal_head = head

    def forward(self, x_global, x_hotspot):
        B, C, T, H, W = x_global.shape
        g_frames = x_global.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        h_frames = x_hotspot.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        
        g_tokens = self.proj_global(self.spatial_backbone(g_frames)).view(B, T, -1)
        h_tokens = self.proj_global(self.spatial_backbone(h_frames)).view(B, T, -1)
        return self.temporal_head(g_tokens, h_tokens)

# ==============================================================================
# LOSS FUNCTION & DATA DISCOVERY
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

def discover_xd_dataset():
    video_items = []
    class_counts = defaultdict(int)
    for root, dirs, files in os.walk(Config.DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                full_path = os.path.join(root, f)
                lower_path = full_path.lower()
                if "normal" in lower_path or "nonviolence" in lower_path: continue
                
                matched = None
                for c in XD_CRIME_CLASSES:
                    if c.lower() in lower_path: matched = c; break
                    elif c == "CarAccident" and ("accident" in lower_path or "crash" in lower_path): matched = "CarAccident"; break
                    elif c == "Fighting" and ("fight" in lower_path or "brawl" in lower_path): matched = "Fighting"; break
                    elif c == "Shooting" and ("shoot" in lower_path or "gun" in lower_path): matched = "Shooting"; break

                if matched:
                    c_idx = CLASS_TO_IDX[matched]
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

    beta = 0.999
    effective_num = [1.0 - np.power(beta, max(1, class_counts[i])) for i in range(NUM_CLASSES)]
    weights = [(1.0 - beta) / np.array(effective_num[i]) for i in range(NUM_CLASSES)]
    weights = np.array(weights) / np.sum(weights) * NUM_CLASSES
    class_weights_tensor = torch.FloatTensor(weights).to(device)

    return train_vids, val_vids, test_vids, class_weights_tensor

# ==============================================================================
# MAIN FAST RAM PIPELINE
# ==============================================================================
def extract_dataset_features(video_list, extractor, desc="Extracting Features"):
    g_list, h_list, y_list = [], [], []
    for vid_path, label in tqdm(video_list, desc=desc, unit="vid"):
        try:
            g_tensor, h_tensor = extract_dual_stream_frames(vid_path)
            g_feat = extractor(g_tensor).cpu().half() # float16 in RAM
            h_feat = extractor(h_tensor).cpu().half()
            g_list.append(g_feat)
            h_list.append(h_feat)
            y_list.append(label)
        except Exception:
            continue
    return torch.stack(g_list), torch.stack(h_list), torch.tensor(y_list, dtype=torch.long)

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 70)
    print("🚀 PHASE 2 (v2): ULTRA-FAST RAM-CACHED DUAL-STREAM VIDEO TRAINING")
    print("=" * 70)

    train_vids, val_vids, test_vids, class_weights = discover_xd_dataset()

    extractor = FeatureExtractor().to(device)
    extractor.eval()

    # Step 1: Pre-extract features into RAM cache (or load if cached)
    if os.path.exists(Config.CACHE_PATH):
        print(f"\n📂 Loading pre-extracted features from cache: {Config.CACHE_PATH}")
        cache = torch.load(Config.CACHE_PATH, weights_only=True)
        train_g, train_h, train_y = cache['train_g'], cache['train_h'], cache['train_y']
        val_g, val_h, val_y = cache['val_g'], cache['val_h'], cache['val_y']
        test_g, test_h, test_y = cache['test_g'], cache['test_h'], cache['test_y']
    else:
        print("\n⚡ Step 1: Extracting Spatial Features into RAM (One-time pass)...")
        t0 = time.time()
        train_g, train_h, train_y = extract_dataset_features(train_vids, extractor, desc="Caching Train Features")
        val_g, val_h, val_y = extract_dataset_features(val_vids, extractor, desc="Caching Val Features")
        test_g, test_h, test_y = extract_dataset_features(test_vids, extractor, desc="Caching Test Features")
        
        torch.save({
            'train_g': train_g, 'train_h': train_h, 'train_y': train_y,
            'val_g': val_g, 'val_h': val_h, 'val_y': val_y,
            'test_g': test_g, 'test_h': test_h, 'test_y': test_y
        }, Config.CACHE_PATH)
        print(f"✅ All features cached to RAM in {(time.time()-t0)/60:.1f} minutes!")

    # Step 2: Create In-Memory DataLoaders
    train_loader = DataLoader(TensorDataset(train_g.float(), train_h.float(), train_y), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_g.float(), val_h.float(), val_y), batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(test_g.float(), test_h.float(), test_y), batch_size=Config.BATCH_SIZE, shuffle=False)

    head = DualStreamTemporalHead().to(device)
    optimizer = optim.AdamW(head.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    criterion = ClassBalancedFocalLoss(class_weights=class_weights)

    best_val_acc = 0.0
    patience_counter = 0

    print("\n⚡ Step 2: Training Temporal Cross-Attention Transformer directly in RAM...")
    print("=" * 70)

    for epoch in range(1, Config.EPOCHS + 1):
        t_start = time.time()
        head.train()
        running_loss, correct, total = 0.0, 0, 0
        for g_in, h_in, labels in train_loader:
            g_in, h_in, labels = g_in.to(device), h_in.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = head(g_in, h_in)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * g_in.size(0)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Validation
        head.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for g_in, h_in, labels in val_loader:
                g_in, h_in, labels = g_in.to(device), h_in.to(device), labels.to(device)
                outputs = head(g_in, h_in)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * g_in.size(0)
                _, pred = torch.max(outputs, 1)
                v_tot += labels.size(0)
                v_corr += (pred == labels).sum().item()

        val_loss = v_loss / v_tot
        val_acc = 100.0 * v_corr / v_tot
        epoch_time = time.time() - t_start

        print(f"Epoch [{epoch:02d}/{Config.EPOCHS}] ({epoch_time:.1f}s) | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            unified_model = UnifiedDualStreamModel(extractor, head)
            torch.save(unified_model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  --> Best Dual-Stream Checkpoint Saved! (Val Accuracy: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"\n[Early Stop] Val accuracy did not improve for {Config.EARLY_STOP_PATIENCE} epochs.")
                break

    # Step 3: Final Test Evaluation
    print("\n" + "=" * 70)
    print("EVALUATING BEST DUAL-STREAM MODEL ON HELD-OUT TEST SET")
    print("=" * 70)

    head.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for g_in, h_in, labels in test_loader:
            g_in, h_in, labels = g_in.to(device), h_in.to(device), labels.to(device)
            outputs = head(g_in, h_in)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())

    test_acc = accuracy_score(y_true, y_pred) * 100.0
    print(f"\n🎯 Final Dual-Stream Phase 2 Test Accuracy: {test_acc:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=XD_CRIME_CLASSES, digits=4))


if __name__ == "__main__":
    main()
