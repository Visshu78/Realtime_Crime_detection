# ==============================================================================
# 🚨 STAGE 2: MULTI-CLASS FINE-GRAINED CRIME ACTION CLASSIFIER (SlowFast Net)
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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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
class Config:
    DATASET_DIR = "UCF Dataset"
    TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
    TEST_DIR = os.path.join(DATASET_DIR, "Test")

    OUTPUT_DIR = "Optimisedmodel"
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_slowfast_crime_classifier.pth")

    MAX_FRAMES = 24
    TARGET_SIZE = (128, 128)
    BATCH_SIZE = 8
    EPOCHS = 35
    BASE_LR = 3e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    USE_TTA = True

    EARLY_STOP_PATIENCE = 8

# UCF-Crime Standard Categories (14 Classes)
CRIME_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism", "NormalVideos"
]

NUM_CLASSES = len(CRIME_CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CRIME_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CRIME_CLASSES)}

# ==============================================================================
# DATASET LOADER FOR PRE-EXTRACTED FRAME SEQUENCES
# ==============================================================================
def compute_motion(frames):
    """
    Computes frame-to-frame difference for motion dynamic representation.
    """
    motion = np.abs(np.diff(frames, axis=0))
    first_diff = motion[0:1]
    motion = np.concatenate([first_diff, motion], axis=0)
    blended = 0.7 * frames + 0.3 * motion
    return blended

class UCFCrimeSequenceDataset(Dataset):
    """
    Groups individual frame PNGs by video sequence and samples 24-frame clips.
    """
    def __init__(self, video_groups, max_frames=Config.MAX_FRAMES, target_size=Config.TARGET_SIZE, augment=False):
        self.video_groups = video_groups  # List of (video_prefix, list_of_frame_paths, class_idx)
        self.max_frames = max_frames
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.video_groups)

    def __getitem__(self, idx):
        video_prefix, frame_paths, class_idx = self.video_groups[idx]
        num_avail = len(frame_paths)

        # Sample sequential frames
        if num_avail >= self.max_frames:
            if self.augment:
                max_start = max(0, num_avail - self.max_frames)
                start_idx = random.randint(0, max_start)
            else:
                start_idx = max(0, (num_avail - self.max_frames) // 2)
            selected_paths = frame_paths[start_idx : start_idx + self.max_frames]
        else:
            # Pad with last frame if sequence is short
            selected_paths = list(frame_paths)
            while len(selected_paths) < self.max_frames:
                selected_paths.append(selected_paths[-1])

        # Load frames
        frames = []
        for p in selected_paths:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, self.target_size)
            img = img.astype("float32") / 255.0
            frames.append(img)

        frames = np.array(frames)

        # Spatial Augmentation
        if self.augment:
            if random.random() > 0.5:
                frames = np.flip(frames, axis=2).copy()
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-0.10, 0.10)
                frames = np.clip(frames * alpha + beta, 0.0, 1.0)

        # Compute Spatio-Temporal Motion Blend
        blended = compute_motion(frames)
        # Format: (C, T, H, W)
        tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2)
        label = torch.tensor(class_idx, dtype=torch.long)

        return tensor, label

# ==============================================================================
# DATASET DISCOVERY & GROUPING
# ==============================================================================
def discover_ucf_dataset(dataset_dir=Config.DATASET_DIR):
    """
    Scans UCF Dataset directory, grouping image frames by original video IDs.
    """
    video_groups = []
    
    search_dirs = []
    if os.path.exists(Config.TRAIN_DIR):
        search_dirs.append(Config.TRAIN_DIR)
    if os.path.exists(Config.TEST_DIR):
        search_dirs.append(Config.TEST_DIR)
    if not search_dirs and os.path.exists(dataset_dir):
        search_dirs.append(dataset_dir)

    for base_dir in search_dirs:
        for class_folder in sorted(os.listdir(base_dir)):
            class_path = os.path.join(base_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            # Map folder name to standard class index
            class_name = class_folder
            if class_name not in CLASS_TO_IDX:
                matched = [c for c in CRIME_CLASSES if c.lower() in class_name.lower()]
                if matched:
                    class_name = matched[0]
                else:
                    continue
            
            class_idx = CLASS_TO_IDX[class_name]

            # Group frames by video ID
            video_frame_dict = defaultdict(list)
            for f in sorted(os.listdir(class_path)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    parts = f.rsplit('_', 1)
                    if len(parts) == 2:
                        vid_key = parts[0]
                    else:
                        vid_key = class_name
                    video_frame_dict[vid_key].append(os.path.join(class_path, f))

            for vid_key, frames in video_frame_dict.items():
                if len(frames) >= 4:
                    video_groups.append((vid_key, frames, class_idx))

    print(f"\nDiscovered {len(video_groups)} total video sequences across {len(CRIME_CLASSES)} crime classes.")
    return video_groups

# ==============================================================================
# STAGE 2: SLOWFAST NETWORK ARCHITECTURE
# ==============================================================================
class Bottleneck3D(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=(1, stride, stride), padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 2, kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)

class SlowFastNet(nn.Module):
    """
    SlowFast Network for Fine-Grained Crime Classification:
    - Slow Pathway: Low frame rate (T/4 = 6 frames), High channel capacity (spatial semantics)
    - Fast Pathway: High frame rate (T = 24 frames), Low channel capacity (motion dynamics)
    - Lateral Connections: Time-to-channel strided 3D Conv fusing Fast into Slow
    - Fused 14-Class Output Head
    """
    def __init__(self, num_classes=NUM_CLASSES, alpha=4, beta_inv=8):
        super(SlowFastNet, self).__init__()
        self.alpha = alpha
        
        # --- SLOW PATHWAY (Spatial Semantics, Low Frame Rate) ---
        self.slow_conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        in_slow1 = 64 + (64 // beta_inv) * 2  # 64 + 16 = 80 channels
        self.slow_res1 = Bottleneck3D(in_slow1, 32, downsample=nn.Sequential(
            nn.Conv3d(in_slow1, 64, kernel_size=1, bias=False),
            nn.BatchNorm3d(64)
        ))
        self.slow_res2 = Bottleneck3D(64, 64, stride=2, downsample=nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(128)
        ))
        
        # --- FAST PATHWAY (High-frequency Motion Dynamics, Lightweight Channels) ---
        fast_c = 64 // beta_inv  # 8 channels
        self.fast_conv1 = nn.Sequential(
            nn.Conv3d(3, fast_c, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(fast_c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.fast_res1 = Bottleneck3D(fast_c, fast_c, stride=1, downsample=nn.Sequential(
            nn.Conv3d(fast_c, fast_c * 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(fast_c * 2)
        ))
        self.fast_res2 = Bottleneck3D(fast_c * 2, fast_c * 2, stride=2, downsample=nn.Sequential(
            nn.Conv3d(fast_c * 2, fast_c * 4, kernel_size=1, stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(fast_c * 4)
        ))
        
        # --- LATERAL CONNECTIONS (Fast -> Slow Fusion) ---
        self.lateral1 = nn.Conv3d(fast_c, fast_c * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0), bias=False)
        
        # --- CLASSIFICATION HEAD ---
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(128 + fast_c * 4, num_classes)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x_slow = x[:, :, ::self.alpha, :, :]
        x_fast = x
        
        # Fast Pathway Stage 1
        f_feat = self.fast_conv1(x_fast)
        f_lat = self.lateral1(f_feat)
        
        # Slow Pathway Stage 1 + Lateral Fusion
        s_feat = self.slow_conv1(x_slow)
        s_feat = torch.cat([s_feat, f_lat], dim=1)
        
        # Fast & Slow Residual Stages
        f_feat = self.fast_res1(f_feat)
        f_feat = self.fast_res2(f_feat)
        
        s_feat = self.slow_res1(s_feat)
        s_feat = self.slow_res2(s_feat)
        
        # Global Pooling & Concatenation
        s_pooled = self.pool(s_feat).flatten(1)
        f_pooled = self.pool(f_feat).flatten(1)
        fused = torch.cat([s_pooled, f_pooled], dim=1)
        
        logits = self.head(self.dropout(fused))
        return logits

# Alias for backward compatibility
CrimeClassifierViT = SlowFastNet

# ==============================================================================
# INFERENCE FUNCTION
# ==============================================================================
def predict_crime_type(video_tensor, model_path=Config.BEST_MODEL_PATH):
    """
    Given a 24-frame tensor (1, C, T, H, W), predicts the specific crime type.
    """
    if not os.path.exists(model_path):
        print(f"Error: Multi-Class Checkpoint '{model_path}' not found.")
        return None, 0.0

    model = SlowFastNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(video_tensor.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = np.argmax(probs)
    predicted_crime = IDX_TO_CLASS[top_idx]
    confidence = probs[top_idx]

    return predicted_crime, confidence

# ==============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(train_loader, desc="Training SlowFast Net", unit="batch", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
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
                flipped = torch.flip(inputs, dims=[4])
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
# MAIN TRAINING PIPELINE
# ==============================================================================
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Dry run sanity check
    if "--dry-run" in sys.argv:
        print("\n--- DRY RUN: STAGE 2 SLOWFAST NETWORK ---")
        model = SlowFastNet(num_classes=NUM_CLASSES).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"SlowFast Net Loaded. Total Parameters: {total_params:,}")
        dummy = torch.randn(2, 3, Config.MAX_FRAMES, Config.TARGET_SIZE[0], Config.TARGET_SIZE[1]).to(device)
        out = model(dummy)
        print(f"Forward Pass Output Shape: {out.shape} (Expected: [2, {NUM_CLASSES}])")
        print("Dry run completed successfully.")
        return

    # Discover and split dataset
    video_groups = discover_ucf_dataset()
    if len(video_groups) == 0:
        print(f"\n[Notice] No video sequences discovered in '{Config.DATASET_DIR}'. Please ensure dataset download is complete.")
        return

    labels = [vg[2] for vg in video_groups]
    train_groups, temp_groups = train_test_split(video_groups, test_size=0.20, stratify=labels, random_state=SEED)
    temp_labels = [tg[2] for tg in temp_groups]
    val_groups, test_groups = train_test_split(temp_groups, test_size=0.50, stratify=temp_labels, random_state=SEED)

    print(f"Dataset Split: {len(train_groups)} Train | {len(val_groups)} Val | {len(test_groups)} Test")

    train_dataset = UCFCrimeSequenceDataset(train_groups, augment=True)
    val_dataset   = UCFCrimeSequenceDataset(val_groups, augment=False)
    test_dataset  = UCFCrimeSequenceDataset(test_groups, augment=False)

    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)

    model = SlowFastNet(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nSlowFast Net Initialized: {total_params:,} parameters across {NUM_CLASSES} classes.")

    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    no_improve_count = 0

    print("\nStarting Stage 2 SlowFast Multi-Class Training...")
    print("=" * 65)

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device, use_tta=Config.USE_TTA)
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{Config.EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  --> Best Stage 2 Checkpoint Saved! (Val Accuracy: {best_val_acc:.2f}%)")
        else:
            no_improve_count += 1

        if no_improve_count >= Config.EARLY_STOP_PATIENCE:
            print(f"\n[Early Stop] Val accuracy did not improve for {Config.EARLY_STOP_PATIENCE} epochs. Stopping.")
            break

    # Final Evaluation on Test Set
    print("\n" + "=" * 65)
    print("EVALUATING BEST SLOWFAST STAGE 2 MODEL ON TEST SET (WITH TTA)")
    print("=" * 65)
    
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    
    test_loss, test_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device, use_tta=True)
    
    print(f"\nFinal Stage 2 Multi-Class Test Accuracy: {test_acc:.2f}%")
    print("\nCLASSIFICATION REPORT:")
    present_classes = sorted(list(set(y_true) | set(y_pred)))
    target_names = [IDX_TO_CLASS[i] for i in present_classes]
    print(classification_report(y_true, y_pred, labels=present_classes, target_names=target_names, digits=4))

if __name__ == "__main__":
    main()
