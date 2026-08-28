# ==============================================================================
# 🚨 PHASE 2: 6-CLASS FINE-GRAINED CRIME ACTION CLASSIFIER (XD-Violence VideoViT)
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

    MAX_FRAMES = 24
    TARGET_SIZE = (160, 160)
    CLIPS_PER_VIDEO = 6
    BATCH_SIZE = 16
    EPOCHS = 35
    BASE_LR = 3e-4
    BACKBONE_LR = 3e-5
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.35
    LABEL_SMOOTHING = 0.05
    USE_TTA = True

    EARLY_STOP_PATIENCE = 8

# 6 Distinct XD-Violence Crime Action Classes (Excluding Normal)
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
# DATASET LOADER FOR VIDEO FILES / CLIPS
# ==============================================================================
def compute_motion(frames):
    motion = np.abs(np.diff(frames, axis=0))
    first_diff = motion[0:1]
    motion = np.concatenate([first_diff, motion], axis=0)
    blended = 0.70 * frames + 0.30 * motion
    return blended

class XDViolenceDataset(Dataset):
    def __init__(self, video_items, max_frames=Config.MAX_FRAMES, target_size=Config.TARGET_SIZE, augment=False):
        self.video_items = video_items  # List of (video_path, class_idx)
        self.max_frames = max_frames
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.video_items)

    def __getitem__(self, idx):
        video_path, class_idx = self.video_items[idx]
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        if total_frames <= self.max_frames:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, self.target_size).astype("float32") / 255.0
                frames.append(frame)
            cap.release()
            while len(frames) < self.max_frames:
                frames.append(frames[-1] if len(frames) > 0 else np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32))
        else:
            if self.augment:
                start_frame = random.randint(0, total_frames - self.max_frames)
            else:
                start_frame = (total_frames - self.max_frames) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(self.max_frames):
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, self.target_size).astype("float32") / 255.0
                frames.append(frame)
            cap.release()
            while len(frames) < self.max_frames:
                frames.append(frames[-1] if len(frames) > 0 else np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32))

        frames = np.array(frames[:self.max_frames])

        if self.augment:
            if random.random() > 0.5:
                frames = np.flip(frames, axis=2).copy()
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-0.10, 0.10)
                frames = np.clip(frames * alpha + beta, 0.0, 1.0)
            if random.random() > 0.5 and len(frames) == self.max_frames:
                idx_pool = sorted(random.sample(range(self.max_frames), self.max_frames - 2))
                idx_pool = [idx_pool[0]] + idx_pool + [idx_pool[-1]]
                frames = frames[idx_pool]

        blended = compute_motion(frames)
        tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2)
        label = torch.tensor(class_idx, dtype=torch.long)
        return tensor, label

# ==============================================================================
# DATASET DISCOVERY
# ==============================================================================
def discover_xd_dataset(dataset_dir=Config.DATASET_DIR):
    video_items = []
    if not os.path.exists(dataset_dir):
        print(f"[Notice] Dataset directory '{dataset_dir}' not found yet. Please ensure download completes.")
        return video_items

    # Scan for folders / video files matching the 6 classes
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                full_path = os.path.join(root, f)
                lower_path = full_path.lower()
                
                # Exclude normal / non-violence videos (Phase 1 handles normal)
                if "normal" in lower_path or "nonviolence" in lower_path:
                    continue
                
                # Match against the 6 XD-Violence crime classes
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
                    class_idx = CLASS_TO_IDX[matched_class]
                    video_items.append((full_path, class_idx))

    print(f"\nDiscovered {len(video_items)} total Crime Videos across {len(XD_CRIME_CLASSES)} XD-Violence Categories.")
    return video_items

# ==============================================================================
# MODEL ARCHITECTURE (VideoViT)
# ==============================================================================
class XDCrimeClassifierViT(nn.Module):
    """
    6-Class Video Vision Transformer for XD-Violence Action Recognition:
    - Pretrained MobileNetV3-Large Spatial Feature Tokenizer (960D -> 512D)
    - 1D Temporal Convolution (Continuous Kinematic Motions)
    - 3-Layer 8-Head Temporal Transformer with Pre-LN LayerNorm
    - 6-Class Classification Head
    """
    def __init__(self, num_classes=NUM_CLASSES, num_frames=Config.MAX_FRAMES, d_model=512, num_layers=3, num_heads=8):
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
# INFERENCE FUNCTION
# ==============================================================================
def predict_xd_crime_type(video_tensor, model_path=Config.BEST_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint '{model_path}' not found.")
        return None, 0.0

    model = XDCrimeClassifierViT(num_classes=NUM_CLASSES).to(device)
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
    
    pbar = tqdm(train_loader, desc="Training XD-Violence VideoViT", unit="batch", leave=False)
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

    if "--dry-run" in sys.argv:
        print("\n--- DRY RUN: XD-VIOLENCE 6-CLASS VIDEO TRANSFORMER ---")
        model = XDCrimeClassifierViT(num_classes=NUM_CLASSES).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Loaded: {total_params:,} parameters across {NUM_CLASSES} Action Classes.")
        dummy = torch.randn(2, 3, Config.MAX_FRAMES, Config.TARGET_SIZE[0], Config.TARGET_SIZE[1]).to(device)
        out = model(dummy)
        print(f"Forward Pass Output Shape: {out.shape} (Expected: [2, {NUM_CLASSES}])")
        print("Dry run completed successfully.")
        return

    video_items = discover_xd_dataset()
    if len(video_items) == 0:
        print(f"\n[Notice] No video files found in '{Config.DATASET_DIR}'. Ensure download is complete.")
        return

    labels = [vi[1] for vi in video_items]
    train_items, temp_items = train_test_split(video_items, test_size=0.20, stratify=labels, random_state=SEED)
    temp_labels = [ti[1] for ti in temp_items]
    val_items, test_items = train_test_split(temp_items, test_size=0.50, stratify=temp_labels, random_state=SEED)

    print(f"Dataset Split: {len(train_items)} Train | {len(val_items)} Val | {len(test_items)} Test Videos")

    train_dataset = XDViolenceDataset(train_items, augment=True)
    val_dataset   = XDViolenceDataset(val_items, augment=False)
    test_dataset  = XDViolenceDataset(test_items, augment=False)

    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)

    model = XDCrimeClassifierViT(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nPhase 2 XD Model Initialized: {total_params:,} parameters across {NUM_CLASSES} Action Classes.")

    train_labels = [ti[1] for ti in train_items]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    total_samples = len(train_labels)
    class_weights = total_samples / (NUM_CLASSES * np.maximum(class_counts, 1).astype(np.float32))
    weights_tensor = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=Config.LABEL_SMOOTHING)
    
    optimizer = optim.AdamW([
        {'params': model.spatial_backbone.parameters(), 'lr': Config.BACKBONE_LR},
        {'params': model.proj.parameters(), 'lr': Config.BASE_LR},
        {'params': model.temporal_conv.parameters(), 'lr': Config.BASE_LR},
        {'params': model.temporal_transformer.parameters(), 'lr': Config.BASE_LR},
        {'params': model.head.parameters(), 'lr': Config.BASE_LR},
        {'params': [model.cls_token, model.pos_embed], 'lr': Config.BASE_LR}
    ], weight_decay=Config.WEIGHT_DECAY)

    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, Config.EPOCHS - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    no_improve_count = 0

    print("\nStarting Phase 2 XD-Violence Action Training...")
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
            print(f"  --> Best Phase 2 Checkpoint Saved! (Val Accuracy: {best_val_acc:.2f}%)")
        else:
            no_improve_count += 1

        if no_improve_count >= Config.EARLY_STOP_PATIENCE:
            print(f"\n[Early Stop] Val accuracy did not improve for {Config.EARLY_STOP_PATIENCE} epochs. Stopping.")
            break

    # Final Evaluation on Test Set
    print("\n" + "=" * 65)
    print("EVALUATING BEST XD-VIOLENCE MODEL ON TEST SET (WITH TTA)")
    print("=" * 65)
    
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    
    test_loss, test_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device, use_tta=True)
    
    print(f"\nFinal Phase 2 XD-Violence Test Accuracy: {test_acc:.2f}%")
    print("\nCLASSIFICATION REPORT:")
    present_classes = sorted(list(set(y_true) | set(y_pred)))
    target_names = [IDX_TO_CLASS[i] for i in present_classes]
    print(classification_report(y_true, y_pred, labels=present_classes, target_names=target_names, digits=4))

if __name__ == "__main__":
    main()
