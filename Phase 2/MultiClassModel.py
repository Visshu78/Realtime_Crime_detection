# ==============================================================================
# 🚨 PHASE 2: 13-CLASS FINE-GRAINED CRIME CLASSIFIER (Multi-Clip VideoViT)
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
    DATASET_DIR = os.path.join(PROJECT_ROOT, "UCF Dataset")
    TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
    TEST_DIR = os.path.join(DATASET_DIR, "Test")

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Optimisedmodel")
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_multiclass_crime_classifier.pth")

    MAX_FRAMES = 24
    TARGET_SIZE = (128, 128)
    CLIPS_PER_VIDEO = 8       # Multi-clip sampling: 8 diverse 24-frame clips per video
    BATCH_SIZE = 16
    EPOCHS = 30
    BASE_LR = 3e-4
    BACKBONE_LR = 3e-5
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    USE_TTA = True

    EARLY_STOP_PATIENCE = 7

# 13 Pure Crime Action Categories (Excluding NormalVideos)
CRIME_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

NUM_CLASSES = len(CRIME_CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CRIME_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CRIME_CLASSES)}

# ==============================================================================
# DATASET LOADER WITH MULTI-CLIP SLICING
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

class MultiClipUCFDataset(Dataset):
    """
    Samples multiple temporal slices across each video sequence to generate
    thousands of rich training clips from the 950 crime videos.
    """
    def __init__(self, clip_samples, max_frames=Config.MAX_FRAMES, target_size=Config.TARGET_SIZE, augment=False):
        self.clip_samples = clip_samples  # List of (list_of_24_frame_paths, class_idx)
        self.max_frames = max_frames
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.clip_samples)

    def __getitem__(self, idx):
        frame_paths, class_idx = self.clip_samples[idx]

        frames = []
        for p in frame_paths:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, self.target_size)
            img = img.astype("float32") / 255.0
            frames.append(img)

        frames = np.array(frames)

        # Spatial & Photometric Augmentation
        if self.augment:
            if random.random() > 0.5:
                frames = np.flip(frames, axis=2).copy()
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-0.10, 0.10)
                frames = np.clip(frames * alpha + beta, 0.0, 1.0)

        # Spatio-Temporal Motion Blend
        blended = compute_motion(frames)
        tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2)
        label = torch.tensor(class_idx, dtype=torch.long)

        return tensor, label

# ==============================================================================
# DATASET DISCOVERY & MULTI-CLIP SLICING
# ==============================================================================
def discover_and_slice_dataset(dataset_dir=Config.DATASET_DIR, clips_per_video=Config.CLIPS_PER_VIDEO):
    """
    Discovers all 13 crime classes (excluding NormalVideos) and slices each video
    into multiple temporal windows.
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
            if class_folder.lower() == "normalvideos":
                continue  # Skip NormalVideos (Phase 1 handles Normal vs Crime)
            
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

            # Group frames by original video ID
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

    print(f"\nDiscovered {len(video_groups)} total Crime Video Sequences across {len(CRIME_CLASSES)} Crime Categories.")
    
    # Split at the video level (no data leakage)
    labels = [vg[2] for vg in video_groups]
    train_vids, temp_vids = train_test_split(video_groups, test_size=0.20, stratify=labels, random_state=SEED)
    temp_labels = [tg[2] for tg in temp_vids]
    val_vids, test_vids = train_test_split(temp_vids, test_size=0.50, stratify=temp_labels, random_state=SEED)

    # Slice videos into multiple clips
    def generate_clips(vids, num_clips, is_train=True):
        clip_list = []
        for vid_key, frame_paths, class_idx in vids:
            num_avail = len(frame_paths)
            if num_avail <= Config.MAX_FRAMES:
                # Pad
                paths = list(frame_paths)
                while len(paths) < Config.MAX_FRAMES:
                    paths.append(paths[-1])
                clip_list.append((paths, class_idx))
            else:
                # Generate multiple temporal slices
                step = max(1, (num_avail - Config.MAX_FRAMES) // max(1, num_clips - 1))
                for i in range(num_clips):
                    start = min(i * step, num_avail - Config.MAX_FRAMES)
                    if is_train and start > 0:
                        start = max(0, start + random.randint(-2, 2))
                        start = min(start, num_avail - Config.MAX_FRAMES)
                    slice_paths = frame_paths[start : start + Config.MAX_FRAMES]
                    clip_list.append((slice_paths, class_idx))
        return clip_list

    train_clips = generate_clips(train_vids, clips_per_video, is_train=True)
    val_clips   = generate_clips(val_vids, max(2, clips_per_video // 2), is_train=False)
    test_clips  = generate_clips(test_vids, max(2, clips_per_video // 2), is_train=False)

    print(f"Generated Slices: {len(train_clips)} Train Clips | {len(val_clips)} Val Clips | {len(test_clips)} Test Clips")
    return train_clips, val_clips, test_clips

# ==============================================================================
# PHASE 2: PRETRAINED VIDEO VISION TRANSFORMER ARCHITECTURE
# ==============================================================================
class CrimeClassifierViT(nn.Module):
    """
    Multi-Class Video Vision Transformer for Fine-Grained Crime Classification:
    - Pretrained MobileNetV3-Large Spatial Feature Tokenizer (960D -> 512D)
    - 1D Temporal Depthwise Conv (Local 3-frame continuous kinematics)
    - 8-Head Temporal Multi-Head Self-Attention Transformer (Pre-LN)
    - 13-Class Classification Output Head
    """
    def __init__(self, num_classes=NUM_CLASSES, num_frames=Config.MAX_FRAMES, d_model=512, num_layers=3, num_heads=8):
        super(CrimeClassifierViT, self).__init__()
        
        # 1. Spatial Tokenizer Backbone
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.spatial_backbone = backbone.features
        
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(960, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 2. Local 1D Temporal Convolution
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        
        # 3. Learnable [CLS] Token & Positional Encodings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, d_model))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # 4. Temporal Multi-Head Self-Attention Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.2,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
        # 5. 13-Class Classification Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # Input: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x_frames = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        
        # Extract Spatial Visual Tokens
        feat = self.spatial_backbone(x_frames)
        tokens = self.proj(feat).view(B, T, -1)
        
        # Local Temporal Motion Refinement
        tokens = tokens + self.temporal_conv(tokens.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Prepend [CLS] token & Positional Encodings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tokens, tokens), dim=1)
        x_tokens = self.pos_drop(x_tokens + self.pos_embed)
        
        # Global Temporal Self-Attention
        trans_out = self.norm(self.temporal_transformer(x_tokens))
        
        # Logits from [CLS] representation
        cls_rep = trans_out[:, 0]
        logits = self.head(cls_rep)
        return logits

# Alias for backward compatibility
SlowFastNet = CrimeClassifierViT

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

    model = CrimeClassifierViT(num_classes=NUM_CLASSES).to(device)
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
    
    pbar = tqdm(train_loader, desc="Training VideoViT Phase 2", unit="batch", leave=False)
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
        print("\n--- DRY RUN: PHASE 2 VIDEO VISION TRANSFORMER ---")
        model = CrimeClassifierViT(num_classes=NUM_CLASSES).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Phase 2 Model Loaded. Total Parameters: {total_params:,}")
        dummy = torch.randn(2, 3, Config.MAX_FRAMES, Config.TARGET_SIZE[0], Config.TARGET_SIZE[1]).to(device)
        out = model(dummy)
        print(f"Forward Pass Output Shape: {out.shape} (Expected: [2, {NUM_CLASSES}])")
        print("Dry run completed successfully.")
        return

    # Discover and multi-clip slice dataset
    train_clips, val_clips, test_clips = discover_and_slice_dataset()
    if len(train_clips) == 0:
        print(f"\n[Notice] No video clips discovered in '{Config.DATASET_DIR}'.")
        return

    train_dataset = MultiClipUCFDataset(train_clips, augment=True)
    val_dataset   = MultiClipUCFDataset(val_clips, augment=False)
    test_dataset  = MultiClipUCFDataset(test_clips, augment=False)

    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)

    model = CrimeClassifierViT(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nPhase 2 Model Initialized: {total_params:,} parameters across {NUM_CLASSES} Crime Categories.")

    # Calculate class weights for balanced learning
    train_labels = [c[1] for c in train_clips]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    total_samples = len(train_labels)
    class_weights = total_samples / (NUM_CLASSES * np.maximum(class_counts, 1).astype(np.float32))
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Applied Class-Balanced Loss Weights.")

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

    print("\nStarting Phase 2 (13-Class VideoViT) Training...")
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
    print("EVALUATING BEST PHASE 2 MODEL ON TEST SET (WITH TTA)")
    print("=" * 65)
    
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    
    test_loss, test_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device, use_tta=True)
    
    print(f"\nFinal Phase 2 Multi-Class Test Accuracy: {test_acc:.2f}%")
    print("\nCLASSIFICATION REPORT:")
    present_classes = sorted(list(set(y_true) | set(y_pred)))
    target_names = [IDX_TO_CLASS[i] for i in present_classes]
    print(classification_report(y_true, y_pred, labels=present_classes, target_names=target_names, digits=4))

if __name__ == "__main__":
    main()
