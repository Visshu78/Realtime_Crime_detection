# ==============================================================================
# 🚨 INTELLIGENT REAL-TIME CRIME DETECTION SYSTEM - OPTIMIZED VideoViT v2
# ==============================================================================

import os
import sys

# Configure GPU MIG compatibility environment flags
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    torch.set_num_threads(12)
    print(f"Using CPU for execution ({torch.get_num_threads()} parallel OpenMP CPU threads).")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    ROOT_DATA = "Dataset/"
    VIOLENCE_DIR = os.path.join(ROOT_DATA, "Violence")
    NON_VIOLENCE_DIR = os.path.join(ROOT_DATA, "NonViolence")

    OUTPUT_DIR = "Optimisedmodel"
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_3dcnn_crime_detector.pth")

    MAX_FRAMES = 24
    TARGET_SIZE = (128, 128)
    BATCH_SIZE = 8
    EPOCHS = 35
    BASE_LR = 3e-4
    BACKBONE_LR = 3e-5
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    USE_TTA = True               # Test-Time Augmentation (flips video horizontally for perspective invariance)

    # Early Stopping Settings
    EARLY_STOP_PATIENCE = 8       # Stop if val acc doesn't improve for 8 consecutive epochs
    OVERFIT_GAP_THRESHOLD = 15.0  # Stop if train_acc > val_acc by more than 15%

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv')

# ==============================================================================
# DATASET LOADER & ADVANCED AUGMENTATION
# ==============================================================================
def load_video_frames(video_path, target_size=Config.TARGET_SIZE, max_frames=Config.MAX_FRAMES, augment=False):
    """
    Reads video file and extracts frames with optional spatiotemporal augmentations.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Temporal sampling jitter for training
        if augment and total_frames > max_frames:
            max_start = max(0, total_frames - max_frames)
            start_idx = random.randint(0, max_start)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frame = frame.astype("float32") / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return None

        # Pad remaining frames if video is shorter
        while len(frames) < max_frames:
            frames.append(frames[-1])

        frames = np.array(frames[:max_frames])

        # Advanced Spatial & Color Augmentations
        if augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                frames = np.flip(frames, axis=2).copy()
            # Random Brightness & Contrast
            if random.random() > 0.5:
                alpha = random.uniform(0.80, 1.20)
                beta = random.uniform(-0.10, 0.10)
                frames = np.clip(frames * alpha + beta, 0.0, 1.0)
            # Random Temporal Subsampling Jitter (skip occasional frame)
            if random.random() > 0.7:
                idx = np.random.choice(max_frames, size=max_frames, replace=True)
                idx.sort()
                frames = frames[idx]

        return frames
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None

def compute_motion(frames):
    """
    Computes frame-to-frame difference for motion dynamic representation.
    """
    motion = np.abs(np.diff(frames, axis=0))
    first_diff = motion[0:1]
    motion = np.concatenate([first_diff, motion], axis=0)
    # Blend 70% RGB with 30% motion difference
    blended = 0.7 * frames + 0.3 * motion
    return blended

class LazyVideoDataset(Dataset):
    def __init__(self, video_paths, labels, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frames = load_video_frames(self.video_paths[idx], augment=self.augment)
        if frames is None:
            frames_tensor = torch.zeros((3, Config.MAX_FRAMES, Config.TARGET_SIZE[0], Config.TARGET_SIZE[1]))
        else:
            blended = compute_motion(frames)
            # PyTorch Video format: (C, T, H, W)
            frames_tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2)

        label_tensor = torch.FloatTensor([self.labels[idx]])
        return frames_tensor, label_tensor

# ==============================================================================
# STRATIFIED DATASET PREPARATION
# ==============================================================================
def prepare_dataset():
    def get_files(folder):
        return sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ])

    vio_files = get_files(Config.VIOLENCE_DIR)
    non_files = get_files(Config.NON_VIOLENCE_DIR)

    all_paths = vio_files + non_files
    all_labels = [1] * len(vio_files) + [0] * len(non_files)

    # 80% train, 20% temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.20, stratify=all_labels, random_state=SEED
    )

    # Split 20% temp into 10% val and 10% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED
    )

    print(f"Dataset Split Summary:")
    print(f"  Total Videos      : {len(all_paths)}")
    print(f"  Training Set      : {len(train_paths)} samples")
    print(f"  Validation Set    : {len(val_paths)} samples")
    print(f"  Test Set          : {len(test_paths)} samples")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

# ==============================================================================
# OPTIMIZED HYBRID SPATIO-TEMPORAL VISION TRANSFORMER (VideoViT v2)
# ==============================================================================
class VideoViT(nn.Module):
    """
    Optimized Video Vision Transformer (VideoViT v2):
    - Pretrained MobileNetV3-Large Spatial Feature Tokenizer (960 dims -> 512 proj)
    - 1D Temporal Depthwise Convolution for smooth local continuous motion
    - 8-Head Temporal Multi-Head Self-Attention Transformer Encoder (3 Layers, Pre-LN)
    - [CLS] Token Classification Head with GELU + Dropout
    """
    def __init__(self, num_frames=Config.MAX_FRAMES, d_model=512, num_layers=3, num_heads=8):
        super(VideoViT, self).__init__()
        
        # 1. Pretrained MobileNetV3-Large Spatial Tokenizer
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.spatial_backbone = backbone.features # Outputs 960 feature channels
        
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(960, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 2. Local 1D Temporal Convolution (Smooth motion tracking across 3 adjacent frames)
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        
        # 3. Learnable [CLS] Token & Temporal Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, d_model))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # 4. Global Temporal Multi-Head Self-Attention Transformer
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
        
        # 5. Classification Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # Input shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x_frames = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        
        # Extract Spatial Visual Tokens
        feat = self.spatial_backbone(x_frames)
        tokens = self.proj(feat).view(B, T, -1) # Shape: (B, T, d_model)
        
        # Local Temporal Motion Refinement (Residual 1D Conv)
        tokens = tokens + self.temporal_conv(tokens.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Prepend [CLS] token & add Temporal Positional Encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tokens, tokens), dim=1)
        x_tokens = self.pos_drop(x_tokens + self.pos_embed)
        
        # Global Temporal Self-Attention across frames
        trans_out = self.norm(self.temporal_transformer(x_tokens))
        
        # Classify using [CLS] token representation
        cls_rep = trans_out[:, 0]
        logits = self.head(cls_rep)
        return logits

# Aliases for backward compatibility
CNN3D_ResSE = VideoViT
CNN3D = VideoViT

# ==============================================================================
# INFERENCE WITH TEST-TIME AUGMENTATION (TTA)
# ==============================================================================
def predict_video(video_path, model_path=Config.BEST_MODEL_PATH, use_tta=True):
    """
    Runs real-time crime detection inference on a single video file with TTA.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found.")
        return None

    model = VideoViT().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    frames = load_video_frames(video_path, augment=False)
    if frames is None:
        print(f"Error: Unable to load video '{video_path}'")
        return None

    blended = compute_motion(frames)
    tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_tta:
            logits_orig = model(tensor)
            flipped_tensor = torch.flip(tensor, dims=[4])
            logits_flip = model(flipped_tensor)
            prob = ((torch.sigmoid(logits_orig) + torch.sigmoid(logits_flip)) / 2.0).item()
        else:
            logits = model(tensor)
            prob = torch.sigmoid(logits).item()
            
        is_crime = prob > 0.50

    label = "CRIME / VIOLENCE DETECTED 🚨" if is_crime else "NORMAL / NON-VIOLENT ✅"
    confidence = prob if is_crime else (1.0 - prob)

    print("\n" + "=" * 55)
    print(f"INFERENCE RESULT: {video_path}")
    print("=" * 55)
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print(f"Raw Output : {prob:.4f}")
    print("=" * 55)

    return is_crime, confidence

# ==============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(train_loader, desc="Training", unit="batch", leave=False)
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
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
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
                probs = (torch.sigmoid(outputs1) + torch.sigmoid(outputs2)) / 2.0
            else:
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)

            loss = criterion(torch.logit(probs.clamp(1e-6, 1.0 - 1e-6)), labels)
            running_loss += loss.item() * inputs.size(0)
            predicted = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())

    acc = 100.0 * correct / total
    avg_loss = running_loss / total
    return avg_loss, acc, np.array(y_true), np.array(y_pred)

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # CLI Predict Check
    if len(sys.argv) > 2 and sys.argv[1] == "--predict":
        predict_video(sys.argv[2])
        return

    # Dry Run check flag
    if "--dry-run" in sys.argv:
        print("\n--- DRY RUN SANITY CHECK ---")
        model = VideoViT().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Video Vision Transformer Loaded. Total Parameters: {total_params:,}")
        dummy_input = torch.randn(2, 3, Config.MAX_FRAMES, Config.TARGET_SIZE[0], Config.TARGET_SIZE[1]).to(device)
        out = model(dummy_input)
        print(f"Forward pass output shape: {out.shape}")
        print("Dry run completed successfully.")
        return

    # Prepare datasets
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_dataset()

    train_dataset = LazyVideoDataset(train_paths, train_labels, augment=True)
    val_dataset   = LazyVideoDataset(val_paths, val_labels, augment=False)
    test_dataset  = LazyVideoDataset(test_paths, test_labels, augment=False)

    num_workers = 0

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # Model & Differential Optimization
    model = VideoViT().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Initialized: Optimized VideoViT v2 (Total Parameters: {total_params:,})")

    criterion = nn.BCEWithLogitsLoss()
    
    # Differential Learning Rate: fine-tune backbone gently, train transformer aggressively
    optimizer = optim.AdamW([
        {'params': model.spatial_backbone.parameters(), 'lr': Config.BACKBONE_LR},
        {'params': model.proj.parameters(), 'lr': Config.BASE_LR},
        {'params': model.temporal_conv.parameters(), 'lr': Config.BASE_LR},
        {'params': model.temporal_transformer.parameters(), 'lr': Config.BASE_LR},
        {'params': model.head.parameters(), 'lr': Config.BASE_LR},
        {'params': [model.cls_token, model.pos_embed], 'lr': Config.BASE_LR}
    ], weight_decay=Config.WEIGHT_DECAY)

    # Linear Warmup (2 epochs) followed by Cosine Annealing
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

    print("\nStarting Training Pipeline...")
    print("=" * 65)

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device, use_tta=Config.USE_TTA)
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{Config.EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  --> Best Checkpoint Saved! (Val Accuracy: {best_val_acc:.2f}%)")
        else:
            no_improve_count += 1

        # ─── Early Stopping Logic ─────────────────────────────────────────────
        overfit_gap = train_acc - val_acc
        if overfit_gap > Config.OVERFIT_GAP_THRESHOLD and epoch > 15:
            print(f"\n[Early Stop] Severe overfitting detected! Train Acc ({train_acc:.2f}%) >> Val Acc ({val_acc:.2f}%) by {overfit_gap:.2f}%. Stopping.")
            break

        if no_improve_count >= Config.EARLY_STOP_PATIENCE:
            print(f"\n[Early Stop] Val accuracy did not improve for {Config.EARLY_STOP_PATIENCE} consecutive epochs (best: {best_val_acc:.2f}%). Stopping.")
            break
        # ─────────────────────────────────────────────────────────────────────

    # Evaluate Best Checkpoint on Test Set
    print("\n" + "=" * 65)
    print("EVALUATING BEST MODEL CHECKPOINT ON TEST SET (WITH TTA)")
    print("=" * 65)
    
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    
    test_loss, test_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device, use_tta=True)
    
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=['Non-Violent', 'Violent'], digits=4))

if __name__ == "__main__":
    main()
