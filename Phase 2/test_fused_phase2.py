# ==============================================================================
# 🚨 PHASE 2: FUSED VIDEO-TRANSFORMER + YOLOV8 MULTI-MODAL EVALUATOR
# ==============================================================================

import os
import sys
import glob
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score

# Paths & Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "Phase" in os.path.dirname(os.path.abspath(__file__)) else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Phase 2"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Phase 3"))

from XD_Model import XDCrimeClassifierViT, Config as XDConfig, XD_CRIME_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, compute_motion
from weapon_detector import WeaponDetector

# Device configuration
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Fused Phase 2 Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Fused Phase 2 Device: CPU")
except Exception:
    device = torch.device("cpu")
    print("Fused Phase 2 Device: CPU")

# ==============================================================================
# FUSED PHASE 2 CLASSIFIER (VideoViT + YOLOv8 Multi-Modal Fusion)
# ==============================================================================
class FusedPhase2Classifier:
    """
    Multi-Modal Crime Classifier combining:
    1. Spatio-Temporal Video Vision Transformer (VideoViT) for kinematic action modeling.
    2. YOLOv8 High-Resolution Keyframe Weapon Detector for dangerous object verification.
    """
    def __init__(self, video_model_path=XDConfig.BEST_MODEL_PATH):
        print("\n" + "=" * 65)
        print("🛡️ INITIALIZING FUSED PHASE 2 MULTI-MODAL CLASSIFIER")
        print("=" * 65)

        # 1. Load Video Vision Transformer
        self.video_model = XDCrimeClassifierViT(num_classes=len(XD_CRIME_CLASSES)).to(device)
        if os.path.exists(video_model_path):
            self.video_model.load_state_dict(torch.load(video_model_path, map_location=device))
            self.video_model.eval()
            print(f"✅ VideoViT Model Loaded: {video_model_path}")
        else:
            print(f"⚠️ Warning: VideoViT weights not found at {video_model_path}")

        # 2. Load YOLOv8 Weapon Detector
        self.weapon_detector = WeaponDetector()
        print("=" * 65 + "\n")

    def classify_clip(self, raw_frames, use_fusion=True):
        """
        Takes raw BGR frames (list of 24 frames @ original resolution) and performs fused classification.
        Returns:
            predicted_class (str): Final crime category
            confidence (float): Confidence score (0.0 to 1.0)
            breakdown (dict): Detailed class probability scores
            weapons_detected (list): Bounding box detections from YOLOv8
        """
        if len(raw_frames) < XDConfig.MAX_FRAMES:
            while len(raw_frames) < XDConfig.MAX_FRAMES:
                raw_frames.append(raw_frames[-1] if len(raw_frames) > 0 else np.zeros((160, 160, 3), dtype=np.uint8))

        # --- 1. VideoViT Branch ---
        resized_frames = [cv2.resize(f, XDConfig.TARGET_SIZE).astype("float32") / 255.0 for f in raw_frames[:XDConfig.MAX_FRAMES]]
        blended = compute_motion(np.array(resized_frames))
        tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.video_model(tensor)
            video_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # --- 2. YOLOv8 Keyframe Branch ---
        keyframe_idx = len(raw_frames) // 2
        keyframe = raw_frames[keyframe_idx]
        weapon_detections = self.weapon_detector.detect_frame(keyframe) if use_fusion else []

        fused_probs = video_probs.copy()

        # --- 3. Decision-Level Fusion Boosting ---
        if use_fusion and weapon_detections:
            for det in weapon_detections:
                if det["is_threat"]:
                    conf = det["conf"]
                    label = det["raw_class"]

                    # Boost Shooting probability if firearms are detected
                    if any(w in label for w in ["gun", "pistol", "rifle", "weapon"]):
                        shoot_idx = CLASS_TO_IDX.get("Shooting", 1)
                        fused_probs[shoot_idx] += 1.5 * conf

                    # Boost Fighting / Assault probability if edged weapons are detected
                    elif any(w in label for w in ["knife", "scissors", "blade"]):
                        fight_idx = CLASS_TO_IDX.get("Fighting", 0)
                        fused_probs[fight_idx] += 1.2 * conf

                    # Boost Fighting probability if baseball bat is detected
                    elif "bat" in label:
                        fight_idx = CLASS_TO_IDX.get("Fighting", 0)
                        fused_probs[fight_idx] += 1.2 * conf

            # Re-normalize probabilities
            fused_probs = fused_probs / np.sum(fused_probs)

        top_idx = int(np.argmax(fused_probs))
        predicted_class = IDX_TO_CLASS[top_idx]
        confidence = float(fused_probs[top_idx])

        breakdown = {IDX_TO_CLASS[i]: float(fused_probs[i]) for i in range(len(XD_CRIME_CLASSES))}
        return predicted_class, confidence, breakdown, weapon_detections

# ==============================================================================
# STANDALONE DEMO & EVALUATION
# ==============================================================================
def evaluate_video(classifier, video_path):
    print(f"\nProcessing Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        if len(frames) == XDConfig.MAX_FRAMES:
            break
    cap.release()

    if len(frames) == 0:
        print("  ❌ Could not read frames from video.")
        return

    # Evaluate without YOLO
    pure_class, pure_conf, _, _ = classifier.classify_clip(frames, use_fusion=False)
    # Evaluate WITH YOLO Fusion
    fused_class, fused_conf, breakdown, weapons = classifier.classify_clip(frames, use_fusion=True)

    print(f"  • Pure VideoViT Guess : {pure_class} ({pure_conf*100:.1f}%)")
    w_names = [w['label'] for w in weapons if w['is_threat']]
    print(f"  • YOLOv8 Detections   : {', '.join(w_names) if w_names else 'No Weapons'}")
    print(f"  • 🎯 FUSED DECISION   : {fused_class} ({fused_conf*100:.1f}%)")

def main():
    classifier = FusedPhase2Classifier()
    
    if len(sys.argv) > 1:
        evaluate_video(classifier, sys.argv[1])
    else:
        sample_vids = glob.glob(os.path.join(PROJECT_ROOT, "Dataset", "**", "*.mp4"), recursive=True)
        if sample_vids:
            for v in sample_vids[:2]:
                evaluate_video(classifier, v)
        else:
            print("Usage: python 'Phase 2/test_fused_phase2.py' <path_to_video.mp4>")

if __name__ == "__main__":
    main()
