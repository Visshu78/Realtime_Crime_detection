# ==============================================================================
# 🚨 END-TO-END CRIME DETECTION & CLASSIFICATION SURVEILLANCE PIPELINE
# ==============================================================================

import os
import sys
import time
import cv2
import torch
import numpy as np
from collections import deque

# Add Phase 1 and Phase 2 directories to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Phase 1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Phase 2"))

# Import Stage 1 and Stage 2 architectures
from Model import VideoViT as Stage1Gate, compute_motion, Config as Stage1Config
from MultiClassModel import SlowFastNet as Stage2Classifier, Config as Stage2Config, CRIME_CLASSES, IDX_TO_CLASS

# Device Setup
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Pipeline Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Pipeline Device: CPU")
except Exception:
    device = torch.device("cpu")
    print("Pipeline Device: CPU")

# Load Stage 1 Gate Model
stage1_model = Stage1Gate().to(device)
if os.path.exists(Stage1Config.BEST_MODEL_PATH):
    stage1_model.load_state_dict(torch.load(Stage1Config.BEST_MODEL_PATH, map_location=device))
    stage1_model.eval()
    print(f"✅ Stage 1 Gate Loaded: {Stage1Config.BEST_MODEL_PATH} (95.5% Test Accuracy)")
else:
    print(f"Warning: Stage 1 weights not found at '{Stage1Config.BEST_MODEL_PATH}'")

# Load Stage 2 SlowFast Classifier Model
stage2_model = Stage2Classifier(num_classes=len(CRIME_CLASSES)).to(device)
if os.path.exists(Stage2Config.BEST_MODEL_PATH):
    stage2_model.load_state_dict(torch.load(Stage2Config.BEST_MODEL_PATH, map_location=device))
    stage2_model.eval()
    print(f"✅ Stage 2 SlowFast Net Loaded: {Stage2Config.BEST_MODEL_PATH}")
else:
    print(f"ℹ️ Stage 2 SlowFast weights will be loaded once trained ({Stage2Config.BEST_MODEL_PATH})")

def run_pipeline(video_path):
    """
    Executes the complete hierarchical surveillance pipeline on a video file or stream:
    1. Stage 1 Binary Gate (VideoViT) continuously monitors 24-frame clips.
    2. When crime is detected (prob > 0.50), Stage 2 (SlowFast Net) classifies the exact crime type.
    3. Outputs a structured, timestamped Incident Dossier.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n" + "=" * 65)
    print("      INTELLIGENT SURVEILLANCE PIPELINE: ACTIVE MONITORING")
    print("=" * 65)
    print(f"Source Feed  : {video_path}")
    print(f"Frame Rate   : {fps:.2f} FPS")
    print(f"Total Frames : {total_frames}")
    print("=" * 65 + "\n")

    frames_buffer = []
    frame_count = 0
    incident_logs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        resized = cv2.resize(frame, Stage1Config.TARGET_SIZE).astype("float32") / 255.0
        frames_buffer.append(resized)

        # When a 24-frame window is ready
        if len(frames_buffer) == Stage1Config.MAX_FRAMES:
            clip = np.array(frames_buffer)
            blended = compute_motion(clip)
            tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2).unsqueeze(0).to(device)

            # --- STAGE 1: BINARY CRIME GATE ---
            with torch.no_grad():
                # TTA Prediction for maximum reliability
                out1 = stage1_model(tensor)
                flipped = torch.flip(tensor, dims=[4])
                out2 = stage1_model(flipped)
                violence_prob = ((torch.sigmoid(out1) + torch.sigmoid(out2)) / 2.0).item()

            current_timestamp = (frame_count - Stage1Config.MAX_FRAMES) / fps
            timestamp_str = time.strftime('%H:%M:%S', time.gmtime(current_timestamp))

            if violence_prob >= 0.50:
                # --- STAGE 2: SLOWFAST FINE-GRAINED ACTION CLASSIFICATION ---
                with torch.no_grad():
                    s2_out1 = stage2_model(tensor)
                    s2_out2 = stage2_model(flipped)
                    action_probs = ((torch.softmax(s2_out1, dim=1) + torch.softmax(s2_out2, dim=1)) / 2.0).cpu().numpy()[0]

                top_action_idx = np.argmax(action_probs)
                detected_crime = IDX_TO_CLASS[top_action_idx]
                action_conf = action_probs[top_action_idx]

                alert_msg = (
                    f"[{timestamp_str}] 🚨 CRIME DETECTED! "
                    f"Threat Score: {violence_prob*100:5.1f}% | "
                    f"Action: {detected_crime} ({action_conf*100:5.1f}%)"
                )
                print(alert_msg)
                incident_logs.append({
                    "timestamp": timestamp_str,
                    "violence_score": violence_prob,
                    "action_type": detected_crime,
                    "action_conf": action_conf
                })
            else:
                print(f"[{timestamp_str}] ✅ Normal Activity (Threat: {violence_prob*100:4.1f}%)")

            # Sliding window stride: advance by 12 frames (50% overlap)
            frames_buffer = frames_buffer[12:]

    cap.release()

    # --- INCIDENT DOSSIER SUMMARY ---
    print("\n" + "=" * 65)
    print("              FINAL INCIDENT EVIDENCE DOSSIER")
    print("=" * 65)
    if len(incident_logs) == 0:
        print("Status: ✅ NO CRIMINAL ACTIVITY DETECTED THROUGHOUT FEED.")
    else:
        print(f"Status: 🚨 {len(incident_logs)} CRIME INCIDENT WINDOWS FLAGGED!")
        print("-" * 65)
        # Find highest threat incident
        worst_incident = max(incident_logs, key=lambda x: x["violence_score"])
        print(f"Primary Crime Type : {worst_incident['action_type']}")
        print(f"Peak Threat Level  : {worst_incident['violence_score']*100:.2f}%")
        print(f"First Detected At  : {incident_logs[0]['timestamp']}")
        print(f"Peak Incident Time : {worst_incident['timestamp']}")
    print("=" * 65)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        feed_path = sys.argv[1]
    else:
        feed_path = input("\nEnter video path or RTSP stream URL for pipeline: ").strip().strip('"')
    
    run_pipeline(feed_path)
