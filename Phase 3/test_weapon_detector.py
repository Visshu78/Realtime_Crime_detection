# ==============================================================================
# 🚨 PHASE 3: STANDALONE WEAPON & DANGEROUS OBJECT EVALUATOR
# ==============================================================================

import os
import sys
import cv2
import glob
from weapon_detector import WeaponDetector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "Phase" in os.path.dirname(os.path.abspath(__file__)) else os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Optimisedmodel", "weapon_test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_image(detector, image_path):
    print(f"\nEvaluating Image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("  ❌ Could not load image.")
        return

    detections = detector.detect_frame(frame)
    annotated = detector.annotate_frame(frame, detections)
    
    out_name = os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_DIR, f"annotated_{out_name}")
    cv2.imwrite(save_path, annotated)

    print(f"  --> Found {len(detections)} objects.")
    for d in detections:
        print(f"      • {d['label']} (Confidence: {d['conf']*100:.1f}%) [Threat: {d['is_threat']}]")
    print(f"  --> Saved annotated visual evidence to: {save_path}")

def test_video(detector, video_path):
    print(f"\nEvaluating Video Stream: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    out_name = os.path.basename(video_path)
    save_path = os.path.join(OUTPUT_DIR, f"annotated_{out_name}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_idx = 0
    threat_frames = 0
    detected_threats = set()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        detections = detector.detect_frame(frame)
        annotated = detector.annotate_frame(frame, detections)
        out.write(annotated)

        for d in detections:
            if d['is_threat']:
                threat_frames += 1
                detected_threats.add(d['label'])

        frame_idx += 1

    cap.release()
    out.release()

    print(f"  --> Processed {frame_idx}/{total_frames} frames.")
    print(f"  --> Dangerous Objects Detected: {', '.join(detected_threats) if detected_threats else 'None'}")
    print(f"  --> Threat Frames: {threat_frames}")
    print(f"  --> Saved annotated incident video to: {save_path}")

def main():
    print("=" * 65)
    print("🛡️ PHASE 3: YOLOV8 WEAPON DETECTION EVALUATOR")
    print("=" * 65)

    detector = WeaponDetector()

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        if target_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            test_video(detector, target_path)
        else:
            test_image(detector, target_path)
    else:
        # Search for any sample video in Dataset
        sample_vids = glob.glob(os.path.join(PROJECT_ROOT, "Dataset", "**", "*.mp4"), recursive=True)
        if sample_vids:
            print(f"Found sample video for testing: {sample_vids[0]}")
            test_video(detector, sample_vids[0])
        else:
            print("Usage: python 'Phase 3/test_weapon_detector.py' <path_to_image_or_video>")

if __name__ == "__main__":
    main()
