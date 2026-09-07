# ==============================================================================
# 🚨 PHASE 3: REAL-TIME WEAPON & DANGEROUS OBJECT DETECTOR (YOLOv8)
# ==============================================================================

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "Phase" in os.path.dirname(os.path.abspath(__file__)) else os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Optimisedmodel")
os.makedirs(MODELS_DIR, exist_ok=True)

class WeaponConfig:
    # Model Weights (Default: YOLOv8n / YOLOv8s)
    MODEL_NAME = "yolov8n.pt"
    MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

    # Minimum confidence threshold for threat objects
    CONFIDENCE_THRESHOLD = 0.30
    IOU_THRESHOLD = 0.45

    # Target Threat & Dangerous Object Categories (COCO Class Mappings & Extensions)
    # COCO: 43: knife, 76: scissors, 34: baseball bat
    THREAT_CLASS_MAP = {
        "knife": "Knife / Edged Weapon",
        "scissors": "Sharp / Edged Object",
        "baseball bat": "Blunt Weapon / Bat",
        "gun": "Firearm / Handgun",
        "pistol": "Firearm / Pistol",
        "rifle": "Firearm / Rifle",
        "weapon": "Lethal Weapon"
    }

# ==============================================================================
# WEAPON DETECTOR CLASS
# ==============================================================================
class WeaponDetector:
    def __init__(self, model_path=WeaponConfig.MODEL_PATH, conf_thresh=WeaponConfig.CONFIDENCE_THRESHOLD):
        self.conf_thresh = conf_thresh
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load YOLOv8 Model (downloads pretrained weights automatically if not cached)
        print(f"Loading YOLOv8 Weapon Detector on {self.device.upper()}...")
        self.model = YOLO(model_path if os.path.exists(model_path) else WeaponConfig.MODEL_NAME)
        
        # Cache weapon class IDs in model
        self.threat_names = list(WeaponConfig.THREAT_CLASS_MAP.keys())
        print(f"✅ YOLOv8 Weapon Detector Active. Monitoring threats: {', '.join(self.threat_names)}")

    def detect_frame(self, frame):
        """
        Detects dangerous objects and weapons in a single BGR image/frame.
        Returns:
            detections (list of dicts): Each dict has 'label', 'conf', 'bbox', 'is_weapon'
        """
        if frame is None:
            return []

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=WeaponConfig.IOU_THRESHOLD,
            device=self.device,
            verbose=False
        )

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id].lower()
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Check if detected object is in our threat dictionary or dangerous object list
                is_threat = False
                display_label = cls_name
                
                for threat_key, threat_val in WeaponConfig.THREAT_CLASS_MAP.items():
                    if threat_key in cls_name:
                        is_threat = True
                        display_label = threat_val
                        break

                detections.append({
                    "raw_class": cls_name,
                    "label": display_label,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "is_threat": is_threat
                })

        return detections

    def annotate_frame(self, frame, detections):
        """
        Draws glowing bounding boxes and alerts over detected objects.
        """
        annotated = frame.copy()
        threat_count = 0

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            label = det["label"]
            is_threat = det["is_threat"]

            if is_threat:
                threat_count += 1
                box_color = (0, 0, 255)       # Red for Lethal Weapons
                text_bg = (0, 0, 180)
            else:
                box_color = (0, 215, 255)     # Amber/Yellow for general objects
                text_bg = (0, 140, 180)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Draw label tag with background
            tag_text = f"{label} {conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), text_bg, -1)
            cv2.putText(annotated, tag_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Top-Left Security Badge
        if threat_count > 0:
            cv2.rectangle(annotated, (15, 15), (340, 55), (0, 0, 180), -1)
            cv2.putText(annotated, f"🚨 WEAPON DETECTED ({threat_count})", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

# ==============================================================================
# DEMO & TESTING
# ==============================================================================
def main():
    print("=" * 65)
    print("🚀 INITIALIZING PHASE 3: YOLOV8 WEAPON DETECTOR")
    print("=" * 65)

    detector = WeaponDetector()

    # Create synthetic test canvas with test shapes
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(canvas, "Phase 3 Weapon Detector Ready", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    detections = detector.detect_frame(canvas)
    print(f"Test Run Completed. Detections found on blank canvas: {len(detections)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
