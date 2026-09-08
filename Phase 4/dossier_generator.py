# ==============================================================================
# 🚨 PHASE 4: AUTOMATED POLICE INCIDENT DOSSIER & EVIDENCE GENERATOR
# ==============================================================================

import os
import json
import time
import cv2
import numpy as np
from datetime import datetime

class IncidentDossierGenerator:
    """
    Automated Incident Dossier & Police Evidence Generator.
    Compiles multi-stage detections into structured forensic reports and saves image evidence.
    """
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "incident_evidence"
        )
        self.reports_dir = os.path.join(self.output_dir, "reports")
        self.snapshots_dir = os.path.join(self.output_dir, "snapshots")
        self.suspects_dir = os.path.join(self.output_dir, "suspect_crops")
        self.weapons_dir = os.path.join(self.output_dir, "weapon_crops")

        for d in [self.reports_dir, self.snapshots_dir, self.suspects_dir, self.weapons_dir]:
            os.makedirs(d, exist_ok=True)

    def determine_severity(self, crime_type, weapons_detected):
        """
        Determines the police alert priority level based on crime nature and weapon presence.
        """
        has_firearm = any(w['label'] in ['gun', 'pistol', 'rifle'] for w in weapons_detected)
        has_blade = any(w['label'] in ['knife', 'blade', 'dagger'] for w in weapons_detected)

        if has_firearm or crime_type in ['Shooting', 'Armed Shooting', 'Explosion']:
            return "CODE RED - CRITICAL EMERGENCY (Armed Threat / Immediate SWAT/Armed Response)"
        elif has_blade or crime_type in ['Knife Assault', 'Riot']:
            return "CODE ORANGE - HIGH THREAT (Assault with Deadly Weapon / Rapid Patrol Response)"
        elif crime_type in ['Fighting', 'Unarmed Brawl', 'CarAccident', 'Abuse']:
            return "CODE YELLOW - ELEVATED ALERT (Physical Violence / Traffic Units Dispatched)"
        else:
            return "CODE BLUE - ROUTINE INVESTIGATION"

    def generate_dossier(self, video_source, crime_type, threat_prob, weapons_detected, suspect_faces, keyframe_bgr=None, camera_id="CCTV_CAM_04"):
        """
        Compiles the complete multi-modal incident dossier, saves image evidence, and writes forensic logs.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        incident_id = f"INCIDENT_{timestamp_str}_{camera_id}"
        severity = self.determine_severity(crime_type, weapons_detected)

        # 1. Save Full-Frame Snapshot
        snapshot_path = ""
        if keyframe_bgr is not None:
            snapshot_filename = f"{incident_id}_keyframe.jpg"
            snapshot_path = os.path.join(self.snapshots_dir, snapshot_filename)
            
            # Annotate keyframe with bounding boxes
            annotated = keyframe_bgr.copy()
            for w in weapons_detected:
                b = w['bbox']
                cv2.rectangle(annotated, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cv2.putText(annotated, f"WEAPON: {w['label'].upper()} ({w['conf']*100:.1f}%)", (b[0], max(15, b[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            for s in suspect_faces:
                b = s['bbox']
                cv2.rectangle(annotated, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.putText(annotated, f"SUSPECT: {s['identity']}", (b[0], max(15, b[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            cv2.imwrite(snapshot_path, annotated)

        # 2. Save Cropped Evidence
        saved_suspects = []
        for idx, s in enumerate(suspect_faces):
            if 'face_crop' in s and s['face_crop'] is not None and s['face_crop'].size > 0:
                crop_name = f"{incident_id}_suspect_{idx}_{s['identity'].replace(' ', '_')}.jpg"
                crop_path = os.path.join(self.suspects_dir, crop_name)
                cv2.imwrite(crop_path, s['face_crop'])
                saved_suspects.append({
                    'identity': s['identity'],
                    'match_score': s.get('match_score', 0.0),
                    'bbox': s['bbox'],
                    'crop_file': crop_path
                })
            else:
                saved_suspects.append({
                    'identity': s['identity'],
                    'match_score': s.get('match_score', 0.0),
                    'bbox': s['bbox']
                })

        saved_weapons = []
        for idx, w in enumerate(weapons_detected):
            saved_weapons.append({
                'type': w['label'],
                'confidence': float(w['conf']),
                'bbox': w['bbox']
            })

        # 3. Create Structured Forensic Record
        dossier_data = {
            'incident_id': incident_id,
            'timestamp': datetime.now().isoformat(),
            'camera_id': camera_id,
            'video_source': video_source,
            'severity_rating': severity,
            'phase_1_threat_score': float(threat_prob),
            'phase_2_crime_classification': crime_type,
            'phase_3a_weapons_count': len(saved_weapons),
            'phase_3a_weapons_evidence': saved_weapons,
            'phase_3b_suspects_count': len(saved_suspects),
            'phase_3b_suspects_identified': saved_suspects,
            'keyframe_snapshot': snapshot_path,
            'dispatch_action_recommended': "Immediate Patrol Dispatch & Medical/SWAT Alert" if "CRITICAL" in severity or "HIGH" in severity else "Dispatch Patrol Unit"
        }

        # 4. Save JSON Dossier
        json_path = os.path.join(self.reports_dir, f"{incident_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dossier_data, f, indent=4)

        # 5. Generate Human-Readable Police Summary Card
        summary_path = os.path.join(self.reports_dir, f"{incident_id}_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("🚨 OFFICIAL POLICE INCIDENT EVIDENCE DOSSIER\n")
            f.write("=" * 70 + "\n")
            f.write(f"Incident ID   : {incident_id}\n")
            f.write(f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Camera ID     : {camera_id}\n")
            f.write(f"Source Feed   : {video_source}\n")
            f.write(f"Priority Level: {severity}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Phase 1 Gate  : VIOLENCE CONFIRMED (Threat Confidence: {threat_prob*100:.1f}%)\n")
            f.write(f"Phase 2 Action: {crime_type.upper()}\n")
            f.write(f"Weapons Found : {len(saved_weapons)} detected\n")
            for w in saved_weapons:
                f.write(f"  - [{w['type'].upper()}] Confidence: {w['confidence']*100:.1f}% | BBox: {w['bbox']}\n")
            f.write(f"Suspects Found: {len(saved_suspects)} detected\n")
            for s in saved_suspects:
                f.write(f"  - [SUSPECT ID: {s['identity']}] Match Score: {s['match_score']*100:.1f}% | BBox: {s['bbox']}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Keyframe File : {snapshot_path}\n")
            f.write(f"JSON Record   : {json_path}\n")
            f.write("=" * 70 + "\n")

        print(f"✅ Police Incident Dossier Generated: {json_path}")
        return dossier_data


if __name__ == "__main__":
    generator = IncidentDossierGenerator()
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dossier = generator.generate_dossier(
        video_source="test_cctv_feed.mp4",
        crime_type="Armed Shooting",
        threat_prob=0.985,
        weapons_detected=[{'label': 'gun', 'conf': 0.94, 'bbox': [120, 80, 240, 200]}],
        suspect_faces=[{'identity': 'Unknown Suspect #1', 'match_score': 0.0, 'bbox': [50, 40, 110, 110], 'face_crop': dummy_frame[40:110, 50:110]}],
        keyframe_bgr=dummy_frame
    )
    print("Dossier Generator Test Passed successfully.")
