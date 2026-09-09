# ==============================================================================
# 📊 BENCHMARK RESULTS EXCEL GENERATOR (results.xlsx)
# ==============================================================================

import os
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(PROJECT_ROOT, "results.xlsx")

wb = openpyxl.Workbook()
wb.remove(wb.active)  # Remove default sheet

# Colors & Styling
HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")  # Navy Blue
HEADER_FONT = Font(name="Arial", size=11, bold=True, color="FFFFFF")

CHAMPION_FILL = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")  # Light Green
CHAMPION_FONT = Font(name="Arial", size=10, bold=True, color="276A3C")

DATA_FONT = Font(name="Arial", size=10)
TITLE_FONT = Font(name="Arial", size=14, bold=True, color="1F4E79")
SUBTITLE_FONT = Font(name="Arial", size=10, italic=True, color="595959")

THIN_BORDER = Border(
    left=Side(style='thin', color='D9D9D9'),
    right=Side(style='thin', color='D9D9D9'),
    top=Side(style='thin', color='D9D9D9'),
    bottom=Side(style='thin', color='D9D9D9')
)

def style_sheet(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

# ==============================================================================
# SHEET 1: PHASE 1 BINARY CRIME GATE
# ==============================================================================
ws1 = wb.create_sheet(title="Phase 1 - Binary Gate")

ws1.append(["🚨 PHASE 1: BINARY CRIME DETECTION GATE (VIOLENCE vs NON-VIOLENCE)"])
ws1.append(["Dataset: Real Life Violence Situations (2,000 CCTV Videos: 1,000 Violent / 1,000 Normal)"])
ws1.append(["Evaluation: 200 Held-Out Isolated CCTV Videos with Test-Time Augmentation (TTA)"])
ws1.append([])

headers1 = [
    "Model Architecture", "Parameters", "Model Size", "Val Accuracy",
    "Test Accuracy", "Crime Recall", "Crime Precision", "F1-Score", "Status"
]
ws1.append(headers1)

phase1_data = [
    ["1. Baseline 3D CNN", "~1.2M", "4.8 MB", "88.00%", "90.00%", "90.00%", "90.00%", "90.00%", "Baseline Model"],
    ["2. 3D Res-SE CNN", "~3.6M", "14.5 MB", "89.00%", "91.50%", "90.00%", "92.78%", "91.37%", "Channel Attention"],
    ["3. VideoViT v1 (Transformer)", "~6.7M", "26.8 MB", "96.00%", "94.50%", "95.00%", "94.06%", "94.53%", "Transformer Encoder"],
    ["4. Optimized VideoViT v2 (TTA)", "~9.8M", "38.4 MB", "95.50%", "95.50%", "96.00%", "95.05%", "95.52%", "🏆 CHAMPION MODEL"]
]

for row in phase1_data:
    ws1.append(row)

ws1.append([])
ws1.append(["📊 Detailed Test Classification Report: Optimized VideoViT v2 (200 Held-Out CCTV Videos)"])
report_headers1 = ["Class Name", "Precision", "Recall", "F1-Score", "Support (Test Videos)"]
ws1.append(report_headers1)

report_data1 = [
    ["Non-Violent (Normal)", "95.96%", "95.00%", "95.48%", "100"],
    ["Violent (Crime Event)", "95.05%", "96.00%", "95.52%", "100"],
    ["Overall Accuracy / Macro Avg", "95.50%", "95.50%", "95.50%", "200"]
]
for row in report_data1:
    ws1.append(row)

# Format Sheet 1
ws1["A1"].font = TITLE_FONT
ws1["A2"].font = SUBTITLE_FONT
ws1["A3"].font = SUBTITLE_FONT

for col_num in range(1, len(headers1) + 1):
    cell = ws1.cell(row=5, column=col_num)
    cell.fill = HEADER_FILL
    cell.font = HEADER_FONT
    cell.alignment = Alignment(horizontal="center", vertical="center")

for row_idx in range(6, 6 + len(phase1_data)):
    is_champ = (row_idx == 9)
    for col_idx in range(1, len(headers1) + 1):
        cell = ws1.cell(row=row_idx, column=col_idx)
        cell.font = CHAMPION_FONT if is_champ else DATA_FONT
        if is_champ: cell.fill = CHAMPION_FILL
        cell.border = THIN_BORDER
        cell.alignment = Alignment(horizontal="center" if col_idx > 1 else "left", vertical="center")

ws1.cell(row=11, column=1).font = Font(name="Arial", size=11, bold=True, color="1F4E79")
for col_num in range(1, len(report_headers1) + 1):
    cell = ws1.cell(row=12, column=col_num)
    cell.fill = HEADER_FILL
    cell.font = HEADER_FONT
    cell.alignment = Alignment(horizontal="center", vertical="center")

for row_idx in range(13, 13 + len(report_data1)):
    for col_idx in range(1, len(report_headers1) + 1):
        cell = ws1.cell(row=row_idx, column=col_idx)
        cell.font = DATA_FONT
        cell.border = THIN_BORDER
        cell.alignment = Alignment(horizontal="center" if col_idx > 1 else "left", vertical="center")

style_sheet(ws1)

# ==============================================================================
# SHEET 2: PHASE 2 MULTI-CLASS ACTION CLASSIFICATION
# ==============================================================================
ws2 = wb.create_sheet(title="Phase 2 - Multi-Class Action")

ws2.append(["🥊 PHASE 2: MULTI-CLASS FINE-GRAINED CRIME ACTION RECOGNITION"])
ws2.append(["Benchmark Comparison across UCF-Crime and Large-Scale XD-Violence Datasets"])
ws2.append([])

headers2 = [
    "Experiment / Model Setup", "Dataset", "Classes", "Train Acc",
    "Val Acc", "Test Accuracy", "Weighted F1", "Key Finding / Status"
]
ws2.append(headers2)

phase2_data = [
    ["1. SlowFast Net (14 Classes)", "UCF-Crime", "14 Classes", "54.00%", "51.58%", "49.47%", "--", "Biased by 19:1 NormalVideos majority class"],
    ["2. Naive VideoViT (13 Classes)", "UCF-Crime", "13 Classes", "97.75%", "33.16%", "28.91%", "28.10%", "Overfit to background wallpapers (unfrozen backbone)"],
    ["3. Regularized VideoViT (13 Classes)", "UCF-Crime", "13 Classes", "96.38%", "29.12%", "22.61%", "22.05%", "Confused by identical motions (Fight vs Assault)"],
    ["4. Hierarchical Action Clusters", "UCF-Crime", "4 Clusters", "98.62%", "56.05%", "52.37%", "51.20%", "Improved with 4 kinematic meta-clusters"],
    ["5. XD-Violence Single-Slice ViT", "XD-Violence", "6 Classes", "72.99%", "66.82%", "71.23%", "71.41%", "Zero visual ambiguity migration"],
    ["6. Refined TSN VideoViT (USM Sharpness) 🏆", "XD-Violence", "6 Classes", "94.46%", "79.15%", "77.36%", "77.61%", "🏆 CHAMPION MODEL (TSN Sampling + Focal Loss)"]
]

for row in phase2_data:
    ws2.append(row)

ws2.append([])
ws2.append(["📊 Detailed Test Classification Report: Refined TSN VideoViT (318 Held-Out Crime Videos)"])
report_headers2 = ["Crime Action Category", "Precision", "Recall", "F1-Score", "Support (Test Videos)", "Physical Characteristics"]
ws2.append(report_headers2)

report_data2 = [
    ["Car Accident (Collision)", "90.41%", "94.29%", "92.31%", "70", "High-speed vehicular crashes & impact dynamics"],
    ["Riot / Mob Violence", "90.14%", "90.14%", "90.14%", "71", "Mass crowd disorder & multi-person street unrest"],
    ["Explosion / Blast Hazard", "82.00%", "75.93%", "78.85%", "54", "Detonations, fireballs & sudden blast expansion"],
    ["Fighting (Physical Brawl)", "71.88%", "63.01%", "67.15%", "73", "Hand-to-hand combat, wrestling & body strikes"],
    ["Shooting (Active Gunfire)", "51.92%", "60.00%", "55.67%", "45", "Disambiguated to 98% with YOLOv8 weapon fusion"],
    ["Abuse (One-on-One Assault)", "25.00%", "40.00%", "30.77%", "5", "Low test support; overlaps physically with fighting"],
    ["Overall Weighted Benchmark", "78.19%", "77.36%", "77.61%", "318", "Final Refined XD-Violence Champion Accuracy"]
]

for row in report_data2:
    ws2.append(row)

# Format Sheet 2
ws2["A1"].font = TITLE_FONT
ws2["A2"].font = SUBTITLE_FONT

for col_num in range(1, len(headers2) + 1):
    cell = ws2.cell(row=4, column=col_num)
    cell.fill = HEADER_FILL
    cell.font = HEADER_FONT
    cell.alignment = Alignment(horizontal="center", vertical="center")

for row_idx in range(5, 5 + len(phase2_data)):
    is_champ = (row_idx == 9)
    for col_idx in range(1, len(headers2) + 1):
        cell = ws2.cell(row=row_idx, column=col_idx)
        cell.font = CHAMPION_FONT if is_champ else DATA_FONT
        if is_champ: cell.fill = CHAMPION_FILL
        cell.border = THIN_BORDER
        cell.alignment = Alignment(horizontal="center" if col_idx in [3,4,5,6,7] else "left", vertical="center")

ws2.cell(row=11, column=1).font = Font(name="Arial", size=11, bold=True, color="1F4E79")
for col_num in range(1, len(report_headers2) + 1):
    cell = ws2.cell(row=12, column=col_num)
    cell.fill = HEADER_FILL
    cell.font = HEADER_FONT
    cell.alignment = Alignment(horizontal="center", vertical="center")

for row_idx in range(13, 13 + len(report_data2)):
    for col_idx in range(1, len(report_headers2) + 1):
        cell = ws2.cell(row=row_idx, column=col_idx)
        cell.font = DATA_FONT
        cell.border = THIN_BORDER
        cell.alignment = Alignment(horizontal="center" if col_idx in [2,3,4,5] else "left", vertical="center")

style_sheet(ws2)

# ==============================================================================
# SHEET 3: END-TO-END PIPELINE & MULTI-MODAL FUSION
# ==============================================================================
ws3 = wb.create_sheet(title="System Architecture & Fusion")

ws3.append(["🏛️ END-TO-END MULTI-STAGE CRIME SURVEILLANCE & EVIDENCE PIPELINE"])
ws3.append(["Hierarchical Real-Time Gating, Action Recognition, Weapon Detection, and Decision Fusion"])
ws3.append([])

headers3 = ["Pipeline Stage", "Module / Model", "Core Technology", "Input Stream", "Output / Role", "Performance Metric"]
ws3.append(headers3)

arch_data = [
    ["Stage 1: Binary Gate", "VideoViT v2", "MobileNetV3 + 1D Conv + Transformer", "24-Frame Video Clips (128x128)", "Filters Normal vs Crime Threat", "95.50% Test Acc / 96.0% Recall"],
    ["Stage 2: Action Classifier", "XD VideoViT", "MobileNetV3 + 8-Head Attention ViT", "24-Frame Video Clips (160x160)", "Categorizes Action (6 Classes)", "71.23% Test Acc / 74.3% Precision"],
    ["Stage 3A: Weapon Detector", "YOLOv8", "Deep CNN Bounding Box Detector", "High-Resolution Keyframe (1080p)", "Pinpoints Guns, Knives, Bats", ">90.0% Detection Precision"],
    ["Stage 3B: Suspect Face ID", "InsightFace", "RetinaFace Detector + ArcFace Embeddings", "Keyframe Suspect Crops", "Extracts Face Evidence for Watchlist", "512-D Identity Embedding"],
    ["Stage 4: Decision Fusion", "Fusion Engine", "Multi-Modal Rule & Softmax Boosting", "Video Actions + Object Bounding Boxes", "Disambiguates Weapons vs Brawls", "95%+ Incident Categorization"],
    ["Stage 5: Police Dossier", "Dossier Generator", "Automated PDF/JSON Alert Dispatcher", "Timestamped Alerts + Crops + Threat", "Official Police Evidence Dossier", "Automated Real-Time Alerts"]
]

for row in arch_data:
    ws3.append(row)

# Format Sheet 3
ws3["A1"].font = TITLE_FONT
ws3["A2"].font = SUBTITLE_FONT

for col_num in range(1, len(headers3) + 1):
    cell = ws3.cell(row=4, column=col_num)
    cell.fill = HEADER_FILL
    cell.font = HEADER_FONT
    cell.alignment = Alignment(horizontal="center", vertical="center")

for row_idx in range(5, 5 + len(arch_data)):
    for col_idx in range(1, len(headers3) + 1):
        cell = ws3.cell(row=row_idx, column=col_idx)
        cell.font = DATA_FONT
        cell.border = THIN_BORDER
        cell.alignment = Alignment(horizontal="center" if col_idx in [1, 6] else "left", vertical="center")

style_sheet(ws3)

# Save Workbook
wb.save(EXCEL_PATH)
print(f"✅ Benchmark Excel Spreadsheet successfully generated at: {EXCEL_PATH}")
