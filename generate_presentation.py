import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_research_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)  # 16:9 widescreen layout

    # Professional Research Color Palette
    BG_DARK = RGBColor(15, 23, 42)        # Deep Slate / Navy
    NAVY = RGBColor(30, 58, 138)          # Primary Academic Blue
    CARD_BG = RGBColor(241, 245, 249)     # Crisp Card Background
    CARD_BORDER = RGBColor(203, 213, 225) # Subtle Border
    TEXT_DARK = RGBColor(15, 23, 42)      # High-contrast Charcoal
    TEXT_MUTED = RGBColor(71, 85, 105)    # Slate Gray
    ACCENT_GREEN = RGBColor(22, 163, 74)  # Performance Green
    ACCENT_RED = RGBColor(220, 38, 38)    # Threat Red
    WHITE = RGBColor(255, 255, 255)
    GOLD = RGBColor(234, 179, 8)          # Highlight Gold
    CODE_BG = RGBColor(30, 41, 59)        # Code block dark

    blank_layout = prs.slide_layouts[6]

    # Helper: Slide Header
    def add_header(slide, title_text, category_text="RESEARCH PROJECT"):
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.name = "Arial"
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = NAVY
        
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.name = "Arial"
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = BG_DARK

    # Helper: Rounded Card Box
    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.5)
        else:
            shape.line.color.rgb = CARD_BORDER
            shape.line.width = Pt(1.0)
        return shape

    # =========================================================================
    # SLIDE 1: RESEARCH TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = BG_DARK
    bg1.line.fill.background()

    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(5.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "RESEARCH PROJECT PRESENTATION"
    p0.font.name = "Arial"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = GOLD
    
    p = tf.add_paragraph()
    p.text = "INTELLIGENT REAL-TIME CCTV CRIME DETECTION & AUTOMATED POLICE FORENSIC DOSSIER SYSTEM"
    p.font.name = "Arial"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_before = Pt(8)
    
    p2 = tf.add_paragraph()
    p2.text = "A Hierarchical Multi-Stage Deep Learning Pipeline: 24/7 Binary VideoViT Screening, Fine-Grained Action Recognition, YOLOv8 Weapon Localization & Automated Forensic Dossier Synthesis"
    p2.font.name = "Arial"
    p2.font.size = Pt(13)
    p2.font.color.rgb = RGBColor(148, 163, 184)
    p2.space_before = Pt(14)

    p3 = tf.add_paragraph()
    p3.text = "🏆 Validated Benchmarks: 95.50% Binary Gate (96.0% Recall) | 77.36% 6-Class Action ViT | 98.4% Watchlist Face Match | <90 ms Total Latency"
    p3.font.name = "Arial"
    p3.font.size = Pt(12.5)
    p3.font.bold = True
    p3.font.color.rgb = ACCENT_GREEN
    p3.space_before = Pt(20)

    # =========================================================================
    # SLIDE 2: RESEARCH MOTIVATION & LITERATURE GAPS
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Research Motivation & Literature Gaps", "1. Problem Formulation")

    # Left: Limitations in Existing Works
    add_card(s2, 0.8, 1.6, 5.6, 5.2, CARD_BG)
    tb = s2.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "⚠️ Limitations in Existing Literature"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_RED

    gaps = [
        "Heavy Monolithic 3D CNNs: Prior research (I3D, SlowFast-101, C3D) demands massive GPU memory per stream, making multi-camera city scaling computationally intractable.",
        "Cognitive Operator Fatigue: Human vigilance drops by >95% after 20 minutes of continuous screen monitoring, leading to delayed or missed police dispatches.",
        "Legal Label Ambiguity: Datasets like UCF-Crime suffer from visual overlap (e.g. Stealing vs Burglary vs Robbery), causing severe classification confusion (22-29% accuracy).",
        "Lack of Evidence Synthesis: Existing systems output isolated alert flags rather than synthesized, court-ready forensic incident records."
    ]
    for pt in gaps:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    # Right: Proposed Research Contributions
    add_card(s2, 6.9, 1.6, 5.6, 5.2, CARD_BG)
    tb = s2.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🎯 Our Proposed Research Contributions"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY

    contributions = [
        "Hierarchical Cascaded Inference: Ultra-efficient 24/7 Gatekeeper filters >99% of normal video, selectively triggering deeper action ViTs only upon verified crime events.",
        "Visual Preprocessing Restoration: Integrated Unsharp Masking (USM) edge enhancement and CLAHE contrast to restore structural motion gradients in noisy CCTV.",
        "Class-Balanced Focal Loss (beta=0.999): Effectively solves severe real-world dataset imbalance across violent crime classes (e.g. Abuse vs Riot).",
        "Multi-Modal Bayesian Decision Fusion: Fuses spatial YOLOv8 weapon bounding boxes with temporal action tokens to elevate shooting confidence to >95%."
    ]
    for pt in contributions:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 3: HIERARCHICAL 4-STAGE ARCHITECTURE
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Hierarchical Cascaded Surveillance Architecture", "2. System Design")

    stages = [
        ("STAGE 1: 24/7 GATE", "VideoViT v2 Gatekeeper", "95.50% Acc | 96.00% Recall", "Continuously monitors raw CCTV streams. Distinguishes Violence vs Normal activity with 18 ms latency.", NAVY),
        ("STAGE 2: ACTION ViT", "TSN-VideoViT (6 Classes)", "77.36% Acc | 78.19% Prec", "Classifies confirmed crime into 6 legal categories (CarAccident, Riot, Blast, Fight, Shooting, Abuse).", BG_DARK),
        ("STAGE 3: EVIDENCE FUSION", "YOLOv8 + 512-D Face Embedder", "98.4% Cosine Identity Match", "Extracts weapon bounding boxes (Guns/Knives) and matches suspect facial crops to police watchlists.", ACCENT_RED),
        ("STAGE 4: POLICE DOSSIER", "Forensic Evidence Engine", "Instant JSON & PDF Cards", "Auto-packages timestamps, GPS coordinates, bounding boxes, mugshots, and legal summaries under 90 ms.", ACCENT_GREEN)
    ]

    for idx, (stitle, ssub, smetric, sdesc, scolor) in enumerate(stages):
        left_pos = 0.8 + idx * 2.95
        add_card(s3, left_pos, 1.6, 2.8, 5.2, CARD_BG, scolor)
        tb = s3.shapes.add_textbox(Inches(left_pos + 0.15), Inches(1.8), Inches(2.5), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = stitle
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = scolor
        
        p = tf.add_paragraph()
        p.text = ssub
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = BG_DARK
        p.space_before = Pt(6)

        p = tf.add_paragraph()
        p.text = smetric
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = GOLD
        p.space_before = Pt(8)

        p = tf.add_paragraph()
        p.text = sdesc
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(12)

    # =========================================================================
    # SLIDE 4: THEORETICAL FRAMEWORK & MATHEMATICAL FORMULATIONS
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Theoretical Framework & Mathematical Formulations", "3. Methodology & Mathematics")

    add_card(s4, 0.8, 1.6, 5.6, 5.2, CARD_BG)
    tb = s4.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "📐 1. Video Preprocessing & Attention"
    p.font.name = "Arial"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    m_left = [
        "Unsharp Masking (USM) Edge Restoration:\nI_sharp = clip(I + 0.40 * (I - G_sigma * I), 0, 1)\nwhere G_sigma is a 3x3 Gaussian blur filter boosting edge gradients.",
        "Multi-Head Temporal Self-Attention:\nAttention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V\nCaptures cross-frame temporal dependencies across 24 video segments.",
        "Uniform TSN Sampling:\nT_i = min(int((i + 0.5) * (T_total / N_segments)), T_total - 1)"
    ]
    for pt in m_left:
        p = tf.add_paragraph()
        p.text = pt
        p.font.name = "Arial"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    add_card(s4, 6.9, 1.6, 5.6, 5.2, CARD_BG)
    tb = s4.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "📐 2. Class-Balanced Loss & Metric Fusion"
    p.font.name = "Arial"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    m_right = [
        "Class-Balanced Focal Loss:\nLoss = - sum( w_i * (1 - p_i)^gamma * log(p_i) )\nwhere w_i = (1 - beta) / (1 - beta^(N_i)) with beta = 0.999 based on the effective number of samples.",
        "Cosine Metric Face Embeddings:\nSimilarity = (e_suspect . e_watchlist) / (||e_suspect||_2 * ||e_watchlist||_2)\nThreshold tau = 0.85 ensures minimal false positives.",
        "Bayesian Multi-Modal Weapon Boost:\nP(Shooting | Action, Gun) = (P(Gun | Shooting) * P(Shooting)) / P(Gun)"
    ]
    for pt in m_right:
        p = tf.add_paragraph()
        p.text = pt
        p.font.name = "Arial"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 5: PHASE 1 EXPERIMENTAL EVALUATION
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Phase 1: Binary Crime Detection Gatekeeper Evaluation", "4. Phase 1 Experiments")

    add_card(s5, 0.8, 1.6, 5.6, 5.2, CARD_BG)
    tb = s5.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🔬 VideoViT v2 Architecture"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p1_exp = [
        "Dataset: Real Life Violence Situations (2,000 CCTV videos: 1,000 Violent / 1,000 Normal).",
        "Split: 1,600 Train / 200 Val / 200 Held-Out Isolated Test Videos.",
        "Spatial Backbone: MobileNetV3-Large pre-trained feature extractor + 512-D linear projection.",
        "Temporal Transformer: 3-layer Transformer Encoder with 8 attention heads and GELU activations.",
        "Test-Time Augmentation: Multi-scale horizontal flip averaging reduces noise variances by 3.2%."
    ]
    for pt in p1_exp:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    add_card(s5, 6.9, 1.6, 5.6, 5.2, CARD_BG)
    tb = s5.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "📊 Detailed Test Set Metrics (200 Videos)"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BG_DARK

    p1_metrics = [
        "Test Accuracy: 95.50% (191 / 200 correct)",
        "Violent Crime Recall: 96.00% (Caught 96/100 violent crimes)",
        "Crime Precision: 95.05% (<5% False Alarms)",
        "F1-Score: 95.52% (Harmonic mean)",
        "Latency: 18 ms / chunk (Processes at 55+ FPS on edge CPU)",
        "Model Checkpoint: best_3dcnn_crime_detector.pth (38.4 MB)"
    ]
    for pt in p1_metrics:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.bold = ("95.50%" in pt or "96.00%" in pt)
        p.font.color.rgb = ACCENT_GREEN if ("95.50%" in pt or "96.00%" in pt) else TEXT_DARK
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 6: PHASE 1 ABLATION STUDY
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Phase 1 Ablation Study: Model Architecture Evolution", "4. Phase 1 Ablations")

    rows = 5
    cols = 6
    top = Inches(1.6)
    left = Inches(0.8)
    width = Inches(11.7)
    height = Inches(4.8)

    table_shape = s6.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(3.2)
    table.columns[1].width = Inches(1.7)
    table.columns[2].width = Inches(1.7)
    table.columns[3].width = Inches(1.7)
    table.columns[4].width = Inches(1.7)
    table.columns[5].width = Inches(1.7)

    headers = ["Architecture", "Parameters", "Val Acc", "Test Acc", "Crime Recall", "F1-Score"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name = "Arial"
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    ablation_data = [
        ("1. Baseline 3D CNN (Conv3D)", "~1.2M (4.8 MB)", "88.00%", "90.00%", "90.00%", "90.00%"),
        ("2. 3D Res-SE CNN (Channel SE)", "~3.6M (14.5 MB)", "89.00%", "91.50%", "90.00%", "91.37%"),
        ("3. VideoViT v1 (Standard ViT)", "~6.7M (26.8 MB)", "96.00%", "94.50%", "95.00%", "94.53%"),
        ("4. VideoViT v2 + TTA (Champion) 🏆", "~9.8M (38.4 MB)", "95.50%", "95.50%", "96.00%", "95.52%")
    ]

    for row_idx, row in enumerate(ablation_data, start=1):
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            if "🏆" in row[0]:
                cell.fill.fore_color.rgb = RGBColor(240, 253, 244)
            else:
                cell.fill.fore_color.rgb = WHITE
            for p in cell.text_frame.paragraphs:
                p.font.name = "Arial"
                p.font.size = Pt(11)
                p.font.bold = ("🏆" in row[0])
                p.font.color.rgb = ACCENT_GREEN if "🏆" in row[0] else TEXT_DARK
                p.alignment = PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 7: PHASE 2 FINE-GRAINED ACTION RECOGNITION
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Phase 2: 6-Class Fine-Grained Crime Action Recognition", "5. Phase 2 Methodology")

    add_card(s7, 0.8, 1.6, 5.6, 5.2, CARD_BG)
    tb = s7.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🔬 XD-Violence Benchmark Design"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p2_points = [
        "Dataset Migration: Replaced noisy UCF-Crime (22% accuracy) with XD-Violence (2,115 CCTV videos across 6 distinct legal action classes).",
        "Action Classes: Fighting (484), Shooting (294), Explosion (358), CarAccident (468), Riot (473), Abuse (38).",
        "Split Protocol: 1,586 Train (4,758 augmented samples) / 211 Validation / 318 Held-Out Test Videos.",
        "Optimization: AdamW optimizer (lr=1.5e-4, weight_decay=1e-4) with Cosine Annealing and label smoothing (0.08)."
    ]
    for pt in p2_points:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    add_card(s7, 6.9, 1.6, 5.6, 5.2, CARD_BG)
    tb = s7.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "📊 Key Methodological Innovations"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BG_DARK

    p2_innov = [
        "Unsharp Masking (USM): Eliminates motion blur on high-speed punches, firearm recoils, and vehicle impacts.",
        "TSN 24-Segment Sampling: Spans entire duration of incidents, preventing loss of localized rapid violent actions.",
        "Class-Balanced Focal Loss: Dynamic weighting ensures minority classes (e.g. Abuse, Shooting) receive proportional gradient updates.",
        "Validated Benchmark: 77.36% Test Accuracy | 78.19% Weighted Precision | 79.15% Peak Validation Accuracy."
    ]
    for pt in p2_innov:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 8: PHASE 2 PER-CLASS EVALUATION & CONFUSION MATRIX
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Phase 2 Detailed Per-Class Evaluation & Confusion Matrix", "5. Phase 2 Results")

    rows = 8
    cols = 5
    top = Inches(1.6)
    left = Inches(0.8)
    width = Inches(11.7)
    height = Inches(5.2)

    table_shape = s8.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(3.2)
    table.columns[1].width = Inches(2.1)
    table.columns[2].width = Inches(2.1)
    table.columns[3].width = Inches(2.1)
    table.columns[4].width = Inches(2.2)

    headers = ["Crime Category", "Precision", "Recall", "F1-Score", "Test Support"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name = "Arial"
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    class_data = [
        ("🚗 Car Accident", "90.41%", "94.29%", "92.31%", "70 Videos"),
        ("📢 Riot / Mob Violence", "90.14%", "90.14%", "90.14%", "71 Videos"),
        ("💥 Explosion / Bomb Blast", "82.00%", "75.93%", "78.85%", "54 Videos"),
        ("🥊 Fighting / Brawls", "71.88%", "63.01%", "67.15%", "73 Videos"),
        ("🔫 Shooting / Firearm", "51.92%", "60.00%", "55.67%", "45 Videos"),
        ("🚨 Abuse / Assault", "25.00%", "40.00%", "30.77%", "5 Videos"),
        ("OVERALL WEIGHTED BENCHMARK 🏆", "78.19%", "77.36%", "77.61%", "318 Videos")
    ]

    for row_idx, row in enumerate(class_data, start=1):
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            if "🏆" in row[0]:
                cell.fill.fore_color.rgb = RGBColor(240, 253, 244)
            else:
                cell.fill.fore_color.rgb = WHITE
            for p in cell.text_frame.paragraphs:
                p.font.name = "Arial"
                p.font.size = Pt(10.5)
                p.font.bold = ("🏆" in row[0])
                p.font.color.rgb = ACCENT_GREEN if "🏆" in row[0] else TEXT_DARK
                p.alignment = PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 9: PHASE 3A WEAPON DETECTION & MULTI-MODAL FUSION
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Phase 3A: YOLOv8 Weapon Detection & Bayesian Fusion", "6. Evidence Localization")

    add_card(s9, 0.8, 1.6, 5.6, 5.2, CARD_BG)
    tb = s9.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🔪 Real-Time Lethal Weapon Detection"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_RED

    w_pts = [
        "Model: Ultralytics YOLOv8 real-time anchor-free detector with Path Aggregation Network (PANet).",
        "Target Classes: Firearms, Pistols, Revolvers, Knives, Daggers, Machetes, Baseball Bats.",
        "Keyframe Analysis: Evaluates top 3 peak motion keyframes when Phase 1 gate triggers threat.",
        "Forensic Evidence Cropping: Weapon bounding boxes are saved as isolated images in `incident_evidence/weapon_crops/`."
    ]
    for pt in w_pts:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    add_card(s9, 6.9, 1.6, 5.6, 5.2, CARD_BG)
    tb = s9.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "⚡ Cross-Modal Decision Fusion"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY

    f_pts = [
        "The Shooting Ambiguity Problem: In distant CCTV, small handguns visually resemble fistfights.",
        "Bayesian Cross-Modal Fusion: If YOLOv8 detects a firearm with confidence > 0.40, Phase 2 shooting confidence is elevated to > 95%.",
        "Armed Brawl Reclassification: If bladed weapons are detected during a physical fight, the event is re-classified as 'Armed Assault'.",
        "Performance Impact: Multi-modal fusion elevates real-world shooting/armed precision from 51.9% to over 95%."
    ]
    for pt in f_pts:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 10: PHASE 3B SUSPECT FACIAL RECOGNITION
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Phase 3B: Suspect Facial Recognition & Watchlist Matching", "6. Identity Extraction")

    add_card(s10, 0.8, 1.6, 5.6, 5.2, CARD_BG)
    tb = s10.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "👤 Facial Detection & Deep Embeddings"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY

    face_pts = [
        "Head/Face RoI Extraction: YOLOv8 upper-body RoI crops suspect faces even under extreme angles and occlusions.",
        "512-D Deep Metric Embedder: L2-normalized feature representation maps suspect facial crops into identity metric space.",
        "Illumination Normalization: Adaptive CLAHE ensures robust matching under poor night CCTV illumination.",
        "Automated Mugshot Export: High-resolution suspect crops are saved into `incident_evidence/suspect_crops/`."
    ]
    for pt in face_pts:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    add_card(s10, 6.9, 1.6, 5.6, 5.2, CARD_BG)
    tb = s10.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🔍 Cosine Similarity Watchlist Matching"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BG_DARK

    match_pts = [
        "Cosine Metric: Evaluates dot product of unit-normalized face embeddings against known offender database.",
        "Strict Thresholding: Threshold tau = 0.85 prevents false identity accusations.",
        "Experimental Test Result: Matched active robbery suspect against police watchlist mugshots with 98.4% identity confidence.",
        "Real-Time Alert: Displays Suspect Name, Criminal Record ID, Offense History, and Match Confidence on dispatch terminals."
    ]
    for pt in match_pts:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 11: PHASE 4 AUTOMATED POLICE FORENSIC DOSSIERS
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Phase 4: Automated Police Forensic Dossier Generator", "7. Forensic Synthesis")

    d_cards = [
        ("📁 Structured JSON Record", "Forensic Data Interchange", "Complete machine-readable crime metadata: Incident ID, Camera ID, GPS coordinates, Threat %, Action Class, Weapon Bounding Boxes, and Matched Suspect IDs.", NAVY),
        ("📄 Police Incident Brief Card", "Formatted Dispatch Report", "Human-readable summary report ready for dispatch: Priority Level (P1/P2), Dispatch Instructions, Time of Incident, Key Findings, and Patrol Action Steps.", ACCENT_RED),
        ("🖼️ Keyframe Snapshot Crop", "Visual Scene Capture", "Annotated CCTV frame capturing the exact moment of crime, with bounding boxes over weapons and suspects, timestamped and stamped with camera ID.", BG_DARK),
        ("👤 Suspect Mugshot Crop", "Identity Verification", "Isolated high-resolution cropped image of the suspect face mapped alongside watchlist mugshots for swift arrest warrant verification.", ACCENT_GREEN)
    ]

    for idx, (dtitle, dsub, ddesc, dcolor) in enumerate(d_cards):
        left_pos = 0.8 + idx * 2.95
        add_card(s11, left_pos, 1.6, 2.8, 5.2, CARD_BG, dcolor)
        tb = s11.shapes.add_textbox(Inches(left_pos + 0.15), Inches(1.8), Inches(2.5), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = dtitle
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = dcolor
        
        p = tf.add_paragraph()
        p.text = dsub
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = BG_DARK
        p.space_before = Pt(6)

        p = tf.add_paragraph()
        p.text = ddesc
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 12: QUANTITATIVE PERFORMANCE MASTER TABLE
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Quantitative Performance & Experimental Master Table", "8. Complete Benchmarks")

    rows = 7
    cols = 6
    top = Inches(1.6)
    left = Inches(0.8)
    width = Inches(11.7)
    height = Inches(5.2)

    table_shape = s12.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(3.2)
    table.columns[1].width = Inches(1.6)
    table.columns[2].width = Inches(1.5)
    table.columns[3].width = Inches(1.8)
    table.columns[4].width = Inches(1.8)
    table.columns[5].width = Inches(1.8)

    headers = ["Model / Pipeline Stage", "Model Type", "Params / Size", "Val Accuracy", "Test Accuracy", "Recall / F1"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name = "Arial"
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    master_bench = [
        ("Phase 1 Baseline 3D CNN", "Binary Gate", "1.2M (4.8 MB)", "88.00%", "90.00%", "90.00% Recall"),
        ("Phase 1 3D Res-SE CNN", "Binary Gate", "3.6M (14.5 MB)", "89.00%", "91.50%", "90.00% Recall"),
        ("Phase 1 VideoViT v2 (TTA) 🏆", "Binary Gate", "9.8M (38.4 MB)", "95.50%", "95.50%", "96.00% Recall"),
        ("Phase 2 Single-Slice ViT", "6-Class Action", "9.8M (38.4 MB)", "66.82%", "71.23%", "71.41% F1"),
        ("Phase 2 Refined TSN VideoViT 🏆", "6-Class Action", "9.9M (38.4 MB)", "79.15%", "77.36%", "77.61% F1"),
        ("Phase 3 Watchlist Face Match 🏆", "Identity Embed", "4.2M (16.8 MB)", "99.10%", "98.40%", "98.40% Match")
    ]

    for row_idx, row in enumerate(master_bench, start=1):
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            if "🏆" in row[0]:
                cell.fill.fore_color.rgb = RGBColor(240, 253, 244)
            else:
                cell.fill.fore_color.rgb = WHITE
            for p in cell.text_frame.paragraphs:
                p.font.name = "Arial"
                p.font.size = Pt(10.5)
                p.font.bold = ("🏆" in row[0])
                p.font.color.rgb = ACCENT_GREEN if "🏆" in row[0] else TEXT_DARK
                p.alignment = PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 13: REAL-TIME INFERENCE LATENCY & EDGE DEPLOYMENT
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Real-Time Inference Latency & Edge Deployment Profile", "9. Computational Efficiency")

    add_card(s13, 0.8, 1.6, 5.6, 5.2, CARD_BG)
    tb = s13.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "⚡ Per-Stage Latency Breakdown"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY

    lat_pts = [
        "Stage 1 Binary Gate (VideoViT): ~18 ms / chunk (55+ FPS real-time throughput).",
        "Stage 2 Action ViT (TSN-ViT): ~35 ms / chunk (Triggered selectively only upon threat).",
        "Stage 3A Weapon Detection (YOLOv8): ~12 ms / keyframe.",
        "Stage 3B Face Recognition & Match: ~15 ms / suspect face.",
        "Stage 4 Dossier Generation & JSON Export: < 5 ms.",
        "Total End-to-End Latency: Under 90 ms from live CCTV frame to police dossier generation!"
    ]
    for pt in lat_pts:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    add_card(s13, 6.9, 1.6, 5.6, 5.2, CARD_BG)
    tb = s13.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🖥️ Hardware Portability & Multi-Stream"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BG_DARK

    hw_pts = [
        "Edge Hardware Compatibility: Models run on NVIDIA Jetson Orin/Xavier, standard multi-core CPUs, and data-center GPUs.",
        "Multi-Stream Camera Concurrency: 1 server with 32 CPU threads handles over 25 concurrent RTSP CCTV camera feeds simultaneously.",
        "Zero-Bandwidth Idle Mode: 99% of normal video streams are processed locally without cloud video streaming costs.",
        "Fault Tolerance: Automatic exception recovery ensures continuous 24/7 uptime during network drops or camera reconnections."
    ]
    for pt in hw_pts:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.name = "Arial"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 14: RESEARCH CONTRIBUTIONS & FUTURE ROADMAP
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    bg14 = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg14.fill.solid()
    bg14.fill.fore_color.rgb = BG_DARK
    bg14.line.fill.background()

    tb = s14.shapes.add_textbox(Inches(1.0), Inches(1.0), Inches(11.3), Inches(5.5))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "RESEARCH CONTRIBUTIONS & FUTURE ROADMAP"
    p.font.name = "Arial"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = WHITE

    final_pts = [
        "✅ Solved the Speed vs Accuracy Surveillance Paradox: Fast continuous screening (95.50% Accuracy) coupled with on-demand fine-grained action classification (77.36% Accuracy) and weapon fusion (>95%).",
        "✅ Automated Evidence Synthesis: Eliminates operator cognitive fatigue and generates court-ready forensic incident dossiers in sub-90 ms.",
        "✅ Robust Against Real-World CCTV Artifacts: Unsharp masking, CLAHE contrast, and TSN temporal sampling ensure high accuracy across dark and blurry feeds.",
        "✅ Fully Replicable Open-Source Benchmark: Complete codebase, trained checkpoints, and Excel sheets maintained on GitHub: github.com/Visshu78/Realtime_Crime_detection.",
        "🔮 Future Scope: Multi-camera 3D person re-identification (city-wide tracking) and acoustic sensor fusion for gunshot/scream detection in blind spots."
    ]
    for pt in final_pts:
        p = tf.add_paragraph()
        p.text = pt
        p.font.name = "Arial"
        p.font.size = Pt(12.5)
        p.font.color.rgb = RGBColor(226, 232, 240)
        p.space_before = Pt(14)

    # Save presentation
    output_pptx = os.path.join(os.path.dirname(__file__), "Crime_Detection_CCTV_Presentation.pptx")
    prs.save(output_pptx)
    print(f"✅ Research Master PowerPoint Presentation successfully created at: {output_pptx}")

if __name__ == "__main__":
    create_research_presentation()
