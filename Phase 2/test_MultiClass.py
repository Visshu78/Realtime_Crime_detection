import cv2
import torch
import numpy as np
import time
import os
from MultiClassModel import SlowFastNet, Config, compute_motion, CRIME_CLASSES, IDX_TO_CLASS

# Device configuration
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
except Exception:
    device = torch.device("cpu")
    print("Device: CPU")

# Load Stage 2 Checkpoint
model = SlowFastNet(num_classes=len(CRIME_CLASSES)).to(device)

if os.path.exists(Config.BEST_MODEL_PATH):
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()
    print("Stage 2 SlowFast Net loaded successfully from:", Config.BEST_MODEL_PATH)
else:
    print(f"Warning: Model checkpoint '{Config.BEST_MODEL_PATH}' not found yet. Please train Stage 2 first.")

def analyze_crime_video(video_path, use_tta=True):
    """
    Reads a video, extracts 24-frame clips, and predicts the specific crime action using SlowFast Net.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n" + "=" * 60)
    print(f"STAGE 2: SLOWFAST CRIME ACTION ANALYSIS")
    print("=" * 60)
    print(f"Video File   : {video_path}")
    print(f"FPS          : {fps}")
    print(f"Total Frames : {total_frames}")
    if fps > 0:
        print(f"Duration     : {total_frames / fps:.2f} seconds")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, Config.TARGET_SIZE).astype("float32") / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print("Error: Could not read video frames.")
        return

    # Create 24-frame clips
    clips = []
    for start in range(0, len(frames), Config.MAX_FRAMES):
        end = start + Config.MAX_FRAMES
        clip = frames[start:end]
        if len(clip) > 0 and len(clip) < Config.MAX_FRAMES:
            while len(clip) < Config.MAX_FRAMES:
                clip.append(clip[-1])
        if len(clip) == Config.MAX_FRAMES:
            clips.append(np.array(clip))

    print(f"Total 24-frame clips to analyze: {len(clips)}")
    print("=" * 60)

    clip_predictions = []
    all_probs = []

    for i, clip in enumerate(clips):
        blended = compute_motion(clip)
        tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            if use_tta:
                out1 = model(tensor)
                flipped = torch.flip(tensor, dims=[4])
                out2 = model(flipped)
                probs = ((torch.softmax(out1, dim=1) + torch.softmax(out2, dim=1)) / 2.0).cpu().numpy()[0]
            else:
                out = model(tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]

        top_idx = np.argmax(probs)
        pred_class = IDX_TO_CLASS[top_idx]
        conf = probs[top_idx]
        all_probs.append(probs)
        clip_predictions.append((pred_class, conf))

        print(f"Clip [{i+1:02d}/{len(clips):02d}] ──► Top Action: {pred_class:<15} (Confidence: {conf*100:6.2f}%)")

    # Aggregate overall decision
    avg_probs = np.mean(all_probs, axis=0)
    top_3_indices = np.argsort(avg_probs)[::-1][:3]

    print("\n" + "=" * 60)
    print("             FINAL STAGE 2 ACTION VERDICT")
    print("=" * 60)
    print(f"🏆 PRIMARY CRIME ACTION : {IDX_TO_CLASS[top_3_indices[0]]} ({avg_probs[top_3_indices[0]]*100:.2f}% Confidence)")
    print("-" * 60)
    print("TOP 3 DETECTED CATEGORIES:")
    for rank, idx in enumerate(top_3_indices, 1):
        print(f"  {rank}. {IDX_TO_CLASS[idx]:<18} : {avg_probs[idx]*100:6.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_input = sys.argv[1]
    else:
        video_input = input("\nEnter video path for Stage 2 analysis: ").strip().strip('"')

    try:
        t0 = time.time()
        analyze_crime_video(video_input)
        print(f"\nElapsed time: {time.time() - t0:.2f} seconds")
    except Exception as e:
        print("\nError during execution:", e)
