import os
import shutil
import kagglehub

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "Phase" in os.path.dirname(os.path.abspath(__file__)) else os.path.dirname(os.path.abspath(__file__))
DESTINATION = os.path.join(PROJECT_ROOT, "XD_Violence_Dataset")
os.makedirs(DESTINATION, exist_ok=True)

print("=" * 65)
print("🚀 Starting download of XD-Violence Dataset via kagglehub...")
print("Destination folder:", DESTINATION)
print("=" * 65)

# Download XD-Violence
path = kagglehub.dataset_download("bypktt/xd-violence")
print(f"\nDownload completed in cache: {path}")

print("\nOrganizing dataset into 'XD_Violence_Dataset/'...")
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(DESTINATION, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("\n" + "=" * 65)
print("✅ XD-Violence Dataset successfully placed in 'XD_Violence_Dataset/'")
print("Classes & Folders available:")
for f in sorted(os.listdir(DESTINATION)):
    item_path = os.path.join(DESTINATION, f)
    if os.path.isdir(item_path):
        num_files = len(os.listdir(item_path))
        print(f"  📁 {f} ({num_files} items)")
    else:
        print(f"  📄 {f}")
print("=" * 65)
