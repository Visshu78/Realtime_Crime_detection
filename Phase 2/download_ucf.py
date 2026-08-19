import os
import shutil
import kagglehub

destination = "/home/amithk/Desktop/CrimeDetectionCCTV/UCF Dataset"
os.makedirs(destination, exist_ok=True)

print("=" * 60)
print("Starting download of odins0n/ucf-crime-dataset via kagglehub...")
print("Destination folder:", destination)
print("=" * 60)

path = kagglehub.dataset_download("odins0n/ucf-crime-dataset")
print(f"\nDownload completed in cache: {path}")

print("\nOrganizing dataset into 'UCF Dataset/'...")
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(destination, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("\n" + "=" * 60)
print("✅ UCF-Crime Dataset successfully placed in 'UCF Dataset/'")
print("Classes & Folders available:")
for f in sorted(os.listdir(destination)):
    item_path = os.path.join(destination, f)
    if os.path.isdir(item_path):
        num_files = len(os.listdir(item_path))
        print(f"  📁 {f} ({num_files} items)")
    else:
        print(f"  📄 {f}")
print("=" * 60)
