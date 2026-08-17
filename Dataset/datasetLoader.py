import kagglehub
import shutil
import os

# Download dataset from Kaggle
# path = kagglehub.dataset_download(
#     "mohamedmustafa/real-life-violence-situations-dataset"
# )
path='';
path = input("Enter the path to the downloaded dataset (or press Enter to use the default path): ") or path;
print("Downloaded to:")
print(path)

# Source dataset folder
source = os.path.join(path, "Real Life Violence Dataset")

# Destination folder
destination = "/home/amithk/Desktop/CrimeDetectionCCTV/Dataset"

# Create destination folder if it doesn't exist
os.makedirs(destination, exist_ok=True)

# Copy the dataset
shutil.copytree(
    source,
    destination,
    dirs_exist_ok=True
)

print("\nDataset copied successfully to:")
print(destination)