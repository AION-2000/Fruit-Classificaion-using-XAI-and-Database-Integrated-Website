import os
import shutil
import random
from tqdm import tqdm

# Paths
SOURCE_DIR = 'data/fruits-360_original-size'
DEST_DIR = 'data/fruits-360_split'
SPLITS = ['train', 'val', 'test']
SPLIT_RATIOS = [0.7, 0.15, 0.15]

# Create folders
for split in SPLITS:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

# Get class folders
classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for fruit_class in tqdm(classes, desc="Splitting dataset"):
    class_dir = os.path.join(SOURCE_DIR, fruit_class)
    images = os.listdir(class_dir)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(SPLIT_RATIOS[0] * n_total)
    n_val = int(SPLIT_RATIOS[1] * n_total)

    split_data = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }

    for split in SPLITS:
        split_class_dir = os.path.join(DEST_DIR, split, fruit_class)
        os.makedirs(split_class_dir, exist_ok=True)
        for img in split_data[split]:
            src = os.path.join(class_dir, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copyfile(src, dst)

print("✅ Dataset splitting complete.")
