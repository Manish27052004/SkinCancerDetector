import os
import shutil
import random

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Set random seed for reproducible splits
random.seed(42)

source_base = "melanoma_cancer_dataset"
target_base = "dataset"

# Target structure
for split in ['train', 'val', 'test']:
    for cls in ['benign', 'melanoma']:
        create_dir(os.path.join(target_base, split, cls))

print("Preparing testing set...")
# Copy testing set (and rename malignant -> melanoma)
test_source_benign = os.path.join(source_base, 'test', 'benign')
test_source_malignant = os.path.join(source_base, 'test', 'malignant')

for file in os.listdir(test_source_benign):
    shutil.copy(os.path.join(test_source_benign, file), os.path.join(target_base, 'test', 'benign', file))
    
for file in os.listdir(test_source_malignant):
    shutil.copy(os.path.join(test_source_malignant, file), os.path.join(target_base, 'test', 'melanoma', file))

print("Preparing training and validation sets...")
# Split training into train/val
val_split_ratio = 0.2

for cls_source, cls_target in [('benign', 'benign'), ('malignant', 'melanoma')]:
    source_dir = os.path.join(source_base, 'train', cls_source)
    files = os.listdir(source_dir)
    random.shuffle(files)
    
    val_size = int(len(files) * val_split_ratio)
    val_files = files[:val_size]
    train_files = files[val_size:]
    
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(target_base, 'val', cls_target, file))
        
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(target_base, 'train', cls_target, file))

print("Dataset successfully structured!")
