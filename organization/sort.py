import os
import sys
import shutil
import random
import numpy as np
from dotenv import load_dotenv
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

load_dotenv()
dataset = os.getenv("WORKING_DATASET")
if dataset is not None:
    print("Working dataset selected as " + dataset)
else:
    print("No dataset specified in .env file")
    sys.exit(1)

def main():
    labels_dir = os.path.join(dataset, 'labels')
    images_dir = os.path.join(dataset, 'images')
    
    for dir_path in [labels_dir, images_dir]:
        for subdir in ['train', 'val']:
            os.makedirs(os.path.join(dir_path, subdir), exist_ok=True)
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    image_files = []
    for filename in os.listdir(images_dir):
        filepath = os.path.join(images_dir, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(filename)
    
    pairs = []
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        if os.path.isfile(label_path):
            pairs.append((os.path.join(labels_dir, label_file), os.path.join(images_dir, image_file)))
    
    all_pairs = []
    all_labels = []
    all_class_ids = set()
    for label_path, _ in pairs:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    all_class_ids.add(int(parts[0]))
    if not all_class_ids:
        print("No class labels found!")
        sys.exit(1)
    num_classes = max(all_class_ids) + 1
    for label_path, image_path in pairs:
        classes_in_file = set()
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes_in_file.add(int(parts[0]))
        multi_hot = np.zeros(num_classes, dtype=int)
        for cls in classes_in_file:
            multi_hot[cls] = 1
        all_pairs.append((label_path, image_path))
        all_labels.append(multi_hot)
    all_labels = np.array(all_labels)
    
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    all_indices = np.arange(len(all_pairs))
    for train_index, val_index in msss.split(all_indices, all_labels):
        train_pairs = [all_pairs[i] for i in train_index]
        val_pairs = [all_pairs[i] for i in val_index]
    
    def move_files(pairs_list, dest):
        for label_path, image_path in pairs_list:
            label_dest = os.path.join(labels_dir, dest, os.path.basename(label_path))
            shutil.move(label_path, label_dest)
            image_dest = os.path.join(images_dir, dest, os.path.basename(image_path))
            shutil.move(image_path, image_dest)
    
    move_files(train_pairs, 'train')
    move_files(val_pairs, 'val')
    
    print(f"Stratified split complete: {len(train_pairs)} pairs moved to train, {len(val_pairs)} pairs moved to val")

if __name__ == '__main__':
    main()