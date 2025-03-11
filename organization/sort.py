import os
import sys
import shutil
import random

from dotenv import load_dotenv

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
    
    # Create train and val directories if they don't exist
    for dir_path in [labels_dir, images_dir]:
        for subdir in ['train', 'val']:
            os.makedirs(os.path.join(dir_path, subdir), exist_ok=True)
    
    # Collect image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    image_files = []
    for filename in os.listdir(images_dir):
        if os.path.isfile(os.path.join(images_dir, filename)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(filename)
    
    # Collect valid pairs
    pairs = []
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        if os.path.isfile(label_path):
            pairs.append((
                os.path.join(labels_dir, label_file),
                os.path.join(images_dir, image_file)
            ))
    
    # Shuffle pairs randomly
    random.shuffle(pairs)
    
    # Calculate split index using rounding for better edge case handling
    split_index = round(0.8 * len(pairs))
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]
    
    # Move files to respective directories
    def move_files(pairs_list, dest):
        for label_path, image_path in pairs_list:
            # Move label
            label_dest = os.path.join(labels_dir, dest, os.path.basename(label_path))
            shutil.move(label_path, label_dest)
            
            # Move image
            image_dest = os.path.join(images_dir, dest, os.path.basename(image_path))
            shutil.move(image_path, image_dest)
    
    move_files(train_pairs, 'train')
    move_files(val_pairs, 'val')
    
    print(f"Split complete: {len(train_pairs)} pairs moved to train, {len(val_pairs)} pairs moved to val")

if __name__ == '__main__':
    main()