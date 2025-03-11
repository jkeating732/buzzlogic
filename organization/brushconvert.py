import json
import os
import glob
import numpy as np
import cv2

json_file = '../project.json'  # Path to Label Studio export JSON file
npy_dir = 'npy'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

image_width, image_height = 640, 480

# Load the Label Studio export metadata
with open(json_file, 'r') as f:
    tasks = json.load(f)

for task in tasks:
    # Extract the task id and the original image filename from metadata.
    task_id = task['id']
    image_path = task['data']['image']
    original_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Search for the corresponding .npy file using a glob pattern "task-{task_id}-*.npy"
    pattern = os.path.join(npy_dir, f"task-{task_id}-*.npy")
    npy_files = glob.glob(pattern)
    if not npy_files:
        print(f"Warning: No .npy file found for task id {task_id}")
        continue
    # Use the first matching file (adjust if multiple masks per task are expected)
    npy_file = npy_files[0]
    
    # Load the mask from the .npy file and process it
    mask = np.load(npy_file).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to hold YOLO-formatted annotation lines for this mask
    annotation_lines = []
    for contour in contours:
        # Approximate the contour to a polygon (tune epsilon as needed)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Reshape and normalize polygon coordinates relative to the image dimensions
        polygon_points = approx.reshape(-1, 2)
        normalized_points = polygon_points / [image_width, image_height]
        
        # Flatten normalized points into a string: "x1 y1 x2 y2 ... xn yn"
        points_str = " ".join(str(coord) for point in normalized_points for coord in point)
        
        # Assuming a single class id (e.g., 0). Adjust if needed.
        annotation_line = f"0 {points_str}"
        annotation_lines.append(annotation_line)
    
    # Save the YOLO annotation to a text file named after the original image file
    output_file = os.path.join(output_dir, f"{original_basename}.txt")
    with open(output_file, 'w') as f:
        for line in annotation_lines:
            f.write(line + "\n")
    
    print(f"Processed task id {task_id} -> {output_file}")

print("Conversion complete.")
