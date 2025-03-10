from ultralytics import YOLO

# Load a model
model = YOLO("models/model.pt")  # pretrained model

numImages = int(input("Enter the number of images to feed the model:\n"))

if numImages >= 1:
    print(str(numImages) + " images selected")

    iteration = 0
    images = []

    while iteration < numImages: 
        image = input("Enter an image path or URL:\n")
        images.append(image)
        iteration += 1
else:
    print("Please enter a whole number greater than zero")

results = model(images)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk