import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

# Load a model
model = YOLO("models/model_segment.pt")  # change as needed

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

results = model.predict(images)  # return a list of Results objects

for result in results:
    plot = result.plot(labels=True, boxes=True, probs=False)
    plot_rgb = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.imshow(plot)
    plt.show()