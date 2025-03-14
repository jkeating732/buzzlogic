import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

# Make use of both models
segment = YOLO("models/model_segment.pt")
detect = YOLO("models/model_detect.pt")

image = input("Please specify a path or URL to an image:\n")

first = segment.predict(image, imgsz=2016)

segment_plot = first[0].plot(labels=True, boxes=True, probs=False)
plot_rgb = cv2.cvtColor(segment_plot, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis("off")
plt.imshow(plot_rgb)
plt.savefig("segmented.jpg", bbox_inches='tight')

second = detect.predict("segmented.jpg", imgsz=2016)
second[0].show()