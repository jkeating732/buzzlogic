import os
import sys
import comet_ml
import torch
from ultralytics import YOLO
from dotenv import load_dotenv

##### ONLY FOR USE WITH COMET.ML #####
#####     https://comet.com      #####
os.environ["COMET_PROJECT_NAME"] = "beehive-segment"
os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "10"
######################################

load_dotenv()

dirname = os.path.dirname(__file__)

weight = os.getenv("MODEL_SEGMENT")

if weight is not None:
    print("Segment model selected as " + weight)
else:
    print("No model specified in .env file")
    sys.exit(1)

model = YOLO(weight) # replace with weighted model 

torch.cuda.empty_cache()

results = model.train(data=os.path.join(dirname, "../datasets/dataset_segment.yaml"),
    task='segment',
    epochs=1000,
    imgsz=1536, 
    batch=4,
    scale=0.5,
    cls=1.5,
    plots=True, 
    label_smoothing=0.1,
    iou=0.2,
    augment=True,
    visualize=False,
    save_period=10,
    patience=50,
    mosaic=0.0,
    multi_scale=True,
    degrees=10,
    shear=2,
    perspective=0.001,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4
)

success = model.export(format="onnx")
