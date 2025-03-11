import os
import sys
import torch
from ultralytics import YOLO
import os

##### ONLY FOR USE WITH COMET.ML #####
#####     https://comet.com      #####
os.environ["COMET_PROJECT_NAME"] = "beehive-segment"
os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "10"
######################################

dataset = os.getenv("DATASET_SEGMENT")

if dataset is not None:
    print("Segment dataset selected as " + dataset)
else:
    print("No dataset specified in .env file")
    sys.exit(1)

model = YOLO("../models/yolo11-seg.yaml") # replace with weighted model 

torch.cuda.empty_cache()

results = model.train(data="../datasets/dataset_segment.yaml",
    task='segment',
    epochs=600,
    imgsz=1280, 
    batch=2, # small batch for now
    plots=True, 
    augment=True, 
    visualize=True,
    save_period=10,
    patience=125,
    cache=False # this helps stop OOM issues on my machine
)

results = model.val()
results = model("https://drkilligans.com/cdn/shop/articles/Queen-bee-eggs-per-day_bdd109b5-e957-41ba-9994-9efd31c584a6.png") # test when done

results[0].show()

success = model.export(format="onnx")
