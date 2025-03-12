import os
import sys
import comet_ml
import torch
from ultralytics import YOLO
from dotenv import load_dotenv

##### ONLY FOR USE WITH COMET.ML #####
#####     https://comet.com      #####
os.environ["COMET_PROJECT_NAME"] = "beehive"
os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "10"
######################################

weight = os.getenv("MODEL_DETECT")

if weight is not None:
    print("Model dataset selected as " + weight)
else:
    print("No model specified in .env file")
    sys.exit(1)

model = YOLO(weight)

torch.cuda.empty_cache()

results = model.train(data="../datasets/dataset_detect.yaml",
    task='detect',
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
