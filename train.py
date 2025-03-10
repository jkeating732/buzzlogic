import comet_ml
import torch
from ultralytics import YOLO
import os

os.environ["COMET_PROJECT_NAME"] = "beehive"
os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "10"  # Log every validation batch
os.environ["COMET_PROJECT_NAME"] = "beehive"


model = YOLO("models/model.pt") # replace with weighted model

torch.cuda.empty_cache()

results = model.train(data="dataset.yaml",
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
