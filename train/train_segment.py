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
    imgsz=1560, 
    batch=3,
    plots=True, 
    augment=True, 
    visualize=True,
    save_period=10,
    patience=200,
    cache=False, # this helps stop OOM issues on my machine
    show_boxes=False
)

results = model.val()
results = model("https://drkilligans.com/cdn/shop/articles/Queen-bee-eggs-per-day_bdd109b5-e957-41ba-9994-9efd31c584a6.png") # test when done

results[0].show()

success = model.export(format="onnx")
