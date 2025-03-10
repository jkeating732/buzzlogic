### buzzlogic
YOLO11 detect model for the modern beehive

### Structuring / Important Info
Models that can be trained off of are located in the `models/` directory. Only use `models/yolo11-obb` if you plan on training a new model from scratch. Otherwise, use `models/model.pt`. Specify the model in `train.py` like so:

```python
model = YOLO("models/model.pt") # replace with weighted model
```

The `organization` directory contains tools that will help you organize and prepare a modified dataset for training. Use `organization/annotations.py` to fetch the number of annotations for a class in the dataset, as well as the total number of annotations in the dataset. This is *very important* for weighting your classes during training. 

Modify weights in `dataset.yaml`. Use the formula `(TOTAL ANNOTATIONS) / (NUM CLASSES * ANNOTATIONS FOR CLASS X)` to calculate the weight for each class. 

The `organization` directory also contains `sort.py`. This organizes the labels and images in your dataset into `train/` and `val/` subdirectories. 80% of images and their respective labels will go into the `train/` subdirectories, while the remaining 20% will be randomly organized into the `val/` subdirectories. It is vital that you do this prior to training to ensure even representation of data. Remember to set the environment variable `DATASET` to the path of your dataset prior to running `organization/sort.py`.

On Linux:
```bash
export DATASET=/path/to/dataset
```

On Windows:
```bash
set DATASET=/path/to/dataset
```

### This project is INCOMPLETE, there will be bugs!

[Dataset Download (unsorted)](https://unlimited.beer:9443/index.php/s/K7YfALD2atnL89T)
