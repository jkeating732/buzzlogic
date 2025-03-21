## buzzlogic

#### View training statistics for models at Comet.ml 
[Detect Model (finds mites, classifies bees)](https://www.comet.com/whoffie/beehive/view) | [Segment Model (classifies cells, brood patterns)](https://www.comet.com/whoffie/beehive-segment/view)

### Setup

Before you get started with buzzlogic, ensure that you have a CUDA-compatible GPU. AMD's ROCM architecture will work as well, but it requires you run a special version of the `pytorch` library while being on Linux.

1. Clone the repository and install system requirements.
```bash
git clone https://github.com/whoffie/buzzlogic
cd buzzlogic
pip install -r requirements.txt
```
If installing requirements failed, please ensure that you are using Python 3.10. If an issue still persists, please [create an issue](https://github.com/Whoffie/buzzlogic/issues).

2. Download the dataset(s)
###### This step is only required if you wish to train the models 
Depending on your specific use case, you may want to use the segment model (which identifies brood types and capped honey), the detect model (which identifies types of bees and varroa), or both. [Download the datasets here](https://unlimited.beer:9443/index.php/s/K7YfALD2atnL89T).

3. Set up your environment file
###### This step is only required if you wish to train the models
Once the datasets have been downloaded, you will need to set up your environment file. Rename `.env.example` to `.env` and add the proper paths for each variable.

4. Run the model
To run a single model, execute:
```bash
python interpret.py
```
Modify the path to the model as needed inside `interpret.py`
```python
model = YOLO("models/model_segment.pt")  # change as needed
```

### Ultralytics
This model runs with the help of the Ultralytics Python library. If you are not familiar with the `train()` method or something else, I highly recommend you visit the [Ultralytics docs page](https://docs.ultralytics.com/modes/train/).

### Structuring / Important Info
Models that can be trained off of are located in the `models/` directory. Only use `models/yolo11-obb.yaml` if you plan on training a new model from scratch. Otherwise, use `models/model.pt`. Specify the model you wish to use in your `.env` file.

The `organization` directory contains tools that will help you organize and prepare a modified dataset for training. Use `organization/annotations.py` to fetch the number of annotations for a class in the dataset, as well as the total number of annotations in the dataset. This is *very important* for weighting your classes during training.

#### Getting Started With the Dataset

Modify weights in `dataset.yaml`. Use the formula `(TOTAL ANNOTATIONS) / (NUM CLASSES * ANNOTATIONS FOR CLASS X)` to calculate the weight for each class. Alternatively, `organization/sort.py` will do this for you if you specify the number of classes in the project. For example:

```bash
python organization/sort.py --annotation "Drone Brood" --class-count 4
```

This command will give you the total number of annotations and the proper weight value for the `Drone Brood` class, provided that the class count is accurate. Remember to have the environment variable `WORKING_PROJECT_FILE` set to the path of the relevant project file to scan.

The `organization` directory also contains `sort.py`. This organizes the labels and images in your dataset into `train/` and `val/` subdirectories. 80% of images and their respective labels will go into the `train/` subdirectories, while the remaining 20% will be randomly organized into the `val/` subdirectories. It is vital that you do this prior to training to ensure even representation of data. Remember to update your .env file with relevant dataset/project JSON paths. To sort the dataset with an 80/20 distribution, simply run:

```bash
python organization/sort.py 
```

### This project is INCOMPLETE, there will be bugs!

[Download Datasets (unsorted)](https://unlimited.beer:9443/index.php/s/K7YfALD2atnL89T)
