# Detect objects
Program for detecting objects in an image.

This program takes an aerial image and detects humans.

## Features
- Find objects in image and draw bounding boxes with probabilities
- Create a dataset using ground truth
- Train a model using a dataset
- View a dataset

# Setup
```
git clone git@github.com:ashwinmr/object_detect.git
cd object_detect
pip install -r requirements.txt
```

# Usage

There are subcommands for performing different functions

## Detect objects using an image and a trained model

```
python main.py detect <path/to/image> <path/to/model.h5>
```

Example:
```
python main.py detect 'TestImage.jpg' 'model.h5'
```

Example output:
![screenshot](screenshots/sc_1.png)

## Create a training/test dataset using an image and ground truth

You can use a tool such as labelImg to create ground truth bounding boxes for an image

You can then use this program to create a dataset of positive and negative examples using the image and ground truth

```
python main.py create <path/to/image> <path/to/ground_truth.xml> <path/to/output/training_set.p> <path/to/output/test_set.p>
```

Example:
```
python main.py create 'TestImage.jpg' 'GroundTruth.xml' 'training_set.p' 'test_set.p'
```

## Use a dataset to train the model and evaluate it

```
python main.py train <path/to/dataset.p> <path/to/output/model.h5>
```

Example:
```
python main.py train 'training_set.p` 'model.h5'
```

Evaluate the model using
```
python main.py eval `test_set.p` `model.h5'
```
