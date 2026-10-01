# Sargassum Image Classification

A deep learning project that classifies beach photos into five levels of sargassum (seaweed) abundance: `nada`, `bajo`, `moderado`, `abundante` and `excesivo` (none, low, moderate, abundant, excessive).

It was built for a **school competition on Kaggle** in a neural networks course (B.Eng. in Artificial Intelligence, ESCOM - IPN). The dataset and the test labels belong to the organizers, so the data is not included in this repo and **no performance metrics are reported here**.

## Approach

- `preprocess.py` cleans the labels file, removes missing or corrupt images, removes exact duplicates (SHA-256) and creates a stratified 70/15/15 train, validation and test split.
- `train.py` fine-tunes a pretrained **EfficientNet-B0** (PyTorch) in two stages:
  1. Train only the new classification head with the backbone frozen.
  2. Fine-tune the last feature blocks with a lower learning rate.
- The classes are imbalanced (almost half of the images are "nada"), so the loss uses class weights and the best checkpoint is chosen by validation macro-F1.
- Training images are augmented with random crops, flips, small rotations and color jitter.
- `predict.py` writes a CSV with one predicted label per image, in the format of the competition submission.

## How to run

```bash
pip install -r requirements.txt
```

Put the dataset in a folder called `DataSet` with this layout:

```
DataSet/
  images/          all the .jpg files
  labels/
    labels.csv     image_name, place, date, label, scene
    test.csv       image_name (images without labels)
```

Then run the scripts in this order:

```bash
python preprocess.py
python train.py
python predict.py
```

The trained weights are saved in `models/`. Training uses the GPU if there is one. Messages and comments in the code are in Spanish, as in the original coursework.

## Technologies

Python, PyTorch, torchvision, scikit-learn, pandas, Pillow

## Credits

Alberto Gomez Mendez. The "Sargazo Dataset" was provided for the competition (project lead: Juan Irving Vasquez, with support from CONACYT).
