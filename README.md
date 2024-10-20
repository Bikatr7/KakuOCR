# KakuOCR
Write Japanese with your finger, and it will be recognized (hopefully,)

# Overview

## Data

You can get the data used from [here](http://etlcdb.db.aist.go.jp/)

## To use

### Requirements

Python (I'm using 3.11.4)

opencv-python

mediapipe

numpy 

pynput

bitstring

Pillow

torch

torchvision

### Setup

Run `pip install -r requirements.txt` to install the dependencies.

Run `python train.py` to train the model. (or use the existing model included `kakuocr_model.pth`, make sure to update the path in `main.py` if you do this)

Run `python main.py`

### Usage

You can hold down the `d` key to trigger drawing, make sure your hand is visible and mapped. Note this is a bit laggy.

Press `s` to submit the drawing, which will display the predicted character in the top left corner.

Press `c` to clear the canvas.

Press/Hold `z` to undo.

Press/Hold `shift+z` to redo.

Press `q` to quit.

### Goals

- ~~Better writing experience (less lag, less jittery)~~
- ~~Should be able to get the writing and display it in the top left or something rather than just the canvas~~
- ~~OCR (I want to 'train' our own model) (We can use the  J-HOCR, JAIST, Kondate, ETL datasets)~~
- ~~Need to get the actual prediction correct~~
- ~~Optimize~~
- ~~Fix the laggy drawing for real this time~~
- ~~Add more data~~
- Try to address relative size playing an issue in detection
- Make it to where it can detect multiple characters

### Model Structure

This data may be slightly off.

Trained on 153,916 images of Japanese characters (ETL8B2C1, ETL8B2C2, ETL8B2C3)

161 occurrences of 956 unique characters (64x64)

The model is a Convolutional Neural Network (CNN) with the following structure:

1. Feature Extraction:
   - Conv2d(1, 32, 3x3, padding=1) + BatchNorm2d + ReLU
   - MaxPool2d(2x2)
   - Conv2d(32, 64, 3x3, padding=1) + BatchNorm2d + ReLU
   - MaxPool2d(2x2)
   - Conv2d(64, 128, 3x3, padding=1) + BatchNorm2d + ReLU
   - MaxPool2d(2x2)
   - Conv2d(128, 256, 3x3, padding=1) + BatchNorm2d + ReLU
   - AdaptiveAvgPool2d(1x1)

2. Classifier:
   - Dropout(0.5)
   - Linear(256, 512) + ReLU
   - Dropout(0.5)
   - Linear(512, num_classes)

The model is trained using:
- Loss function: Cross Entropy Loss with label smoothing (0.1)
- Optimizer: Adam with weight decay (1e-5)
- Batch size: 128
- Number of epochs: Up to 200 with early stopping (patience: 20)
- Learning rate scheduler: ReduceLROnPlateau
- Initial learning rate: 0.0003

Data augmentation techniques:
- Random rotation (±10 degrees)
- Random affine transformations (translation, scaling)
- Elastic transform
- Normalization

The dataset used combines ETL8B2C1, ETL8B2C2, and ETL8B2C3, which contain a wide range of Japanese characters including kana and kanji. The data is split into 80% training and 20% testing sets.

Output: Predicted Japanese character (from the set of 956 unique characters in the combined ETL8B datasets)

### Analysis

The model performs exceptionally well on the combined ETL8B dataset, achieving an accuracy of over 96% on the test set. Training for about 67 epochs yields this high accuracy, with the loss decreasing significantly throughout the training process.

Fine tuning this on my hand drawn data would probably improve the accuracy even more.

Key observations:
1. The model quickly improves from an initial low accuracy to over 90% within the first 20-25 epochs.
2. By epoch 30, the accuracy reaches around 95% and continues to improve gradually.
3. In the later epochs (50+), the accuracy consistently stays above 95.7%, with peaks reaching 96.5%.
4. The loss decreases from initial values around 6.8 to final values around 1.7-1.8.

The current model shows significant improvement over the previous version:
- It now handles a much larger set of characters (956 vs 320 previously).
- The accuracy has improved from about 98% on a smaller character set to over 96% on a much larger and more diverse character set.
- The model demonstrates good generalization despite the increased complexity of the task.

Potential areas for further improvement:
1. Fine-tuning on hand-drawn data: Although the model generalizes well, creating a small dataset of hand-drawn characters and fine-tuning on it could potentially improve performance on user input.
2. Exploring more complex architectures: While the current CNN performs well, experimenting with more advanced architectures (e.g., residual networks, attention mechanisms) might yield even better results.
3. Optimizing for inference speed: Depending on the target hardware, the model could be optimized for faster inference while maintaining high accuracy.

Overall, the current model demonstrates strong performance and practical usability for recognizing a wide range of Japanese characters, including both kana and a substantial set of kanji. As even with that much data 900 is still a bit low.

## Want to help?

I'm a busy college student, if you want to help out, feel free to directly contribute via issues or even better, make a PR. Email me at [kadenbilyeu@proton.me](mailto:kadenbilyeu@proton.me) if you have any questions.
