# KakuOCR
Write Japanese with your finger, and it will be recognized (hopefully,)

# Overview

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
- Fix the laggy drawing for real this time
- Add more data

### Model Structure

Trained on 51200 images of Japanese characters (ETL8BC1)

160 occurrences of 320 characters (64x64) (Pretty basic kana and kanji, can add the other ETL8B datasets later)

The model is a Convolutional Neural Network (CNN) with the following structure:

1. Feature Extraction:
   - Conv2d(1, 32, 3x3, padding=1) + ReLU
   - Conv2d(32, 64, 3x3, padding=1) + ReLU
   - MaxPool2d(2x2)
   - Conv2d(64, 128, 3x3, padding=1) + ReLU
   - Conv2d(128, 128, 3x3, padding=1) + ReLU
   - MaxPool2d(2x2)
   - Conv2d(128, 256, 3x3, padding=1) + ReLU
   - AdaptiveAvgPool2d(1x1)

2. Classifier:
   - Dropout(0.5)
   - Linear(256, 512) + ReLU
   - Dropout(0.5)
   - Linear(512, num_classes)

3. Output: Log Softmax

The model is trained using:
- Loss function: Negative Log Likelihood Loss (NLL Loss)
- Optimizer: Adam
- Batch size: 64
- Number of epochs: Up to 100 with early stopping
- Learning rate scheduler: ReduceLROnPlateau

Data augmentation techniques:
- Random affine transformations (rotation, translation, scaling, shearing)
- Random perspective
- Random inversion
- Random Gaussian blur

The dataset used is the ETL8B dataset, which contains Japanese characters. The data is split into 80% training and 20% testing sets.

Output: Predicted Japanese character (from the set of characters in the ETL8B dataset)

### Analysis

The model performs exceptionally well on the ETL8B dataset, achieving an accuracy of over 98% on the test set. Training for about 100 epochs yields this high accuracy, with the loss decreasing significantly throughout the training process.

Key observations:
1. The model quickly improves from an initial low accuracy to over 90% within the first 10 epochs.
2. By epoch 30, the accuracy reaches around 97% and continues to improve gradually.
3. In the final epochs, the accuracy consistently stays above 98%, with peaks reaching 98.5%.
4. The loss decreases from initial values around 5.7 to final values around 0.05.

The current model shows significant improvement over the previous version:
- It no longer "completely fails on anything that's not in the training set."
- The accuracy has improved from about 80% to over 98%.
- The model now generalizes well to hand-drawn input, despite being trained on the ETL8B dataset.

Potential areas for further improvement:
1. Expanding the dataset: While the model performs well with ETL8BC1, incorporating more data from other ETL8B datasets could further improve its robustness and coverage of characters.
2. Fine-tuning on hand-drawn data: Although the model generalizes well, creating a small dataset of hand-drawn characters and fine-tuning on it could potentially improve performance on user input.
3. Exploring more complex architectures: While the current CNN performs well, experimenting with more advanced architectures (e.g., residual networks, attention mechanisms) might yield even better results.
4. Optimizing for inference speed: Depending on the target hardware, the model could be optimized for faster inference while maintaining high accuracy.

Overall, the current model demonstrates strong performance and practical usability for recognizing both printed and hand-drawn Japanese characters. Although I suspect it will completely botch anything that isn't kana or basic kanji.

## Want to help?

I'm a busy college student, if you want to help out, feel free to directly contribute via issues or even better, make a PR. Email me at [kadenbilyeu@proton.me](mailto:kadenbilyeu@proton.me) if you have any questions.
