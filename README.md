# KakuOCR
Write Japanese with your finger, and it will be recognized.

## Overview

"Planned things"

- ~~Better writing experience (less lag, less jittery)~~
- ~~Should be able to get the writing and display it in the top left or something rather than just the canvas~~
- ~~OCR (I want to 'train' our own model) (We can use the  J-HOCR, JAIST, Kondate, ETL datasets)~~
- Need to get the actual prediction correct

## Model Structure

Trained on 51200 images of Japanese characters (ETL8BC1)

160 occurences of 320 characters (64x63)

The model is a Convolutional Neural Network (CNN) with the following structure:

1. Convolutional Layer 1:
   - Input channels: 1 (grayscale)
   - Output channels: 32
   - Kernel size: 3x3
   - Stride: 1
   - Activation: ReLU

2. Convolutional Layer 2:
   - Input channels: 32
   - Output channels: 64
   - Kernel size: 3x3
   - Stride: 1
   - Activation: ReLU

3. Max Pooling Layer:
   - Kernel size: 2x2

4. Dropout Layer 1:
   - Dropout rate: 0.25

5. Flatten Layer

6. Fully Connected Layer 1:
   - Input: Dynamically sized based on the flattened output
   - Output: 128 neurons
   - Activation: ReLU

7. Dropout Layer 2:
   - Dropout rate: 0.5

8. Fully Connected Layer 2 (Output Layer):
   - Input: 128
   - Output: Number of classes (varies based on the dataset)

9. Log Softmax activation for output probabilities

The model is trained using:
- Loss function: Negative Log Likelihood Loss (NLL Loss)
- Optimizer: Adam
- Batch size: 64
- Number of epochs: 50

The dataset used is the ETL8B dataset, which contains Japanese characters. The data is split into 80% training and 20% testing sets.

Output: Predicted Japanese character (from the set of characters in the ETL8B dataset)

## Analysis

It does "okay". Training at 50 epochs yields about 80% accuracy, with loss ending at around 1.0.

This can be circumvented by using more data (which we have, we're only using ETL8BC1)
Or by advancing the model to a more complex one (maybe a transformer?)

But completely fails on anything that's not in the training set. LOL so it's useless right now.

## Want to help?

I'm a busy college student, if you want to help out, feel free to directly contribute via issues or even better, make a PR. Email me at [kadenbilyeu@proton.me](mailto:kadenbilyeu@proton.me) if you have any questions.
