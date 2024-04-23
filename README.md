# Deep Image Classifier

This project implements a deep learning-based image classifier using Convolutional Neural Networks (CNNs) with the CIFAR-10 dataset. It utilizes the Keras library, which is part of TensorFlow, for model building and training.

## Overview

The image classifier is trained to classify images from the CIFAR-10 dataset into one of 10 classes: 'Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', and 'Truck'. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Features

- Loads the CIFAR-10 dataset using Keras.datasets.
- Preprocesses the image data by normalizing pixel values to the range [0, 1].
- Visualizes sample images from the dataset with their corresponding class labels.
- Builds a CNN model using Keras.Sequential with multiple convolutional and pooling layers.
- Compiles the model with the Adam optimizer and sparse categorical cross-entropy loss.
- Trains the model on the training data and evaluates its performance on the testing data.
- Saves the trained model for future use.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/deep-image-classifier.git

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

3. Run main.py

   ```
   python main.py

