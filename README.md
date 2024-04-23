# Deep Image Classifier

This project implements a deep learning-based image classifier using Convolutional Neural Networks with the CIFAR-10 dataset. It uses the Keras library from TensorFlow for the model's building and training. The model has been trained to classify images into its corresponding category. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

<p align="center">
  <img src="https://github.com/sakeefh/Deep-Image-Classifier/assets/91638600/d992ba97-adbb-473b-8090-b08f90f2c3a8" alt="ImageClassifierv1" width="400" height="400">
</p>

<p align="center">Sample Images from the CIFAR-10 Dataset</p>


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
   git clone https://github.com/sakeefh/deep-image-classifier.git

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

3. Run main.py

   ```
   python main.py
