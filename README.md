# Deep Image Classifier

This project implements a deep learning-based image classifier using Convolutional Neural Networks with the CIFAR-10 dataset. It uses the Keras library from TensorFlow for the model's building and training. The model has been trained to classify images into its corresponding category. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Technologies Used

- `Python`
- `TensorFlow`, `Keras`
- `NumPy`
- `Matplotlib`
- `OpenCV`
  
## Sample Images from the CIFAR-10 Dataset
<p align="center">
  <img src="https://github.com/sakeefh/Deep-Image-Classifier/assets/91638600/d992ba97-adbb-473b-8090-b08f90f2c3a8" alt="ImageClassifierv1" width="400" height="400">
</p>

## How it Works

1. The script loads the CIFAR-10 dataset using Keras.datasets.
2. The image data is then preprocessed by normalizing pixel values to the range [0, 1].
3. Sample images are first visualized from the dataset with their corresponding class labels.
4. The program then builds a CNN model using Keras.Sequential with multiple convolutional and pooling layers.
5. The model is trained on the training data and evaluates its performance on the testing data.
6. The trained model,`image_classifier.keras`, is saved for future use.

## Example Use Case
<img src="https://github.com/sakeefh/Deep-Image-Classifier/assets/91638600/5d1440ce-6a93-4def-9e09-0f9f9aa9a63f" alt="process" width="900">

## What I Learned

- The reason why images are resized to a fixed size such as 32x32 pixels is because Convolutional Neural Networks (CNNs) typically expect input images of the same size.
- Resizing images to a fixed size reduces the computational overhead, making training faster and more efficient.
- I learned how to train deep learning models using real-world datasets like CIFAR-10 and evaluate their performance using metrics like accuracy and loss.
- I learned about data preprocessing techniques and how to use them such as normalization and resizing.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/sakeefh/deep-image-classifier.git

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

3. Resize your picture to 32x32x pixels and update the following statement with the image file name

   ```bash
   img = cv.imread('XYZ.jpg')

3. Run main.py

   ```
   python main.py
