# metal_classification

## Overview
This project demonstrates how to perform image classification using TensorFlow and Keras. We classify images into four categories: brass, copper, steel, and others. The images are organized into folders based on their category, and we use a Convolutional Neural Network (CNN) for the classification task.

## Steps
## Data Preparation:
Organize images into folders named 'brass', 'copper', 'steel', and 'others', each containing respective images.
Images are preprocessed and resized to a standard size (e.g., 224x224 pixels) using TensorFlow's ImageDataGenerator.

## Model Building:
We build a CNN model using TensorFlow's Keras API.
The model consists of convolutional layers, max-pooling layers, flattening layer, and dense layers.
The final dense layer has 4 units with softmax activation function for multi-class classification.

## Model Compilation:
The model is compiled with the Adam optimizer and categorical cross-entropy loss function.
We use accuracy as the metric to monitor during training.

## Data Generators:
Custom data generators are defined to load and preprocess images on-the-fly in batches.
For training, we use a custom generator to load training images and their corresponding labels.
Similarly, a separate custom generator is used for validation.

## Model Training:
The model is trained using the fit() method from Keras.
We specify the number of steps per epoch based on the number of training samples and batch size.
Training is performed for a fixed number of epochs (e.g., 10).
## Model Evaluation:
After training, the model is evaluated on the validation set using the evaluate() method.
We calculate evaluation metrics such as loss and accuracy.

## Inference:
Once trained, the model can be used for inference to classify new images into the specified categories.
