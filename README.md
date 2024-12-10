# Captcha

The purpose of this project was to stay sharp and train my skills in Python and Deep Learning.
This repository contains the implementation of a PyTorch-based deep learning model, `CaptchaModel`, designed for solving CAPTCHA tasks. The model predicts a sequence of characters (e.g., 6 characters per image) from grayscale CAPTCHA images of fixed dimensions.

## Model Architecture

The `CaptchaModel` is a convolutional neural network (CNN) combined with fully connected layers, built to extract features from images and predict multiple characters. Below is a detailed description of the architecture:

### Components

1. **Input Dimensions**:
    - Input: A single-channel (grayscale) image with size `(height, width)`.

2. **Convolutional Layers**:
    - **Conv1**: 16 filters, kernel size 3x3, stride 1, padding 1.
    - **Conv2**: 32 filters, kernel size 3x3, stride 1, padding 1.
    - **Conv3**: 32 filters, kernel size 3x3, stride 1, padding 1.

3. **Pooling**:
    - A `MaxPool2d` layer with a kernel size of 2x2 and stride 2 is applied after each convolutional layer to reduce the spatial dimensions by half.

4. **Batch Normalization**:
    - A `BatchNorm2d` layer is applied after the third convolutional layer to normalize feature maps.

5. **Flatten**:
    - The 2D output of the convolutional and pooling layers is flattened into a 1D vector for the fully connected layers.

6. **Fully Connected Layers**:
    - The model has a fully connected network for each character to be predicted:
        - First dense layer: 64 neurons with ReLU activation.
        - Dropout: 50% dropout to prevent overfitting.  (30% or 80% has been tried)
        - Output layer: Number of unique characters in the CAPTCHA (e.g., 36 for alphanumeric) with a Sigmoid activation function.

7. **Output**:
    - The model predicts a list of outputs, each representing the probability distribution over all possible characters for one position in the CAPTCHA.

### Forward Pass

1. The input image passes through three convolutional and pooling layers.
2. The output is flattened into a 1D vector.
3. For each character position in the CAPTCHA, the flattened vector is passed through a fully connected subnetwork to predict the character.

## Parameters

- **`size_image`**: Tuple specifying the dimensions of the input image `(height, width)`.
- **`n_characters_per_image`**: Number of characters to predict in each CAPTCHA.
- **`n_characters_total`**: Number of unique possible characters (e.g., 36 for [0-9-A-Z]).
- **`dropout_val`**: The dropout randomly selects a subset of neurons to be deactivated with a predefined probability.

## Dataset

 If you want to try it out, you can use the following link: https://www.kaggle.com/datasets/nasironeeb/captcha-dataset-6-character/data to get the data

the file model_full_1.pth is the model after training for 20 epochs with 0.001 learning rate and 0.8 dropout.
the file model_full_2.pth is the model after training for 20 epochs with 0.001 learning rate and 0.2 dropout.
the file model_full_3.pth is the model after training for 20 epochs with 0.001 learning rate and 0.5 dropout.
the file model_full_4.pth is the model after training for 40 epochs with 0.001 learning rate and 0.8 dropout.
The best model is the model_full_4.pth

