
from collections import Counter

import pandas as pd
import numpy as np

import os
import cv2
import string
from PIL import Image
from collections import Counter

import matplotlib.pyplot as plt #for graphs
from keras import layers #for building layers of neural net
from keras.models import Model
from keras import callbacks #for training logs, saving to disk periodically 
from sklearn.utils import shuffle

def count_characters_in_filenames(directory_path):
    """
    Function that takes as input the path to a directory and returns a DataFrame
    containing the number of occurrences of each character (letters and digits) in
    the names of the .png files in the directory.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing the .png files

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the number of occurrences of each character
    """
    counter = Counter()
    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            without_extension = os.path.splitext(filename)[0]
            counter.update(without_extension)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    counter = {character: counter.get(character, 0) for character in alphabet}
    df = pd.DataFrame.from_dict(counter, orient="index")
    return df
            

def preprocess(file_path, size_image, ncharacter_per_image, ncharacter_total):
    """
    Function that loads all the images in a folder and preprocesses them
    to be used in the model. The images are scaled between 0 and 1 and reshaped
    to the size specified in size_image. The labels are also transformed to a
    one-hot vector.

    Parameters
    ----------
    file_path : str
        Path to the folder containing the .png files
    size_image : tuple
        Size of the images as (height, width, channels)
    ncharacter_per_image : int
        Number of characters in each image
    ncharacter_total : int
        Number of possible characters in the captcha

    Returns
    -------
    X : numpy array
        Array containing all the images scaled and reshaped
    y : numpy array
        Array containing the one-hot vector of the labels
    filenames : list
        List of the filenames of the images
    """
    # Initialize arrays to store the preprocessed images and labels
    X = np.zeros((len(os.listdir(file_path)), *size_image))
    y = np.zeros((ncharacter_per_image, len(os.listdir(file_path)), ncharacter_total))
    filenames = []
    character = string.ascii_uppercase + "0123456789"
    # Iterate over the images in the folder
    for i, filename in enumerate(os.listdir(file_path)):
        # Read the image and convert it to grayscale
        img = cv2.imread(os.path.join(file_path, filename), cv2.IMREAD_GRAYSCALE)

        # Scale the image to be between 0 and 1 and reshape it
        img = img / 255.0
        img = np.reshape(img, size_image)

        # Get the label of the image by removing the .png extension
        label = filename[:-4]

        # Check that the length of the label is not more than the number of characters per image
        if len(label) <= ncharacter_per_image:
            # Create a one-hot vector for the label
            target = np.zeros((ncharacter_per_image, ncharacter_total))
            for j, k in enumerate(label):
                index = character.find(k)
                target[j, index] = 1

            # Store the preprocessed image and label
            X[i] = img
            y[:, i] = target
            filenames.append(filename)

    return X, y, filenames


def create_model(size_image, n_characters_per_image, n_characters_total):
    """
    Function to create a model for recognizing characters in an image.

    Parameters
    ----------
    size_image : tuple
        Size of the image as (height, width, channels)
    n_characters_per_image : int
        Number of characters in each image
    n_characters_total : int
        Number of possible characters in the captcha

    Returns
    -------
    Model
        Compiled model
    """
    # Input layer
    image_input = layers.Input(shape=size_image)

    # Convolutional layers
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(image_input)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3)
    mp3 = layers.MaxPooling2D(padding='same')(bn)

    # Flatten layer
    flat = layers.Flatten()(mp3)

    # Dense layers
    outs = []
    for _ in range(n_characters_per_image):
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1)
        output = layers.Dense(n_characters_total, activation='sigmoid')(drop)
        outs.append(output)

    # Compile model
    model = Model(image_input, outs)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', 'accuracy', 'accuracy', 'accuracy', 'accuracy', 'accuracy'] )#* n_characters_per_image

    return model

