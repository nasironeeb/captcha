import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
import matplotlib.pyplot as plt

class CaptchaDataset(Dataset):
    def __init__(self, file_path, size_image, ncharacter_per_image, ncharacter_total):
        self.file_path = file_path
        self.size_image = size_image
        self.ncharacter_per_image = ncharacter_per_image
        self.ncharacter_total = ncharacter_total
        self.character = string.ascii_uppercase + "0123456789"
        self.filenames = os.listdir(file_path)

    def __len__(self):
        
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = cv2.imread(os.path.join(self.file_path, filename), cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Adding channel dimension

        label = filename[:-4]
        target = torch.zeros((self.ncharacter_per_image, self.ncharacter_total), dtype=torch.float32)
        for j, char in enumerate(label):
            index = self.character.find(char)
            if index != -1:
                target[j, index] = 1
        return img, target
    
class CaptchaModel(nn.Module):
    def __init__(self, size_image, n_characters_per_image, n_characters_total, dropout_val):
        """
        Initializes the CaptchaModel with the specified image size, number of characters per image,
        and number of characters total. This is a convolutional neural network with three convolutional
        layers followed by three fully connected layers. The output of the network is a tensor of size
        (n_characters_per_image, n_characters_total).

        Parameters
        ----------
        size_image : tuple
            Size of the images as (height, width, channels)
        n_characters_per_image : int
            Number of characters in each image
        n_characters_total : int
            Number of possible characters in the captcha
        dropout_val : float
            Dropout value
        """
        super(CaptchaModel, self).__init__()

        # Convolutional layers
        # The first convolutional layer takes the input image, which is a 3D tensor with shape
        # (height, width, channels), and outputs a 3D tensor with shape (height, width, 16)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # The second convolutional layer takes the output of the first convolutional layer and
        # outputs a 3D tensor with shape (height, width, 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # The third convolutional layer takes the output of the second convolutional layer and
        # outputs a 3D tensor with shape (height, width, 32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Max pooling layer
        # The max pooling layer takes the output of the third convolutional layer and downsamples
        # it by a factor of 2, taking the maximum value in each 2x2 window
        self.pool = nn.MaxPool2d(2, 2)

        # Batch normalization layer
        # The batch normalization layer normalizes the output of the convolutional layers
        self.bn = nn.BatchNorm2d(32)

        # Flatten layer
        # The flatten layer takes the output of the convolutional layers and flattens it into a
        # 1D tensor of size (height * width * channels)
        self.flatten_size = 32 * (size_image[0] // 8) * (size_image[1] // 8)

        # Fully connected layers
        # The fully connected layers take the output of the flatten layer and output a tensor of
        # size (n_characters_per_image, n_characters_total)
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.flatten_size, 64),
                nn.ReLU(),  # Activation function
                nn.Dropout(dropout_val),  # Dropout layer with probability dropout_val
                nn.Linear(64, n_characters_total),
                nn.Sigmoid()  # Activation function
            ) for _ in range(n_characters_per_image)
        ])

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.bn(x)
        x = x.view(-1, self.flatten_size)
        
        outputs = [fc(x) for fc in self.fc_layers]
        return outputs
    
    
def predict_1(model, images):
    """
    Makes predictions on a list of images

    Args:
        model (nn.Module): trained captcha model
        images (list of PIL images): list of images to make predictions on

    Returns:
        list of str: list of strings, one for each image in the input list
    """
    # Define the characters that can appear in the captcha
    characters = string.ascii_uppercase + string.digits

    # Set the model to evaluation mode
    model.eval()

    # Make predictions on the input images
    predictions = model(images.unsqueeze(0))

    # Convert the predictions to strings
    to_return = [characters[torch.argmax(p, dim=1).item()] for p in predictions]
    return to_return
