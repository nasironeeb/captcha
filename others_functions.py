# -*- coding: utf-8 -*-

"""
Created on Tue Jul  6 11:26:17 2022

@author: Oneeb Nasir
""" 

import os
from PIL import Image
import imageio
import re

def convert_avif_to_png(input_folder, output_folder):
    """
    Convert all .avif files in the input folder to .png files in the output folder.
    
    Args:
        input_folder (str): The folder containing the .avif files.
        output_folder (str): The folder where the .png files will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an .avif file
        #print(filename)
        if filename.endswith('.avif'):
            ## modifiy the name of the file
            # Extract the number from the filename
            number = re.search(r'\d+', filename).group()
            # Build the new filename
            filename_modified = "img_" + number + ".png" 
            # Construct the full file path
            avif_path = os.path.join(input_folder, filename)
            # Construct the full file path
            avif_path = os.path.join(input_folder, filename)
            # Load the .avif image
            avif_image = imageio.imread(avif_path)
            # Convert to a Pillow Image object
            pil_image = Image.fromarray(avif_image)
            # Create the output file path
            png_filename = filename_modified
            png_path = os.path.join(output_folder, png_filename)
            # Save the image as .png
            pil_image.save(png_path)