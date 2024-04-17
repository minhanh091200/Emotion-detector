# This script will analyze the data and output some visualizations

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from PIL import Image
import numpy as np

# Directory where the data is stored
data_directory = '../../Data' # IMPORTANT: Need to make sure this is the path to the dataset folder
class_names = ['Engaged', 'Happy', 'Neutral', 'Surprised']

# Create new directory for visualizations if it doesn't exist
output_dir = "../../Visualization" # IMPORTANT: Need to make sure where you want to save the output visualizations
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Class distribution graph
class_counts = {class_name: len(os.listdir(os.path.join(data_directory, class_name))) for class_name in class_names}
plt.figure(figsize=(10, 5))
plt.bar(*zip(*class_counts.items()))
plt.title('Number of Images in Each Class')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.savefig(os.path.join(output_dir, 'Class_distribution.png'))

# Randomly choose 25 pictures from each class
random.seed(10)
num_images_per_class = 25
random_images = {}

for class_name in class_names:
    all_images = os.listdir(os.path.join(data_directory, class_name))
    random_images[class_name] = random.sample(all_images, num_images_per_class)


# Generate 5 x 5 grid from randomly chosen sample images
grid = 5

for class_name in class_names:
    fig, axes = plt.subplots(grid, grid, figsize=(10, 10))
    fig.suptitle(f'Random Images from Class: {class_name}')
    images = random_images[class_name]
    for i, image_name in enumerate(images):
        img = Image.open(os.path.join(data_directory, class_name, image_name))
        # Get the row and column indices of subplot in the grid 
        ax = axes[i // grid, i % grid]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.savefig(os.path.join(output_dir, f'{class_name}_random_images.png'))


# Pixel intensity histogram for each class based on the randomly chosen samples
for class_name in class_names:
    pixel_values = []
    images = random_images[class_name]
    for image_name in images:
        img = Image.open(os.path.join(data_directory, class_name, image_name))
        img_array = np.array(img)
        pixel_values.extend(img_array.flatten())
    plt.figure(figsize=(10, 5))
    plt.hist(pixel_values, bins=256)
    plt.title('Distribution of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{class_name}_pixel_intensity_histogram.png'))

