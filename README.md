# Facial Expression Classification Project

## Overview

This repository is dedicated to the Facial Expression Classification project completed as part of the COMP-472 course. The objective of this project is to develop a model capable of accurately classifying facial expressions into four categories: neutral, engaged/focused/concentrated, surprised, and happy.

## Dataset

The dataset used for this project consists of facial images, with each expression class having 400 images for training and 100 images for testing. The dataset is carefully curated to ensure a balanced representation of each facial expression.

## Project Structure

### Code/Scripts:

Contains the python scripts used for renaming the images according to their class and for generating the data visualization files:

**DataVisual.py:** Python script that generates important data visualization charts using Matplotlib. First it iterates over the entire dataset to construct a breakdown based on class. then it randomly selects 25 images from each class and generates a 5 x 5 grid with them. Finally it generates a pixel intensity graph for each class based on the latter's 25 images sample acquired before.
**To run:** Open the Scripts folder in an anaconda environment with python 3.6 and type in python DataVisual.py.

**ImageRenamer.py:** Python script that iterates over the entire dataset and adds each file's class name to it's proper to ensure easier readability and data management.
**To run:** Open the Scripts folder in an anaconda environment with python 3.6 and type in python ImageRenamer.py.

### Code/DL:

**NOTE: AS WE ARE USING OS TO CONFIGURE DATA PATHS, IF YOU ARE USING TERMINAL TO RUN THE PROGRAMS, YOU MIGHT NEED TO RUN IT FROM THE ROOT FOLDER.**

**Main.py:** Python program to train and evaluate the main model. The CNN's main architecture is defined, then the Dataset is loaded and split into 3 parts: training (trainset), validation (predset) and evaluation (testset). After that the data is run through the model, with the utilization of early stopping techniques to prevent overfitting and save resources. Finally the best performing model is saved and the evaluation results are displayed.
**To run:** Open the DL folder in an anaconda environment with python 3.8 and run python Main.py.

**Variant1.py:** Python program to train and evaluate the first variant of the model, which introduces 2 more convolutional layers. The CNN's main architecture is defined, then the Dataset is loaded and split into 3 parts: training (trainset), validation (predset) and evaluation (testset). After that the data is run through the model, with the utilization of early stopping techniques to prevent overfitting and save resources. Finally the best performing model is saved and the evaluation results are displayed.
**To run:** Open the DL folder in an anaconda environment with python 3.8 and run python Variant1.py.

**Variant2.py:** Python program to train and evaluate the second variant of the model, this time with a larger kernel size instead. The CNN's main architecture is defined, then the Dataset is loaded and split into 3 parts: training (trainset), validation (predset) and evaluation (testset). After that the data is run through the model, with the utilization of early stopping techniques to prevent overfitting and save resources. Finally the best performing model is saved and the evaluation results are displayed.
**To run:** Open the DL folder in an anaconda environment with python 3.8 and run python Variant2.py.

**Load.py:** Python program that can load and run one or all of the saved models. It can make the model(s) evaluate either a single image or the entire dataset. Additionally, you can enter the name of a model without the extension to be used to test the model. When choosing the entire dataset, a confusion matrix is generated at the end.
**To run:** Open the DL folder in an anaconda environment with python 3.8 and run python Load.py. You will be prompted to enter the name of the model you wish to test or enter 'all' if you want to test all the models. After that indicate whether you want to test the model on a the existing dataset or a single image (The image's name has to be provided. Please check the 'Data' folder for image names). Remember that 3rd party windows might open up. You need to close all these windows to continue with the code.

**Bias.py:** Python program that can load one of the saved models. It uses the biased data to detect biases in the trained model. You can check for age or gender bias or both. At the end, you can also check to evaluate the entire original dataset.
**To run:** Open the DL folder in an anaconda environment with python 3.8 and run python Bias.py. You will be prompted to enter the name of the model you wish to test. You need to enter the models name without the extension (.pth). After that, indicate whether you want to test the model with the gender or age or both categories. Finally, indicate yes/no to evaluate the model with orginal dataset.

**Robust.py:** Python program that uses the extra data to train 3 models with 3 levels of bias. The 3 levels are currently set to 15%, 30% and 50% more female data in addition to the original data.
**To run:** Open the DL folder in an anaconda environment with python 3.8 and run python Robust.py. As it trains the 3 models, the will see the loss, accuracy as well as other metrics.

**K-Fold_V2.py:** Python program that will use the Variant_2 model definition and train a new model using 10-fold cross validation.
**To run:** Open the DL folder in an anaconda environment with python 3.8 and run python K-Fold_V2.py. As it trains the 3 models, the will see the loss, accuracy as well as other metrics.

**Utilities.py:** Contains functions and classes that are being used by several files. The main export is the custom dataset class which imports and creates the dataset.

## Available Data:

### Data

Contains sample data acquired from the FER-2013 dataset separated into 4 classes:

- **Engaged:** Contains all the engaged images.

- **Happy:** Contains all the happy images.

- **Neutral:** Contains all the neutral images.

- **Surprised:** Contains all the surprised images.

### Extra Data

Contains 50% more extra female data for each class.

### Biased Data

Containes the original data categorized into different age and gender groups. Each group (e.g. Gender) and sub-group (e.g. Male/Female) contains the initial 4 classes (i.e. Engaged, Happy, Neutral and Surprised)

## Visualization:

Contains all the data visualization files generated by python scripts:

- **class_distribution:** break down of the dataset by class.

- **Engaged_random_images:** Sample Images for the Engaged class.

- **Engaged_pixel_intensity_histogram:** Pixel intensity for the Engaged sample images.

- **Happy_random_images:** Sample Images for the Happy class.

- **Happy_pixel_intensity_histogram:** Pixel intensity for the Happy sample images.

- **Neutral_random_images:** Sample Images for the Neutral class.

- **Neutral_pixel_intensity_histogram:** Pixel intensity for the Neutral sample images.

- **Surprised_random_images:** Sample Images for the Surprised class.

- **Surprised_pixel_intensity_histogram:** Pixel intensity for the Surprised sample images.

## Model

The model architecture and training process are detailed in the source code. The primary goal is to develop a robust model that accurately predicts the facial expression category for a given image.

## Reports

[Report](https://docs.google.com/document/d/11mVMNCOpyGs4PMVn1RaUgLxiiG_p1zrKOziUlxrH2SQ/edit?usp=sharing)

## Contributors

Ibrahim Daami | Minh Anh Dao | Tasnim Niloy
