import os
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Class that imports dataset and transforms the data
class CustomDataset(Dataset):
    def __init__(self, main_data_path, class_names):
        self.main_data_path = main_data_path
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(48, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, ], std=[0.225, ])
        ])
        self.image_data = []
        self.label_data = []

        for idx, class_name in enumerate(class_names):
            class_folder_path = os.path.join(main_data_path, class_name)
            if os.path.exists(class_folder_path):
                images = os.listdir(class_folder_path)
                for image in images:
                    image_path = os.path.join(class_folder_path, image)
                    img = Image.open(image_path).convert('L')
                    self.image_data.append(img)
                    self.label_data.append(idx)
            else:
                print(f"Class folder not found: {class_name}")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = self.label_data[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Function that takes in true labes, predicted labels and class names then displays a confusion matrix
def displayConfusionMatrix(trueLabels, predLabels, classNames):
    conf_matrix = confusion_matrix(trueLabels, predLabels)
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classNames).plot()
    plt.show()

# Function that denormalizes. It was taken from the lab assignment
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        return tensor

# Function that visualizes the predictions done by the CNN models. It was taken from the lab assignment
def visualizeResults(model, loader, classNames, x=2, y=5):
    # Width per image (inches)
    width_per_image = 2.4
    # Get a batch of images and labels
    data_iter = iter(loader)
    images, labels = next(data_iter)
    # Select x*y random images from the batch
    indices = np.random.choice(images.size(0), x*y, replace=False)
    random_images = images[indices]
    random_labels = labels[indices]
    outputs = model(random_images)
    _, predicted = torch.max(outputs.data, 1)
    # Fetch class names 
    classes = classNames
    fig, axes = plt.subplots(x, y, figsize=(y * width_per_image, x * width_per_image))
    # Iterate over the random images and
    # display them along with their predicted labels
    for i, ax in enumerate(axes.ravel()):
        # Denormalize image
        img = denormalize(random_images[i], mean=[0.485, ], std=[0.225, ])
        img = img.permute(1, 2, 0).numpy() # Convert image from CxHxW to HxWxC format for plotting
        true_label = classes[random_labels[i]]
        pred_label = classes[predicted[i]]
        ax.imshow(img)
        ax.set_title(f"true='{true_label}', pred='{pred_label}'", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()