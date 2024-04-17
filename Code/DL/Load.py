import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from Utilities import CustomDataset, denormalize, displayConfusionMatrix

from Main import MainMultiLayerFCNet
from Variant1 import V1MultiLayerFCNet
from Variant2 import V2MultiLayerFCNet

# Define the class names (if not already defined)
classNames = ['Engaged', 'Happy', 'Neutral', 'Surprised']
# Get the absolute path of the current working directory
current_directory = os.getcwd()
dataPath = os.path.abspath(os.path.join(current_directory, 'Data'))
batch_size = 64

# Define the transformation to be applied to the input image
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.225])
])

def load_model(modelName):
    model = None

    # Get the parent directory of the Data directory
    parent_directory = os.path.dirname(dataPath)
    # Define the save directory path
    save_dir = os.path.join(parent_directory, 'SavedModels')

    os.makedirs(save_dir, exist_ok=True)
    if (modelName == 'main'):
        model = MainMultiLayerFCNet(output_size=len(classNames))
        model.load_state_dict(torch.load(os.path.join(save_dir, 'Main.pth'), map_location=torch.device('cpu')))
    elif (modelName == 'v1'):
        model = V1MultiLayerFCNet(output_size=len(classNames))
        model.load_state_dict(torch.load(os.path.join(save_dir, 'Variant1.pth'), map_location=torch.device('cpu')))
    elif (modelName == 'v2'):
        model = V2MultiLayerFCNet(output_size=len(classNames))
        model.load_state_dict(torch.load(os.path.join(save_dir, 'Variant2.pth'), map_location=torch.device('cpu')))
    else:
        model = V2MultiLayerFCNet(output_size=len(classNames))
        model.load_state_dict(torch.load(os.path.join(save_dir, modelName + '.pth'), map_location=torch.device('cpu')))

    model.eval()
    return model

def predict_image(model, image_path, true_label):
    # Open the image file
    img = Image.open(image_path).convert('L')
    # Apply the transformation
    img = transform(img)
    # Add batch dimension
    img = img.unsqueeze(0)
    # Perform prediction
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    # Get the predicted class label
    predicted_label = classNames[predicted.item()]
    
    # Denormalize image
    img = denormalize(img.squeeze(), mean=[0.485, ], std=[0.225, ])
    img = img.numpy() # Convert image from tensor to numpy array for plotting

    # Display the image, true label, and predicted label
    plt.imshow(img, cmap='gray')  # Specify cmap='gray' for grayscale images
    plt.title(f"true='{true_label}', pred='{predicted_label}'", fontsize=8)
    plt.axis('off')
    plt.show()

def evaluate_dataset(model, dataset_loader):
    correct = 0
    total = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in dataset_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_labels.extend(labels.numpy())
            pred_labels.extend(predicted.numpy())

    accuracy = correct / total
    return accuracy, true_labels, pred_labels

if __name__ == '__main__':
    # Ask which model to test
    modelName = input("\nEnter which model you want to test (main or v1 or v2 or all or exactly the name of the model w/o '.pth'): ")

    # Load the model
    if modelName == 'all':
        mdlName = ['main', 'v1', 'v2']
        model = []
        model.append(load_model('main'))
        model.append(load_model('v1'))
        model.append(load_model('v2'))
    else:
        model = load_model(modelName)

    evaluation_type = input("Enter 'dataset' to evaluate a dataset or 'image' to evaluate an individual image: ")

    if evaluation_type.lower() == 'dataset':
        dataset = CustomDataset(dataPath, classNames)
        testLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        if modelName == 'all':
            for idx, mdl in enumerate(model):
                complete_accuracy, trueLabels, predLabels = evaluate_dataset(mdl, testLoader)
                print(f"\nAccuracy on complete dataset for '{mdlName[idx]}' model:", complete_accuracy)
                displayConfusionMatrix(trueLabels, predLabels, classNames)
        else:
            complete_accuracy, trueLabels, predLabels = evaluate_dataset(model, testLoader)
            print(f"\nAccuracy on complete dataset for '{modelName}' model:", complete_accuracy)
            displayConfusionMatrix(trueLabels, predLabels, classNames)
    elif evaluation_type.lower() == 'image':
        image_name = input("Enter the image name: ")
        class_name, image_number = image_name.split('_')
        image_path = os.path.join(dataPath, class_name, f"{image_name}.png")
        if modelName == 'all':
            for idx, mdl in enumerate(model):
                print(f'\nPlease close the display window for {mdlName[idx]} model to continue')
                predict_image(mdl, image_path, class_name)
        else:
            print(f'\nPlease close the display window for {modelName} model to continue')
            predict_image(model, image_path, class_name)
