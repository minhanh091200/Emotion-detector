import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim, nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Utilities import CustomDataset, displayConfusionMatrix, visualizeResults
import os

# Get the absolute path of the current working directory
current_directory = os.getcwd()

# Hyperparameters and Settings
dataPath = os.path.abspath(os.path.join(current_directory, 'Data'))
classNames = ['Engaged', 'Happy', 'Neutral', 'Surprised']
num_classes = len(classNames)
epochs = 20
learning_rate = 0.001
batch_size = 64
input_size = 1 * 48 * 48  # 1 channels, 48x48 image size
output_size = 4
device = torch.device("cpu")

# Variant 1 (More Layers) model to train the data
class V1MultiLayerFCNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.layer1 = nn.Conv2d(1, 32, 3, padding=1, stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool = nn.MaxPool2d(2) # Maxpool is used to aggregate the image
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)

        # New convolutional layers
        self.layer5 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.B5 = nn.BatchNorm2d(128)
        self.layer6 = nn.Conv2d(128, 128, 3, padding=1, stride=1)
        self.B6 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128 * 6 * 6, output_size) 

    def forward(self, x):
        x = self.B1(F.leaky_relu(self.layer1(x)))
        x = self.B2(self.Maxpool(F.leaky_relu(self.layer2(x))))
        x = self.B3(F.leaky_relu(self.layer3(x)))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))

        # Additional convolutional layers
        x = self.B5(F.leaky_relu(self.layer5(x)))
        x = self.B6(self.Maxpool(F.leaky_relu(self.layer6(x))))

        return self.fc(x.view(x.size(0), -1))

# Main
if __name__ == '__main__':

    dataset = CustomDataset(dataPath, classNames)
    # Get the parent directory of the Data directory
    parent_directory = os.path.dirname(dataPath)
    
    # Define the save directory path
    save_dir = os.path.join(parent_directory, 'SavedModels')
    os.makedirs(save_dir, exist_ok=True)

    trainSet, tempSet = train_test_split(dataset, test_size=0.3, random_state=42)
    valSet, testSet = train_test_split(tempSet, test_size=0.5, random_state=42)

    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, pin_memory=True)
    valLoader = DataLoader(valSet, batch_size=batch_size, shuffle=True, pin_memory=True)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = V1MultiLayerFCNet(output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    bestAccuracy = 0
    patience = 5

    model.train()

    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainLoader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        printLoss = running_loss / len(trainLoader)

        # Train accuracy
        model.eval()
        with torch.no_grad():
            valSamps = 0
            correct = 0

            for images, labels in valLoader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                valSamps = outputs.size(0)
                correct = (predicted == labels).sum().item()

            ACC = correct / valSamps

            print('Loss =', printLoss, '| Accuracy =', ACC * 100)

            # Early stopping condition
            if ACC <= bestAccuracy:
                patience -= 1
                if patience <= 0 and epoch > 10:
                    print("\nEarly stopping as accuracy didn't improve")
                    break
            else:
                patience = 5  # Reset patience
                bestAccuracy = ACC
                torch.save(model.state_dict(), os.path.join(save_dir, 'Variant1.pth'))

    # After training loop
    loadedModel = V1MultiLayerFCNet(output_size)
    loadedModel.load_state_dict(torch.load(os.path.join(save_dir, 'Variant1.pth')))

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in testLoader:
            outputs = loadedModel(images)
            predicted = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.numpy())
            pred_labels.extend(predicted.numpy())

    # Calculate precision, recall, and F1 score
    precision_macro = precision_score(true_labels, pred_labels, average='macro')
    recall_macro = recall_score(true_labels, pred_labels, average='macro')
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    precision_micro = precision_score(true_labels, pred_labels, average='micro')
    recall_micro = recall_score(true_labels, pred_labels, average='micro')
    f1_micro = f1_score(true_labels, pred_labels, average='micro')
    accuracy = accuracy_score(true_labels, pred_labels)

    print("\nMacro-average precision:", precision_macro)
    print("Macro-average recall:", recall_macro)
    print("Macro-average F1 score:", f1_macro)
    print("Micro-average precision:", precision_micro)
    print("Micro-average recall:", recall_micro)
    print("Micro-average F1 score:", f1_micro)
    print("Accuracy:", accuracy)

    # Display confusion matrix
    print("\nConfsion matrix was displayed. Please close the window to continue")
    displayConfusionMatrix(true_labels, pred_labels, classNames)

    # Display image with their prediction
    print("\nPrediction results were visualized. Please close the window to continue\n")
    visualizeResults(model, testLoader, classNames)