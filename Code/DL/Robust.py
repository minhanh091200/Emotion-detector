import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Utilities import CustomDataset
from Variant2 import V2MultiLayerFCNet

# Define the class names (if not already defined)
classNames = ['Engaged', 'Happy', 'Neutral', 'Surprised']
# Get the absolute path of the current working directory
current_directory = os.getcwd()
dataPath = os.path.abspath(os.path.join(current_directory, 'Data'))
extraPath = os.path.abspath(os.path.join(current_directory, 'Extra Data'))

# Hyperparameters and Settings
batch_size = 64
output_size = 4
percentExtra = [0.3, 0.6, 1] # 30% extra is 15% more females in the total dataset and so on
learning_rate = 0.001
epochs = 20

# Main
if __name__ == '__main__':

    dataset = CustomDataset(dataPath, classNames)
    extraDataset = CustomDataset(extraPath, classNames)
    # Get the parent directory of the Data directory
    parent_directory = os.path.dirname(dataPath)
    
    # Define the save directory path
    save_dir = os.path.join(parent_directory, 'SavedModels')
    os.makedirs(save_dir, exist_ok=True)

    trainSet, tempSet = train_test_split(dataset, test_size=0.3, random_state=42)
    valSet, testSet = train_test_split(tempSet, test_size=0.5, random_state=42)

    valLoader = DataLoader(valSet, batch_size=batch_size, shuffle=True, pin_memory=True)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, pin_memory=True)

    for percent in percentExtra:
        model = V2MultiLayerFCNet(output_size)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        bestAccuracy = 0
        patience = 5
        extraSet = []
    
        if (percent == 1):
            extraSet = list(extraDataset)
        else:
            extraSet, tempSet = train_test_split(extraDataset, train_size=percent, random_state=42)

        trainLoader = DataLoader(trainSet + extraSet, batch_size=batch_size, shuffle=True, pin_memory=True)
        
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
                    torch.save(model.state_dict(), os.path.join(save_dir, f'V2_F_Bias_{int((percent * 100) / 2)}.pth'))

        # After training loop
        loadedModel = V2MultiLayerFCNet(output_size)
        loadedModel.load_state_dict(torch.load(os.path.join(save_dir, f'V2_F_Bias_{int((percent * 100) / 2)}.pth')))

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
        print("Accuracy:", accuracy, "\n")