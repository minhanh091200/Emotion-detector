import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim, nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Utilities import CustomDataset
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

# Variant 2 (Increased kernel size) model to train the data
class V2MultiLayerFCNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.layer1 = nn.Conv2d(1, 32, 5, padding=2, stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 5, padding=2, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool = nn.MaxPool2d(2) # Maxpool is used to aggregate the image
        self.layer3 = nn.Conv2d(32, 64, 5, padding=2, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 5, padding=2, stride=1)
        self.B4 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 12 * 12, output_size) # output_size is the number of classes

    def forward(self, x):
        x = self.B1(F.leaky_relu(self.layer1(x)))
        x = self.Maxpool(F.leaky_relu(self.layer2(x)))
        x = self.B2(x)
        x = self.B3(F.leaky_relu(self.layer3(x)))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))

        return self.fc(x.view(x.size(0), -1))

# Main
if __name__ == '__main__':

    dataset = CustomDataset(dataPath, classNames)
    # Get the parent directory of the Data directory
    parent_directory = os.path.dirname(dataPath)
    
    # Define the save directory path
    save_dir = os.path.join(parent_directory, 'SavedModels')
    os.makedirs(save_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    accuracy_list = []
    precision_macro_list = []
    recall_macro_list = []
    f1_macro_list = []
    precision_micro_list = []
    recall_micro_list = []
    f1_micro_list = []

    for fold, (train_index, test_index) in enumerate(skf.split(dataset.image_data, dataset.label_data), 1):
        trainSet = torch.utils.data.Subset(dataset, train_index)
        testSet = torch.utils.data.Subset(dataset, test_index)

        trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, pin_memory=True)
        testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False, pin_memory=True)

        model = V2MultiLayerFCNet(output_size)
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

                for images, labels in testLoader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    valSamps = outputs.size(0)
                    correct += (predicted == labels).sum().item()

                ACC = correct / len(testLoader.dataset)

                print(f"Fold [{fold}/{skf.get_n_splits()}] | Epoch [{epoch + 1}/{epochs}] | Loss = {printLoss} | Accuracy = {ACC * 100}")

                # Early stopping condition
                if ACC <= bestAccuracy:
                    patience -= 1
                    if patience <= 0:
                        print("\nEarly stopping as accuracy didn't improve")
                        break
                else:
                    patience = 5  # Reset patience
                    bestAccuracy = ACC
                    torch.save(model.state_dict(), os.path.join(save_dir, f'Variant2_fold{fold}.pth'))

        # After training loop
        loadedModel = V2MultiLayerFCNet(output_size)
        loadedModel.load_state_dict(torch.load(os.path.join(save_dir, f'Variant2_fold{fold}.pth')), strict=False)

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

        accuracy_list.append(accuracy)
        precision_macro_list.append(precision_macro)
        recall_macro_list.append(recall_macro)
        f1_macro_list.append(f1_macro)
        precision_micro_list.append(precision_micro)
        recall_micro_list.append(recall_micro)
        f1_micro_list.append(f1_micro)

        print(f"\nMetrics for Fold {fold}:")
        print("Accuracy:", accuracy)
        print("Precision (Macro):", precision_macro)
        print("Recall (Macro):", recall_macro)
        print("F1 Score (Macro):", f1_macro)
        print("Precision (Micro):", precision_micro)
        print("Recall (Micro):", recall_micro)
        print("F1 Score (Micro):", f1_micro)
        print("="*50)

    # Calculate average metrics across all folds
    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    avg_precision_macro = sum(precision_macro_list) / len(precision_macro_list)
    avg_recall_macro = sum(recall_macro_list) / len(recall_macro_list)
    avg_f1_macro = sum(f1_macro_list) / len(f1_macro_list)
    avg_precision_micro = sum(precision_micro_list) / len(precision_micro_list)
    avg_recall_micro = sum(recall_micro_list) / len(recall_micro_list)
    avg_f1_micro = sum(f1_micro_list) / len(f1_micro_list)

    print("\nAverage Metrics Across All Folds:")
    print("Average Accuracy:", avg_accuracy)
    print("Average Precision (Macro):", avg_precision_macro)
    print("Average Recall (Macro):", avg_recall_macro)
    print("Average F1 Score (Macro):", avg_f1_macro)
    print("Average Precision (Micro):", avg_precision_micro)
    print("Average Recall (Micro):", avg_recall_micro)
    print("Average F1 Score (Micro):", avg_f1_micro)