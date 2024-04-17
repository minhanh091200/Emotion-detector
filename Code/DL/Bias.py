import os
import torch
from torch.utils.data import DataLoader
from Utilities import CustomDataset
from Variant2 import V2MultiLayerFCNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Definitions
classNames = ['Engaged', 'Happy', 'Neutral', 'Surprised']
output_size = 4
current_directory = os.getcwd()
dataPath = os.path.abspath(os.path.join(current_directory, 'Data'))
parent_directory = os.path.dirname(dataPath)
biasedDataPath = os.path.join(parent_directory, 'Biased Data')
batch_size = 64

def evaluate_dataset(dataset_loader, loadedModel):
    # Evaluate the model on the entire dataset
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in dataset_loader:
            outputs = loadedModel(images)
            predicted = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.numpy())
            pred_labels.extend(predicted.numpy())

    # Calculate precision, recall, and F1 score
    precision_macro = precision_score(true_labels, pred_labels, average='macro')
    recall_macro = recall_score(true_labels, pred_labels, average='macro')
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    accuracy = accuracy_score(true_labels, pred_labels)

    return precision_macro, recall_macro, f1_macro, accuracy

def detect_age_bias(model):
    # Evaluate age groups bias
    age_accuracy = []
    age_precision = []
    age_recall = []
    age_f1 = []

    for age in ageGroups:
        dataset = CustomDataset(os.path.join(biasedDataPath, 'Age', age), classNames)
        testLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        prec, recall, f1, acc = evaluate_dataset(testLoader, model)
        print(f"\nEvaluation metrics for '{age}':")
        print("\nMacro-average precision:", prec)
        print("Macro-average recall:", recall)
        print("Macro-average F1 score:", f1)
        print("Accuracy:", acc)

        age_accuracy.append(acc)
        age_precision.append(prec)
        age_recall.append(recall)
        age_f1.append(f1)
    
    avg_age_accuracy = sum(age_accuracy) / len(age_accuracy)
    avg_age_precision = sum(age_precision) / len(age_precision)
    avg_age_recall = sum(age_recall) / len(age_recall)
    avg_age_f1 = sum(age_f1) / len(age_f1)

    print("\nAverage Metrics Across All Ages:")
    print("\nAverage Precision (Macro):", avg_age_precision)
    print("Average Recall (Macro):", avg_age_recall)
    print("Average F1 Score (Macro):", avg_age_f1)
    print("Average Accuracy:", avg_age_accuracy)

def detect_gender_bias(model):
    # Evaluate gender group bias
    gender_accuracy = []
    gender_precision = []
    gender_recall = []
    gender_f1 = []

    for gender in genderGroups:
        dataset = CustomDataset(os.path.join(biasedDataPath, 'Gender', gender), classNames)
        testLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        prec, recall, f1, acc = evaluate_dataset(testLoader, model)
        print(f"\nEvaluation metrics for '{gender}':")
        print("\nMacro-average precision:", prec)
        print("Macro-average recall:", recall)
        print("Macro-average F1 score:", f1)
        print("Accuracy:", acc)

        gender_accuracy.append(acc)
        gender_precision.append(prec)
        gender_recall.append(recall)
        gender_f1.append(f1)
    
    avg_gender_accuracy = sum(gender_accuracy) / len(gender_accuracy)
    avg_gender_precision = sum(gender_precision) / len(gender_precision)
    avg_gender_recall = sum(gender_recall) / len(gender_recall)
    avg_gender_f1 = sum(gender_f1) / len(gender_f1)

    print("\nAverage Metrics Across All Genders:")
    print("\nAverage Precision (Macro):", avg_gender_precision)
    print("Average Recall (Macro):", avg_gender_recall)
    print("Average F1 Score (Macro):", avg_gender_f1)
    print("Average Accuracy:", avg_gender_accuracy, "\n")

def evaluate_entire_dataset(model):
    # Evaluate model on entire dataset
    dataset = CustomDataset(dataPath, classNames)
    testLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    prec, recall, f1, acc = evaluate_dataset(testLoader, model)
    print("\nEvaluation metrics for the entire dataset:")
    print("\nMacro-average precision:", prec)
    print("Macro-average recall:", recall)
    print("Macro-average F1 score:", f1)
    print("Accuracy:", acc)

if __name__ == '__main__':
    ageGroups = ['Adult', 'Baby', 'Children', 'Senior']
    genderGroups = ['Female', 'Male']

    modelName = input("\nEnter the exact name of the model you want to load without the extension: ")

    model = V2MultiLayerFCNet(output_size)

    model.load_state_dict(torch.load(os.path.join(parent_directory, 'SavedModels/' + modelName + '.pth')))

    evaluationCategory = input("Do you want to evaluate age or gender or both?: ")

    if evaluationCategory == "age" or evaluationCategory == "both":
        detect_age_bias(model)
    elif evaluationCategory == "gender" or evaluationCategory == "both":
        detect_gender_bias(model)
        
    shouldEvaluateEntireSet = input("Do you want to evaluate the entire dataset using this model? (yes/no): ")

    if shouldEvaluateEntireSet == "yes":
        evaluate_entire_dataset(model)