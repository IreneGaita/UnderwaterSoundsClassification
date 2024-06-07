import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import csv
import torch.nn as nn


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.data = []
        self.targets = []

        for idx, cls in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, cls)
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                if os.path.isfile(file_path):
                    self.data.append(file_path)
                    self.targets.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target

    def get_num_classes(self):
        return len(self.classes)


def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def calculate_metrics(loader, model, device, num_classes):
    model.eval()
    all_labels = []
    all_preds = []
    pbar = tqdm(loader, desc="Calcolo metriche")
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy() if num_classes == 1 else torch.softmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.round())
            pbar.update()

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    if num_classes == 1:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
    else:
        accuracy = accuracy_score(all_labels, all_preds.argmax(axis=1))
        precision = precision_score(all_labels, all_preds.argmax(axis=1), average='weighted')
        recall = recall_score(all_labels, all_preds.argmax(axis=1), average='weighted')
        f1 = f1_score(all_labels, all_preds.argmax(axis=1), average='weighted')

    return accuracy, precision, recall, f1


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset_path = r'C:\Users\biagi\PycharmProjects\gruppo17\Testing_Norm'
    test_dataset = CustomDataset(test_dataset_path, transform=transform)

    num_classes = test_dataset.get_num_classes()
    if num_classes == 2:
        print("Classificazione binaria")
    else:
        print("Classificazione multiclasse")

    binary_classification = (num_classes == 2)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model = models.alexnet(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1 if binary_classification else num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint_dir = 'Checkpoints_Binary' if binary_classification else 'Checkpoints_Multiclass'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model_path = os.path.join(checkpoint_dir, 'alexnet_model_binary_classification.pth') if binary_classification else os.path.join(checkpoint_dir, 'alexnet_model_multiclass_classification.pth')

    try:
        model = load_model(model_path, model)
    except KeyError as e:
        print(f"Errore nel caricamento del modello: {e}")
        exit(1)
    except Exception as e:
        print(f"Errore sconosciuto nel caricamento del modello: {e}")
        exit(1)

    accuracy, precision, recall, f1 = calculate_metrics(test_loader, model, device, 1 if binary_classification else num_classes)

    print(f"Accuratezza Test: {accuracy:.4f}")
    print(f"Precisione Test: {precision:.4f}")
    print(f"Recall Test: {recall:.4f}")
    print(f"F1 Score Test: {f1:.4f}")

    metrics_path = os.path.join(checkpoint_dir, 'test_metrics.csv')
    if not os.path.exists(metrics_path):
        with open(metrics_path, 'w', newline='') as csvfile:
            fieldnames = ['accuracy', 'precision', 'recall', 'f1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(metrics_path, 'a', newline='') as csvfile:
        fieldnames = ['accuracy', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    print(f"Metriche di test salvate in: {metrics_path}")
