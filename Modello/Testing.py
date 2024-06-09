import os
import csv
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn


# Funzione per il calcolo delle metriche
def calculate_metrics(loader, model, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))
            correct_predictions += (predicted == labels.view_as(predicted)).sum().item()
            total_predictions += labels.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Definizione del percorso del modello
model_path = "alexnet_model_binary_classification.pth"

# Caricamento del modello
model = models.alexnet()
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
model.load_state_dict(torch.load(model_path))
model.eval()

# Definizione del percorso del set di dati di test
test_dataset_path = r'C:\Users\biagi\PycharmProjects\gruppo17\Set_Dati_Test'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Creazione del DataLoader per il set di dati di test
test_dataset = CustomDataset(test_dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# Calcolo delle metriche sul set di dati di test
test_accuracy = calculate_metrics(test_loader, model, device)

# Salvataggio delle metriche in un file CSV
test_metrics_path = os.path.join(checkpoint_dir, 'test_metrics.csv')
with open(test_metrics_path, 'w', newline='') as csvfile:
    fieldnames = ['test_accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'test_accuracy': test_accuracy})

print(f"Metriche di test salvate in: {test_metrics_path}")
