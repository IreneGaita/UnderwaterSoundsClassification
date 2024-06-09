import torch
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast
import os
import csv
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


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


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_epoch += 1
    metrics = checkpoint['metrics']
    return model, optimizer, start_epoch, metrics


def save_checkpoint(checkpoint_state, checkpoint_path):
    torch.save(checkpoint_state, checkpoint_path)



def calculate_metrics(loader, model, device):
    model.eval()
    all_labels = []
    all_preds = []
    pbar = tqdm(loader, desc="Calcolo metriche")
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.round())
            pbar.update()

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset_path = r'C:\Users\biagi\PycharmProjects\gruppo17\Bilanciamento_Allenamento'
    val_dataset_path = r'C:\Users\biagi\PycharmProjects\gruppo17\Validazione_Norm'

    train_dataset = CustomDataset(train_dataset_path, transform=transform)
    val_dataset = CustomDataset(val_dataset_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    # Caricamento del modello EfficientNet B3 pre-addestrato
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

    # Modifica dell'ultimo strato per la classificazione binaria
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)


    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    torch.backends.cudnn.benchmark = True
    scaler = GradScaler()

    checkpoint_dir = '../Esperimento 1/EfficientNetB3_Binary'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    latest_checkpoint = glob.glob(os.path.join(checkpoint_dir, 'best_checkpoint.pth'))
    if latest_checkpoint:
        latest_checkpoint = max(latest_checkpoint, key=os.path.getctime)
        model, optimizer, start_epoch, metrics = load_checkpoint(latest_checkpoint, model, optimizer)
    else:
        model, optimizer, start_epoch, metrics = model, optimizer, 0, []

    num_epochs = 50
    checkpoint_interval = 1
    patience = 5
    performance_drop_patience = 3
    best_val_loss = float('inf')
    patience_counter = 0
    performance_drop_counter = 0
    last_val_metrics = None
    gradient_accumulation_steps = 4

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad()
        pbar_train = tqdm(train_loader, desc=f"Epoca {epoch + 1}/{num_epochs} - Addestramento")
        for step, (inputs, labels) in enumerate(pbar_train):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                train_loss = criterion(outputs, labels.float().view(-1, 1))
            scaler.scale(train_loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_train_loss += train_loss.item()
            pbar_train.set_postfix({'Loss': running_train_loss / (step + 1)})

        validation_interval = 1

        if (epoch + 1) % validation_interval == 0:
            model.eval()
            running_val_loss = 0.0
            pbar_val = tqdm(val_loader, desc=f"Epoca {epoch + 1}/{num_epochs} - Valutazione")
            for inputs, labels in pbar_val:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.no_grad():
                    with autocast():
                        outputs = model(inputs)
                        val_loss = criterion(outputs, labels.float().view(-1, 1))
                    running_val_loss += val_loss.item()
                    pbar_val.set_postfix({'Val Loss': running_val_loss / len(val_loader)})

            train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(train_loader, model, device)
            val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(val_loader, model, device)
            epoch_train_loss = running_train_loss / len(train_loader)
            epoch_val_loss = running_val_loss / len(val_loader)

            print(f"Epoca {epoch + 1}/{num_epochs} - "
                  f"Loss Addestramento: {epoch_train_loss:.4f} - Acc Addestramento: {train_accuracy:.4f} - "
                  f"Precision Addestramento: {train_precision:.4f} - Recall Addestramento: {train_recall:.4f} - "
                  f"F1 Addestramento: {train_f1:.4f} - "
                  f"Loss Valutazione: {epoch_val_loss:.4f} - Acc Valutazione: {val_accuracy:.4f} - "
                  f"Precision Valutazione: {val_precision:.4f} - Recall Valutazione: {val_recall:.4f} - F1 Valutazione: {val_f1:.4f}")

            metrics.append({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_accuracy": train_accuracy,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "val_loss": epoch_val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            })

            metrics_path = os.path.join(checkpoint_dir, f'training_metrics_epoch_{epoch + 1}.csv')
            with open(metrics_path, 'w', newline='') as csvfile:
                fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
                              'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for metric in metrics:
                    writer.writerow(metric)

            print(f"Metriche per l'epoca {epoch + 1} salvate in: {metrics_path}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                performance_drop_counter = 0
                checkpoint_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }
                checkpoint_path = os.path.join(checkpoint_dir, f'best_checkpoint.pth')
                save_checkpoint(checkpoint_state, checkpoint_path)
                print(f"Miglior checkpoint salvato in: {checkpoint_path}")
            else:
                patience_counter += 1

            if last_val_metrics:
                if (val_accuracy < last_val_metrics['val_accuracy'] and
                        val_precision < last_val_metrics['val_precision'] and
                        val_recall < last_val_metrics['val_recall'] and
                        val_f1 < last_val_metrics['val_f1']):
                    performance_drop_counter += 1
                    print(f"Prestazioni calate all'epoca {epoch + 1}, contatore di calo: {performance_drop_counter}")
                else:
                    performance_drop_counter = 0

            last_val_metrics = {
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            }

            if patience_counter >= patience:
                print(f"Early stopping attivato dopo {epoch + 1} epoche senza miglioramenti")
                break

            if performance_drop_counter >= performance_drop_patience:
                print(f"Addestramento interrotto dopo {epoch + 1} epoche di calo delle prestazioni consecutive")
                break

    torch.save(model.state_dict(), 'alexnet_model_binary_classification.pth')
    model_path = os.path.abspath('alexnet_model_binary_classification.pth')
    print(f"Modello salvato in: {model_path}")