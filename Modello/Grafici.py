import pandas as pd
import matplotlib.pyplot as plt

# Leggi i dati dal file CSV
df = pd.read_csv(r'/20240605_172503/best_checkpoint.pth')

# Estrai le metriche di addestramento e validazione
train_metrics = ['train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1']
val_metrics = ['val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1']

# Crea un grafico separato per ogni coppia di metriche
for train_metric, val_metric in zip(train_metrics, val_metrics):
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df[train_metric], label='Train ' + train_metric.split('_')[1].capitalize())
    plt.plot(df['epoch'], df[val_metric], label='Validation ' + val_metric.split('_')[1].capitalize())
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title(train_metric.split('_')[1].capitalize() + ' Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()