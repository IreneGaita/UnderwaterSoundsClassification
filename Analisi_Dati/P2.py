import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

class ResizeWithPadding(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        img = image.copy().convert("RGB")

        # Calcoliamo la scala per il ridimensionamento mantenendo le proporzioni originali
        img.thumbnail((self.output_size[0], self.output_size[1]), Image.BILINEAR)

        # Calcoliamo il padding necessario per ottenere le dimensioni desiderate
        delta_w = self.output_size[0] - img.size[0]
        delta_h = self.output_size[1] - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

        # Aggiungiamo il padding all'immagine
        img = ImageOps.expand(img, padding, fill=(0, 0, 0))

        return img

def tensor_to_image(tensor, output_path):
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Rinormalizza l'immagine per visualizzarla correttamente
    tensor = np.clip(tensor, 0, 1)

    image = Image.fromarray((tensor * 255).astype(np.uint8))
    image.save(output_path)

preprocess = transforms.Compose([
    ResizeWithPadding((224, 224)),
    transforms.ToTensor(),
])

image_path = r'C:\Users\biagi\PycharmProjects\gruppo17\Allenamento_Target\Cargo\20171104-1_1_part8.png'
output_path = r'C:\Users\biagi\Desktop\Nuova\preprocessed_image.png'

try:
    original_image = Image.open(image_path)
    preprocessed_image = preprocess(original_image)
    preprocessed_image = preprocessed_image.unsqueeze(0)

    # Utilizza valori medi e deviazione standard comunemente usati (ImageNet)
    mean = torch.tensor([0, 0, 0])
    std = torch.tensor([1.5, 1.5, 1.5])

    # Esegui la normalizzazione
    preprocessed_image = (preprocessed_image - mean[:, None, None]) / std[:, None, None]

    tensor_to_image(preprocessed_image, output_path)

    print("Valori minimi e massimi dell'immagine originale:")
    extrema = original_image.getextrema()
    num_channels = len(extrema)
    # Considera solo i primi tre canali
    for i in range(min(num_channels, 3)):
        print(f"Canale {i + 1}: {extrema[i]}")

    print("\nValori minimi e massimi dell'immagine normalizzata:")
    for i in range(3):  # L'immagine pre-elaborata ha solo 3 canali (RGB)
        channel_data = preprocessed_image[0, i, :, :]
        print(f"Canale {i + 1}: {channel_data.min().item()} {channel_data.max().item()}")

except Exception as e:
    print("Errore durante il pre-processing dell'immagine:", e)
