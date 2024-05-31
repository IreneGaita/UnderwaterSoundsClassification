import os
import random
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import logging
from multiprocessing import set_start_method, Queue as MPQueue, Process
import torch
import torch.nn.functional as F
from torch import tensor
import sys

# Imposta il seed per la riproducibilità
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funzione per applicare trasformazioni casuali alle immagini
def apply_random_transform(image):
    width, height = image.size
    image_tensor = tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

    # Flip orizzontale con probabilità del 50%
    if random.random() < 0.5:
        image_tensor = torch.flip(image_tensor, [2])

    # Time Masking con probabilità del 50%
    if random.random() < 0.5:
        start_x = random.randint(0, width // 4)
        end_x = random.randint(start_x, start_x + width // 4)
        image_tensor[:, :, start_x:end_x] = 0

    # Frequency Masking con probabilità del 50%
    if random.random() < 0.5:
        start_y = random.randint(0, height // 4)
        end_y = random.randint(start_y, start_y + height // 4)
        image_tensor[:, start_y:end_y, :] = 0

    # Livelli randomici di rumore con probabilità del 50%
    if random.random() < 0.5:
        noise_level = random.uniform(0.02, 0.06)  # Livelli di rumore tra 2% e 6%
        noise = torch.randn_like(image_tensor) * noise_level
        image_tensor = image_tensor + noise
        image_tensor = torch.clamp(image_tensor, 0, 1)

    # Time shifting con probabilità del 50%
    if random.random() < 0.5:
        shift = random.randint(-width // 4, width // 4)  # Shift casuale
        image_tensor = torch.roll(image_tensor, shifts=shift, dims=2)

    # Ritaglio per riportare l'immagine alla dimensione originale
    image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
    image = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    return image

# Funzione per processare i file dalla coda
def process_file(file_queue, progress_queue, seed):
    while True:
        try:
            class_folder_path, sample, balanced_class_folder_path, is_augmented, index = file_queue.get_nowait()
        except queue.Empty:
            return

        sample_path = os.path.join(class_folder_path, sample)
        image = Image.open(sample_path)

        # Imposta il seed per ogni processo di trasformazione per garantire riproducibilità
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if is_augmented:
            image = apply_random_transform(image)
            transformed_sample_path = os.path.join(balanced_class_folder_path, f"aug_{index}_{sample}")
        else:
            transformed_sample_path = os.path.join(balanced_class_folder_path, sample)

        image.save(transformed_sample_path)
        image.close()
        progress_queue.put(1)

def update_progress_bar(progress_queue, total_operations):
    with tqdm(total=total_operations, desc="Processing all classes") as pbar:
        while True:
            update = progress_queue.get()
            if update is None:
                break
            pbar.update(update)

def processing_scalograms(subfolder_paths, output_base_path, seed):
    total_operations = 0
    file_queue = MPQueue()
    progress_queue = MPQueue()

    for subfolder_path in subfolder_paths:
        for root, _, files in os.walk(subfolder_path):
            original_samples = [file for file in files if file.endswith(".png")]
            num_samples = len(original_samples)
            max_samples = 10934  # Modifica questo numero in base alle tue necessità

            # Create the corresponding balanced folder path
            relative_root = os.path.relpath(root, subfolder_path)
            balanced_class_folder_path = os.path.join(output_base_path, relative_root)
            os.makedirs(balanced_class_folder_path, exist_ok=True)

            # Aggiungere campioni originali alla coda
            for sample in original_samples:
                file_queue.put((root, sample, balanced_class_folder_path, False, None))

            # Se il numero di campioni è inferiore al massimo, esegui il sovracampionamento
            if num_samples > 0 and num_samples < max_samples:
                num_additional_samples = max_samples - num_samples

                for i in range(num_additional_samples):
                    sample = random.choice(original_samples)
                    file_queue.put((root, sample, balanced_class_folder_path, True, i))

            total_operations += max_samples if num_samples < max_samples else num_samples

    # Esecuzione del processing usando multiprocessing e la barra di progresso
    num_threads = max(1, os.cpu_count() // 2)

    progress_process = Process(target=update_progress_bar, args=(progress_queue, total_operations))
    progress_process.start()

    processes = []
    for _ in range(num_threads):
        p = Process(target=process_file, args=(file_queue, progress_queue, seed))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    progress_queue.put(None)
    progress_process.join()

    logging.info("Processing completed!")

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Normalizzazione")
    output_base_path = os.path.join(os.path.dirname(dataset_folder_path), "Bilanciamento_Scalogrammi")
    subfolders = ["Target", "Non-Target"]
    subfolder_paths = [os.path.join(dataset_folder_path, subfolder) for subfolder in subfolders]

    processing_scalograms(subfolder_paths, output_base_path, SEED)

    sys.stdout.write("\nProcessing completed!\n")
