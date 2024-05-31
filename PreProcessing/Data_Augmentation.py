import os
import random
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import sys

# Imposta il seed per la riproducibilità
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funzione per applicare trasformazioni casuali alle immagini
def apply_random_transform(image):
    width, height = image.size

    # Flip orizzontale con probabilità del 50%
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    # Time Masking con probabilità del 50%
    if random.random() < 0.5:
        start_x = random.randint(0, width // 4)
        end_x = random.randint(start_x, start_x + width // 4)
        mask_color = (0, 0, 139)  # Colore blu scuro per mascherare
        mask = Image.new('RGB', (end_x - start_x, height), mask_color)
        image.paste(mask, (start_x, 0))

    # Frequency Masking con probabilità del 50%
    if random.random() < 0.5:
        start_y = random.randint(0, height // 4)
        end_y = random.randint(start_y, start_y + height // 4)
        mask_color = (0, 0, 139)  # Colore blu scuro per mascherare
        mask = Image.new('RGB', (width, end_y - start_y), mask_color)
        image.paste(mask, (0, start_y))

    # Livelli randomici di rumore con probabilità del 50%
    if random.random() < 0.5:
        noise_level = random.uniform(5, 15)  # Livelli di rumore tra 5 e 15
        noise = np.random.normal(0, noise_level, (height, width, 3))
        image_array = np.array(image) + noise
        image = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))

    # Time shifting con probabilità del 50%
    if random.random() < 0.5:
        shift = random.randint(-width // 4, width // 4)  # Shift casuale
        image_array = np.array(image)
        image_array = np.roll(image_array, shift, axis=1)
        image = Image.fromarray(image_array)

    # Ritaglio per riportare l'immagine alla dimensione originale
    image = ImageOps.fit(image, (width, height), method=0, bleed=0.0, centering=(0.5, 0.5))

    return image

# Funzione per processare i file dalla coda
def process_file(queue, pbar, seed):
    while not queue.empty():
        class_folder_path, sample, balanced_class_folder_path, is_augmented, index = queue.get()
        sample_path = os.path.join(class_folder_path, sample)
        image = Image.open(sample_path)

        # Imposta il seed per ogni processo di trasformazione per garantire riproducibilità
        random.seed(seed)
        np.random.seed(seed)

        if is_augmented:
            image = apply_random_transform(image)
            transformed_sample_path = os.path.join(balanced_class_folder_path, f"aug_{index}_{sample}")
        else:
            transformed_sample_path = os.path.join(balanced_class_folder_path, sample)

        image.save(transformed_sample_path)
        image.close()
        pbar.update(1)
        queue.task_done()

def processing_scalograms(subfolder_paths, output_base_path, seed):
    total_operations = 0
    file_queue = Queue()

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

    # Esecuzione del processing usando threading e la barra di progresso
    num_threads = max(1, os.cpu_count() // 2)

    with tqdm(total=total_operations, desc="Processing all classes") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                executor.submit(process_file, file_queue, pbar, seed)

        file_queue.join()

    logging.info("Processing completed!")

if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Normalizzazione")
    output_base_path = os.path.join(os.path.dirname(dataset_folder_path), "Bilanciamento_Scalogrammi")
    subfolders = ["Target", "Non-Target"]
    subfolder_paths = [os.path.join(dataset_folder_path, subfolder) for subfolder in subfolders]

    processing_scalograms(subfolder_paths, output_base_path, SEED)

    sys.stdout.write("\nProcessing completed!\n")