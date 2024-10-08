
# Data Augmentation modificato eliminando le maschere di frequenza e temporali

import os
import random
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
import logging
from queue import PriorityQueue
import numpy as np
import sys
import itertools
import uuid

# Imposta il seed per la riproducibilità
SEED = 10
random.seed(SEED)
torch.manual_seed(SEED)

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Contatore globale per tenere traccia dell'ordine di scoperta dei file
global_counter = itertools.count()


# Funzione per applicare trasformazioni casuali alle immagini
def apply_random_transform(image, unique_seed):
    # Imposta un seed unico per ogni immagine basato su unique_seed
    random.seed(unique_seed)
    torch.manual_seed(unique_seed)

    width, height = image.size

    # Flip orizzontale con probabilità del 50%
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    # Livelli randomici di rumore con probabilità del 50%
    if random.random() < 0.5:
        noise_level = random.uniform(5, 15)  # Livelli di rumore tra 5 e 15
        noise = torch.normal(0, noise_level, size=(height, width, 3))
        image_array = torch.tensor(np.array(image)) + noise
        image = Image.fromarray(np.uint8(torch.clamp(image_array, 0, 255).numpy()))

    # Time shifting con probabilità del 50%
    if random.random() < 0.5:
        shift = random.randint(-width // 4, width // 4)  # Shift casuale
        image_array = torch.tensor(np.array(image))
        image_array = torch.roll(image_array, shifts=shift, dims=1)
        image = Image.fromarray(image_array.numpy())

    # Ritaglio per riportare l'immagine alla dimensione originale
    image = ImageOps.fit(image, (width, height), method=0, bleed=0.0, centering=(0.5, 0.5))

    return image


def process_file(queue, pbar):
    while not queue.empty():
        _, (class_folder_path, sample, balanced_class_folder_path, is_augmented, unique_seed) = queue.get()
        try:
            sample_path = os.path.join(class_folder_path, sample)
            image = Image.open(sample_path)

            if is_augmented:
                unique_id = str(uuid.uuid4())  # Genera un identificatore univoco
                image = apply_random_transform(image, unique_seed)
                sample_name, extension = os.path.splitext(sample)
                transformed_sample_path = os.path.join(balanced_class_folder_path,
                                                       f"aug_{unique_seed}_{sample_name}{extension}")
            else:
                transformed_sample_path = os.path.join(balanced_class_folder_path, sample)

            image.save(transformed_sample_path)
            image.close()
            pbar.update(1)
            queue.task_done()
        except Exception as e:
            logging.error(f"Error processing file {sample_path}: {e}")
            queue.task_done()



def processing_scalograms(input_base_path, output_base_path, max_samples_per_leaf, seed):
    total_operations = 0
    file_queue = PriorityQueue()  # Utilizza una coda prioritaria anziché FIFO

    for root, dirs, files in sorted(os.walk(input_base_path)):
        original_samples = sorted([file for file in files if file.endswith(".png")])
        num_samples = len(original_samples)
        if num_samples == 0:
            continue

        # Creare il percorso della cartella bilanciata corrispondente
        relative_root = os.path.relpath(root, input_base_path)
        balanced_class_folder_path = os.path.join(output_base_path, relative_root)
        os.makedirs(balanced_class_folder_path, exist_ok=True)

        # Contatore locale per la cartella corrente
        local_counter = itertools.count()

        # Aggiungere campioni originali alla coda con priorità basata sull'ordine di scoperta
        for sample in original_samples:
            priority = next(global_counter)  # Priorità basata sull'ordine di scoperta
            file_queue.put((priority, (root, sample, balanced_class_folder_path, False, None)))

        # Se il numero di campioni è inferiore al massimo, esegui il sovracampionamento
        if num_samples < max_samples_per_leaf:
            num_additional_samples = max_samples_per_leaf - num_samples

            for i in range(num_additional_samples):
                sample_index = i % num_samples  # Usa un indice sequenziale
                sample = original_samples[sample_index]
                priority = next(global_counter)  # Priorità basata sull'ordine di scoperta
                unique_seed = seed + next(local_counter)  # Seed basato su SEED + contatore locale
                file_queue.put((priority, (root, sample, balanced_class_folder_path, True, unique_seed)))

        total_operations += max_samples_per_leaf if num_samples < max_samples_per_leaf else num_samples

    logging.info(f"Starting processing with {total_operations} total operations")

    with tqdm(total=total_operations, desc="Processing all classes") as pbar:
        process_file(file_queue, pbar)

    logging.info("Processing completed!")



if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Allenamento_Target")
    output_base_path = os.path.join(os.path.dirname(dataset_folder_path), "Bilanciamento_Allenamento_Target")
    max_samples_per_leaf = 10934  # Imposta il numero massimo di campioni per foglia
    input_base_path = dataset_folder_path

    processing_scalograms(input_base_path, output_base_path, max_samples_per_leaf, SEED)

    sys.stdout.write("\nProcessing completed!\n")
