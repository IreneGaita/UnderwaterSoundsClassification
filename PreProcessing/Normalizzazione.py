import os
from PIL import Image
import numpy as np
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from queue import Queue
from tqdm import tqdm

# Configura il logging per monitorare lo stato
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def image_to_array(image):
    return np.array(image) / 255.0

def normalize_image(image_array, mean, std):
    return (image_array - mean) / std

def array_to_image(image_array):
    image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(image_array)

def process_image(file_path, output_path):
    mean = np.array([0, 0, 0])
    std = np.array([1.5, 1.5, 1.5])

    try:
        original_image = Image.open(file_path).convert("RGB")
        image_array = image_to_array(original_image)
        normalized_array = normalize_image(image_array, mean, std)
        final_image = array_to_image(normalized_array)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_image.save(output_path)

        return True
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return False

def count_files(directory):
    return sum(1 for root, _, files in os.walk(directory) for file in files if file.lower().endswith('.png'))

def process_file(queue, output_folder, pbar):
    while not queue.empty():
        file_path, relative_path = queue.get()
        try:
            output_path = os.path.join(output_folder, relative_path)
            process_image(file_path, output_path)
        finally:
            pbar.update(1)
            queue.task_done()

def process_images_in_directory(input_directories, output_directory):
    total_files = sum(count_files(subfolder) for subfolder in input_directories)
    num_threads = max(1, multiprocessing.cpu_count() // 2)

    file_queue = Queue()
    for input_directory in input_directories:
        for root, _, files in os.walk(input_directory):
            for file_name in files:
                if file_name.lower().endswith('.png'):
                    file_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(file_path, input_directory)
                    file_queue.put((file_path, relative_path))

    with tqdm(total=total_files, desc="Processing images", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} files processed [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                executor.submit(process_file, file_queue, output_directory, pbar)

        file_queue.join()

    logging.info(f"Number of files successfully processed: {total_files}")

if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Validazione_Scalogrammi")
    output_directory = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Validazione_Norm")
    subfolders = ["Target", "Non-Target"]
    subfolder_paths = [os.path.join(dataset_folder_path, subfolder) for subfolder in subfolders]

    os.makedirs(output_directory, exist_ok=True)
    process_images_in_directory(subfolder_paths, output_directory)

    sys.stdout.write("\nProcessing completed!\n")
