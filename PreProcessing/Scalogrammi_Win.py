import os
import sys
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading
from queue import Queue
from tqdm import tqdm

num_cores = multiprocessing.cpu_count()
numexpr_max_threads = max(1, num_cores // 2)
os.environ['NUMEXPR_MAX_THREADS'] = str(numexpr_max_threads)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_gpu_info():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        return f"Using GPU: {device_name}, Total Memory: {total_memory} bytes"
    else:
        return "No GPU available. Using CPU."

logging.info(get_gpu_info())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

plot_lock = threading.Lock()

def create_scalogram(audio_path, output_path, log_file):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) == 0:
            raise ValueError("Audio file is empty")

        y_gpu = torch.tensor(y, device=device, dtype=torch.float32)

        scales = torch.arange(2, 250, device=device, dtype=torch.float32)
        coefficients, _ = pywt.cwt(y_gpu.cpu().numpy(), scales.cpu().numpy(), 'morl', sampling_period=1 / sr)
        power = torch.tensor(np.abs(coefficients) ** 2, device=device)

        if power.shape[0] == 0 or power.shape[1] == 0:
            raise ValueError("CWT coefficients are empty")

        power_cpu = power.cpu().numpy()

        if power_cpu.ndim != 2:
            raise ValueError("CWT coefficients are not 2D")

        with plot_lock:
            plt.figure(figsize=(10, 5))
            plt.imshow(power_cpu, extent=[0, len(y) / sr, 2, 250], interpolation='bilinear', aspect='auto', cmap='jet')
            plt.axis('off')

            os.makedirs(output_path, exist_ok=True)

            output_file_path = os.path.join(output_path, os.path.basename(audio_path).replace('.wav', '.png'))
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Update log file
            with open(log_file, 'a') as f:
                f.write(audio_path + '\n')

        return output_file_path
    except Exception as e:
        logging.error(f"Error creating scalogram for file {audio_path}: {e}")
        return None

def count_files(directory):
    return sum(1 for root, _, files in os.walk(directory) for file in files if file.endswith('.wav'))

def process_file(queue, output_folder, pbar, log_file):
    while not queue.empty():
        file_path = queue.get()
        try:
            if not os.path.exists(log_file) or file_path not in open(log_file).read():
                relative_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(file_path)))
                output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
                create_scalogram(file_path, output_subfolder, log_file)
        finally:
            pbar.update(1)
            queue.task_done()

def processing_scalograms(subfolder_paths, output_base_path):
    log_file = os.path.join(output_base_path, 'processing_log.txt')
    total_files = sum(count_files(subfolder) for subfolder in subfolder_paths)
    num_threads = max(1, num_cores // 2)

    file_queue = Queue()
    for subfolder in subfolder_paths:
        for root, dirs, _ in os.walk(subfolder):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                for file in os.listdir(dir_path):
                    if file.endswith('.wav'):
                        file_queue.put(os.path.join(dir_path, file))

    with tqdm(total=total_files, desc="Processing scalograms", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} files processed [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                executor.submit(process_file, file_queue, output_base_path, pbar, log_file)

        file_queue.join()

    logging.info(f"Number of files successfully processed: {total_files}")

if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "NewDataset")
    output_base_path = os.path.join(os.path.dirname(dataset_folder_path), "Scalograms")
    subfolders = ["Target", "Non-Target"]
    subfolder_paths = [os.path.join(dataset_folder_path, subfolder) for subfolder in subfolders]

    processing_scalograms(subfolder_paths, output_base_path)

    sys.stdout.write("\nProcessing completed!\n")
