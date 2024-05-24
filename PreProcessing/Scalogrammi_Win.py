import os
import sys
import cupy as cp
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

def get_available_gpus():
    try:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        logging.info(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            device_properties = cp.cuda.runtime.getDeviceProperties(i)
            logging.info(f"GPU {i}: {device_properties['name']}")
    except Exception as e:
        logging.error(f"Error getting available GPUs: {e}")
        num_gpus = 0
    return num_gpus

gpu_id = 0

plot_lock = threading.Lock()

def create_scalogram(audio_path, output_path, log_file):
    try:
        with cp.cuda.Device(gpu_id):
            y, sr = librosa.load(audio_path, sr=None)
            if len(y) == 0:
                raise ValueError("Audio file is empty")

            y_gpu = cp.array(y)

            scales = np.arange(2, 250)
            coefficients, _ = pywt.cwt(cp.asnumpy(y_gpu), scales, 'morl', sampling_period=1 / sr)
            power = cp.array(np.abs(coefficients) ** 2)

            if power.shape[0] == 0 or power.shape[1] == 0:
                raise ValueError("CWT coefficients are empty")

            power_cpu = cp.asnumpy(power)

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
    available_gpus = get_available_gpus()
    if available_gpus > 0:
        current_file = os.path.abspath(__file__)
        dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "NewDataset")
        output_base_path = os.path.join(os.path.dirname(dataset_folder_path), "Scalograms")
        subfolders = ["Target", "Non-Target"]
        subfolder_paths = [os.path.join(dataset_folder_path, subfolder) for subfolder in subfolders]

        processing_scalograms(subfolder_paths, output_base_path)

        sys.stdout.write("\nProcessing completed!\n")
    else:
        logging.error("No GPUs available. Please check your CUDA installation and GPU configuration.")
