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

# Determina il numero di core disponibili e imposta il numero di thread per NumExpr
num_cores = multiprocessing.cpu_count()
numexpr_max_threads = max(1, num_cores // 2)
os.environ['NUMEXPR_MAX_THREADS'] = str(numexpr_max_threads)

# Configura il logging per monitorare lo stato
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funzione per ottenere il numero di GPU disponibili
def get_available_gpus():
    try:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        logging.info(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            device_properties = cp.cuda.runtime.getDeviceProperties(i)
            logging.info(f"GPU {i}: {device_properties['name']}")
    except Exception as e:
        logging.error(f"Errore nell'ottenere il numero di GPU disponibili: {e}")
        num_gpus = 0
    return num_gpus

# Imposta l'ID della GPU da utilizzare (modifica questo valore in base all'output di get_available_gpus)
gpu_id = 0  # Assicurati che questo ID sia valido

# Lock per sincronizzare l'accesso alla sezione critica
plot_lock = threading.Lock()

def create_scalogram(audio_path, output_path):
    try:
        with cp.cuda.Device(gpu_id):
            # Carica l'audio
            y, sr = librosa.load(audio_path, sr=None)
            if len(y) == 0:
                raise ValueError("Audio file is empty")

            # Converti l'audio a un array CuPy
            y_gpu = cp.array(y)

            # Calcola la Continuous Wavelet Transform (CWT) utilizzando NumPy
            scales = np.arange(2, 250)  # Gamma di scale focalizzata
            coefficients, _ = pywt.cwt(cp.asnumpy(y_gpu), scales, 'morl', sampling_period=1 / sr)
            power = cp.array(np.abs(coefficients) ** 2)  # Converti i risultati di nuovo a CuPy

            # Verifica delle dimensioni dei coefficienti
            if power.shape[0] == 0 or power.shape[1] == 0:
                raise ValueError("CWT coefficients are empty")

            # Converti i risultati di nuovo a NumPy per la visualizzazione
            power_cpu = cp.asnumpy(power)

            # Verifica che power_cpu sia un array 2D
            if power_cpu.ndim != 2:
                raise ValueError("CWT coefficients are not 2D")

            # Crea lo scalogramma
            with plot_lock:
                plt.figure(figsize=(10, 5))
                plt.imshow(power_cpu, extent=[0, len(y) / sr, 2, 250], interpolation='bilinear', aspect='auto', cmap='jet')
                plt.colorbar(label='Power')
                plt.ylabel('Scale')
                plt.xlabel('Time [s]')
                plt.title('Scalogram')

                # Crea la cartella di output se non esiste
                os.makedirs(output_path, exist_ok=True)

                # Salva lo scalogramma come immagine
                output_file_path = os.path.join(output_path, os.path.basename(audio_path).replace('.wav', '.png'))
                plt.savefig(output_file_path)
                plt.close()

            return output_file_path
    except Exception as e:
        logging.error(f"Errore durante la creazione dello scalogramma per il file {audio_path}: {e}")
        return None

def progress_callback(processed, total):
    progress = (processed / total) * 100
    sys.stdout.write(f"\rProgresso: {progress:.2f}%")
    sys.stdout.flush()

def count_files(directory):
    return sum(1 for root, _, files in os.walk(directory) for file in files if file.endswith('.wav'))

def process_file(queue, output_folder, total_files, processed_files_lock, processed_files_counter):
    while not queue.empty():
        file_path = queue.get()
        try:
            relative_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(file_path)))
            output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
            create_scalogram(file_path, output_subfolder)
        finally:
            with processed_files_lock:
                processed_files_counter[0] += 1
                progress_callback(processed_files_counter[0], total_files)
            queue.task_done()

def processing_scalograms(subfolder_paths, output_base_path):
    total_files = sum(count_files(subfolder) for subfolder in subfolder_paths)
    processed_files_counter = [0]
    processed_files_lock = threading.Lock()

    num_threads = max(1, num_cores // 2)  # Utilizza la metà dei core disponibili
    logging.info(f"Numero di core disponibili: {num_cores}, utilizzando {num_threads} thread")

    file_queue = Queue()
    for subfolder in subfolder_paths:
        for root, dirs, _ in os.walk(subfolder):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                for file in os.listdir(dir_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(dir_path, file)
                        file_queue.put(file_path)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(num_threads):
            executor.submit(process_file, file_queue, output_base_path, total_files, processed_files_lock, processed_files_counter)

    file_queue.join()

    logging.info(f"Number of files successfully processed: {processed_files_counter[0]}")
    logging.info(f"Number of files failed to process: {total_files - processed_files_counter[0]}")

if __name__ == "__main__":
    available_gpus = get_available_gpus()
    if available_gpus > 0:
        current_file = os.path.abspath(__file__)
        dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "NewDataset")
        output_base_path = os.path.join(os.path.dirname(dataset_folder_path), "Scalograms")
        subfolders = ["Target", "Non-Target"]
        subfolder_paths = [os.path.join(dataset_folder_path, subfolder) for subfolder in subfolders]

        processing_scalograms(subfolder_paths, output_base_path)

        sys.stdout.write("\nElaborazione completata!.\n")
    else:
        logging.error("No GPUs available. Please check your CUDA installation and GPU configuration.")
