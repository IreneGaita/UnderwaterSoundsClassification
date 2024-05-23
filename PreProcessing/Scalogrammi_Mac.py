import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
from tqdm import tqdm  # Importa tqdm per la barra di progresso

# Determina il numero di core disponibili e imposta il numero di thread per NumExpr
import multiprocessing
num_cores = multiprocessing.cpu_count()
numexpr_max_threads = max(1, num_cores // 2)
os.environ['NUMEXPR_MAX_THREADS'] = str(numexpr_max_threads)

# Configura il logging per monitorare lo stato
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lock per sincronizzare l'accesso alla sezione critica
plot_lock = threading.Lock()

def create_scalogram(audio_path, output_path):
    try:
        # Carica l'audio
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) == 0:
            raise ValueError("Audio file is empty")

        # Calcola la Continuous Wavelet Transform (CWT) utilizzando NumPy
        scales = np.arange(2, 250)  # Gamma di scale focalizzata
        coefficients, _ = pywt.cwt(y, scales, 'morl', sampling_period=1 / sr)
        power = np.abs(coefficients) ** 2  # Calcola il power

        # Verifica delle dimensioni dei coefficienti
        if power.shape[0] == 0 or power.shape[1] == 0:
            raise ValueError("CWT coefficients are empty")

        # Verifica che power sia un array 2D
        if power.ndim != 2:
            raise ValueError("CWT coefficients are not 2D")

        # Crea lo scalogramma
        with plot_lock:
            plt.figure(figsize=(10, 5))
            plt.imshow(power, extent=[0, len(y) / sr, 2, 250], interpolation='bilinear', aspect='auto', cmap='jet')
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

def count_files(directory):
    return sum(1 for root, _, files in os.walk(directory) for file in files if file.endswith('.wav'))

def process_file(queue, output_folder, total_files, pbar):
    while not queue.empty():
        file_path = queue.get()
        try:
            relative_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(file_path)))
            output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
            create_scalogram(file_path, output_subfolder)
        finally:
            pbar.update(1)  # Aggiorna la barra di progresso
            queue.task_done()

def process_scalograms(subfolder_paths, output_base_path):
    total_files = sum(count_files(subfolder) for subfolder in subfolder_paths)
    num_threads = max(1, num_cores // 2)  # Utilizza la metà dei core disponibili
    logging.info(f"Numero di core disponibili: {num_cores}, utilizzando {num_threads} thread")

    file_queue = Queue()
    for subfolder in subfolder_paths:
        for root, dirs, _ in os.walk(subfolder):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                for file in os.listdir(dir_path):
                    if file.endswith('.wav'):
                        file_queue.put(os.path.join(dir_path, file))

    # Crea la barra di progresso con una formattazione personalizzata
    with tqdm(total=total_files, desc="Elaborazione degli scalogrammi", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} files processed [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                executor.submit(process_file, file_queue, output_base_path, total_files, pbar)

        file_queue.join()

    logging.info(f"Number of files successfully processed: {total_files}")

if __name__ == "__main__":
    # Non verificare GPU su macOS
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "NewDataset")
    output_base_path = os.path.join(os.path.dirname(dataset_folder_path), "Scalograms")
    subfolders = ["Target", "Non-Target"]
    subfolder_paths = [os.path.join(dataset_folder_path, subfolder) for subfolder in subfolders]

    process_scalograms(subfolder_paths, output_base_path)

    sys.stdout.write("\nElaborazione completata!.\n")
