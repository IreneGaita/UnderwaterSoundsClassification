import os
import librosa
import soundfile as sf
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Configura il logging per monitorare lo stato
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definisci i file da escludere
exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}

def process_file(file_path):
    target_sample_rate = 86400
    try:
        if file_path.endswith(".mp3"):
            # Load MP3 file using librosa with target sample rate
            y, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)

            # Convert to WAV format
            wav_file_path = file_path.replace(".mp3", ".wav")
            sf.write(wav_file_path, y, sr, subtype='PCM_16')
            os.remove(file_path)
        else:
            # Load WAV file (or other formats) using librosa with target sample rate
            y, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)

            # Standardize bit depth to 16-bit
            sf.write(file_path, y, sr, subtype='PCM_16')
        return True
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return False

def count_files(directory, exclude_files):
    return sum(1 for root, _, files in os.walk(directory) for file in files if file not in exclude_files)

def process_audio_files(directories):
    total_files = sum(count_files(directory, exclude_files) for directory in directories)
    file_count = 0

    # Determina il numero di core disponibili e imposta il numero di thread
    num_cores = multiprocessing.cpu_count()
    num_threads = max(1, num_cores // 2)  # Utilizza la metà dei core disponibili
    logging.info(f"Numero di core disponibili: {num_cores}, utilizzando {num_threads} thread")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file_name in files:
                    if file_name in exclude_files:
                        continue
                    file_path = os.path.join(root, file_name)
                    futures.append(executor.submit(process_file, file_path))

                    # Limita la coda delle attività per evitare sovraccarico
                    if len(futures) >= num_threads * 2:  # Limita a due volte il numero di thread
                        for future in as_completed(futures):
                            result = future.result()
                            if result:
                                file_count += 1
                            progress = (file_count / total_files) * 100
                            sys.stdout.write(f"\rProgresso: {progress:.2f}%")
                            sys.stdout.flush()
                        futures = []

        # Completa le rimanenti attività
        for future in as_completed(futures):
            result = future.result()
            if result:
                file_count += 1
            progress = (file_count / total_files) * 100
            sys.stdout.write(f"\rProgresso: {progress:.2f}%")
            sys.stdout.flush()

    sys.stdout.write('\n')

if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "NewDataset")
    subfolders = ["Target", "Non-Target"]
    subfolder_paths = [os.path.join(dataset_folder_path, subfolder) for subfolder in subfolders]

    process_audio_files(subfolder_paths)

    sys.stdout.write("\nElaborazione completata!.\n")
