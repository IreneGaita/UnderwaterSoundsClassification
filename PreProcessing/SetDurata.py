import os
import librosa
import soundfile as sf
import numpy as np
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from threading import Lock

# Configura il logging per monitorare lo stato
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def adjust_audio_length(y, sr, target_length):
    current_length = len(y) / sr  # current length in seconds
    if current_length < target_length:
        while len(y) / sr < target_length:
            start = np.random.randint(0, len(y) - 1)
            end = min(start + int(target_length * sr), len(y))
            y = np.concatenate([y, y[start:end]])
        y = y[:int(target_length * sr)]
    return y


def segment_audio(y, sr, segment_length):
    segments = []
    for start in range(0, len(y), segment_length * sr):
        end = min(start + segment_length * sr, len(y))
        segment = y[start:end]
        if len(segment) < segment_length * sr:
            # Ignora la parte in eccesso se è inferiore a 1 secondo
            if len(segment) < sr:
                break
            segment = adjust_audio_length(segment, sr, segment_length)
        segments.append(segment)
    return segments


def process_audio_file(file_path, output_dir, segment_length, processed_files_counter, lock):
    output_subdir = os.path.join(output_dir, os.path.relpath(os.path.dirname(file_path), input_dir))
    os.makedirs(output_subdir, exist_ok=True)
    base_name, ext = os.path.splitext(os.path.basename(file_path))
    try:
        y, sr = librosa.load(file_path, sr=None)
        segments = segment_audio(y, sr, segment_length)
        for idx, segment in enumerate(segments):
            output_file_path = os.path.join(output_subdir, f"{base_name}_part{idx}{ext}")
            sf.write(output_file_path, segment, sr)
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
    finally:
        with lock:
            processed_files_counter[0] += 1
            progress = (processed_files_counter[0] / total_files) * 100
            sys.stdout.write(f"\rProgresso: {progress:.2f}%")
            sys.stdout.flush()


def process_audio_files(input_dir, output_dir, segment_length):
    global total_files
    exclude_files = ['.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv']
    total_files = sum(len([f for f in files if f.endswith('.wav') and f not in exclude_files])
                      for _, _, files in os.walk(input_dir))

    processed_files_counter = [0]
    lock = Lock()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for root, _, files in os.walk(input_dir):
            for file_name in files:
                if not file_name.endswith('.wav') or file_name in exclude_files:
                    continue
                file_path = os.path.join(root, file_name)
                future = executor.submit(process_audio_file, file_path, output_dir, segment_length,
                                         processed_files_counter, lock)
                futures.append(future)

    # Attendi il completamento di tutte le attività
    for future in as_completed(futures):
        future.result()

    sys.stdout.write('\n')


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    parent_folder = os.path.dirname(os.path.dirname(current_file))
    input_dir = os.path.join(parent_folder, "Dataset")
    output_dir = os.path.join(parent_folder, "NewDataset")
    os.makedirs(output_dir, exist_ok=True)

    segment_length = 3  # segment length in seconds

    # Determina il numero di core disponibili e imposta il numero di thread
    num_cores = multiprocessing.cpu_count()
    num_threads = max(1, num_cores // 2)  # Utilizza la metà dei core disponibili
    logging.info(f"Numero di core disponibili: {num_cores}, utilizzando {num_threads} thread")

    process_audio_files(input_dir, output_dir, segment_length)

    sys.stdout.write("\nElaborazione completata.\n")
