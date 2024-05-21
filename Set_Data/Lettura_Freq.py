import sys
import librosa
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import logging
from Caricamento_Audio import load_audio_files

# Configura il logging per monitorare lo stato
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_audio_frequency(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        return sr
    except Exception as e:
        logging.error(f"Errore nel decodificare il file {file_path}: {e}")
        return 0


def find_max_frequency(file_path, high_pass_filter=False, cutoff_freq=100.0, amplitude_threshold=0.001):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if np.all(y == 0):
            return 0.0, sr
        if high_pass_filter:
            y = librosa.effects.preemphasis(y, coef=cutoff_freq / sr)
        n_fft = min(len(y), 4096)
        S = np.abs(librosa.stft(y, n_fft=n_fft))
        S_mean = np.mean(S, axis=1)
        frequencies = np.linspace(0, sr / 2, len(S_mean))
        threshold = amplitude_threshold * np.max(S_mean)
        significant_indices = np.where(S_mean > threshold)[0]
        if len(significant_indices) > 0:
            max_frequency = frequencies[significant_indices[-1]]
        else:
            max_frequency = frequencies[np.argmax(S_mean)]
        return round(max_frequency), sr
    except Exception as e:
        logging.error(f"Errore nel calcolare la frequenza massima per il file {file_path}: {e}")
        return 0, 0


def process_file(file_path):
    result = {}
    try:
        sampling_frequency = get_audio_frequency(file_path)
        if sampling_frequency > 0:
            result['sampling_frequency'] = sampling_frequency

        max_frequency, sr = find_max_frequency(file_path, high_pass_filter=True, cutoff_freq=100.0,
                                               amplitude_threshold=0.001)
        if max_frequency > 0:
            result['max_frequency'] = max_frequency

    except Exception as e:
        logging.error(f"Errore nel processare il file {file_path}: {e}")
    return result


def analyze_audio_files():
    audio_files = load_audio_files()
    total_files = len(audio_files)

    sampling_frequency_counter = defaultdict(int)
    max_frequency_counter = defaultdict(int)

    file_count = 0

    # Determina il numero di core disponibili e imposta il numero di thread
    num_cores = multiprocessing.cpu_count()
    num_threads = max(1, num_cores // 2)  # Utilizza la metà dei core disponibili
    logging.info(f"Numero di core disponibili: {num_cores}, utilizzando {num_threads} thread")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_file, file_path) for file_path in audio_files]

        for future in as_completed(futures):
            result = future.result()
            if 'sampling_frequency' in result:
                sampling_frequency_counter[result['sampling_frequency']] += 1
            if 'max_frequency' in result:
                max_frequency_counter[result['max_frequency']] += 1

            file_count += 1
            sys.stdout.write(f"\rProgresso: {(file_count / total_files) * 100:.2f}%")
            sys.stdout.flush()

    sys.stdout.write('\n')

    return sampling_frequency_counter, max_frequency_counter


if __name__ == "__main__":
    sampling_frequency_counter, max_frequency_counter = analyze_audio_files()

    print("\nFrequenze di campionamento:")
    for frequency, count in sorted(sampling_frequency_counter.items()):
        print(f"Frequenza di campionamento: {frequency} Hz, Numero di file: {count}")

    print("\nFrequenze massime durante la riproduzione:")
    for frequency, count in sorted(max_frequency_counter.items()):
        print(f"Frequenza massima: {frequency} Hz, Numero di file: {count}")