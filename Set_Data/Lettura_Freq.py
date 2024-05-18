import os
from pydub import AudioSegment
import sys
import librosa
import numpy as np
from collections import defaultdict

def count_files(directory, exclude_files):
    # Conta i file totali escludendo quelli nella lista exclude_files.
    return sum(1 for root, _, files in os.walk(directory) for file in files if file not in exclude_files)

def get_audio_frequency(file_path):
    # Ottiene la frequenza di un file audio basato sull'estensione del file.
    try:
        audio = AudioSegment.from_file(file_path)  # Funziona per tutti i formati supportati da pydub
        return audio.frame_rate
    except Exception as e:
        print(f"Errore durante la decodifica del file {file_path}: {e}")
        return 0

def find_max_frequency(file_path, high_pass_filter=False, cutoff_freq=100.0, amplitude_threshold=0.001):
    # Carica l'audio con la frequenza di campionamento originale
    y, sr = librosa.load(file_path, sr=None)

    # Verifica se l'audio è silenzioso
    if np.all(y == 0):
        return 0.0, sr

    # Applica un filtro passa-alto se richiesto
    if high_pass_filter:
        y = librosa.effects.preemphasis(y, coef=cutoff_freq / sr)

    # Calcola lo spettrogramma con n_fft dinamico
    n_fft = min(len(y), 4096)
    S = np.abs(librosa.stft(y, n_fft=n_fft))

    # Calcola lo spettro medio (media lungo il tempo)
    S_mean = np.mean(S, axis=1)

    # Calcola le frequenze corrispondenti ai valori dello spettro
    frequencies = np.linspace(0, sr / 2, len(S_mean))

    # Trova la frequenza massima significativa (superiore alla soglia)
    threshold = amplitude_threshold * np.max(S_mean)
    significant_indices = np.where(S_mean > threshold)[0]

    if len(significant_indices) > 0:
        max_frequency = frequencies[significant_indices[-1]]
    else:
        max_frequency = frequencies[np.argmax(S_mean)]

    return round(max_frequency), sr

def conteggio_frequenze_campionamento():
    exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}
    current_file = os.path.abspath(__file__)
    dataset_folder = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    sampling_frequency_counter = {}

    total_files = sum(count_files(os.path.join(dataset_folder, subfolder), exclude_files) for subfolder in subfolders)
    file_count = 0

    for subfolder in subfolders:
        path_main = os.path.join(dataset_folder, subfolder)
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name in exclude_files:
                    continue
                file_path = os.path.join(root, file_name)
                try:
                    audio = AudioSegment.from_file(file_path)
                    sampling_frequency = audio.frame_rate
                    sampling_frequency_counter[sampling_frequency] = sampling_frequency_counter.get(sampling_frequency, 0) + 1
                    file_count += 1
                    sys.stdout.write(f"\rProgresso: {(file_count / total_files) * 100:.2f}%")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"Errore durante la decodifica del file {file_path}: {e}")

    sys.stdout.write('\n')  # Vai a capo una volta completato il processo

    return sampling_frequency_counter

def conteggio_massime_frequenze_riproduzione():
    exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}
    current_file = os.path.abspath(__file__)
    dataset_folder = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    max_frequency_counter = defaultdict(int)

    total_files = sum(count_files(os.path.join(dataset_folder, subfolder), exclude_files) for subfolder in subfolders)
    file_count = 0

    for subfolder in subfolders:
        path_main = os.path.join(dataset_folder, subfolder)
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name in exclude_files or not file_name.endswith(".wav"):
                    continue
                file_path = os.path.join(root, file_name)
                try:
                    max_frequency, sr = find_max_frequency(file_path, high_pass_filter=True, cutoff_freq=100.0, amplitude_threshold=0.001)

                    if max_frequency > 0:
                        max_frequency_counter[max_frequency] += 1

                    print(f"File: {file_name}, Posizione: {subfolder}, Frequenza di campionamento: {sr}, Massima frequenza di riproduzione: {max_frequency}")

                    file_count += 1
                    sys.stdout.write(f"\rProgresso: {(file_count / total_files) * 100:.2f}%")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"Errore durante la decodifica del file {file_path}: {e}")

    sys.stdout.write('\n')  # Vai a capo una volta completato il processo

    return max_frequency_counter

if __name__ == "__main__":
    # Eseguiamo i metodi per ottenere i dizionari con i conteggi delle frequenze
    # sampling_frequency_counter = conteggio_frequenze_campionamento()
    max_frequency_counter = conteggio_massime_frequenze_riproduzione()

    # Stampiamo i risultati per le massime frequenze di riproduzione
    print("\nMassime frequenze durante la riproduzione:")
    for frequency, count in sorted(max_frequency_counter.items()):
        print(f"Frequenza massima: {frequency} Hz, Numero di file: {count}")
