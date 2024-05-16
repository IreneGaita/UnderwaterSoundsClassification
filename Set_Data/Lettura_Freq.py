import os
from pydub import AudioSegment
import sys

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

    print("Analisi delle frequenze di campionamento completata.\nTotale file audio:", file_count)
    return sampling_frequency_counter


def conteggio_massime_frequenze_riproduzione():
    exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}
    current_file = os.path.abspath(__file__)
    dataset_folder = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    max_frequency_counter = {}

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
                    max_amplitude = audio.max
                    max_frequency = round(
                        max_amplitude / 1000)  # Calcoliamo la massima frequenza relativa all'ampiezza massima dell'audio
                    max_frequency_counter[max_frequency] = max_frequency_counter.get(max_frequency, 0) + 1

                    # Stampiamo le informazioni richieste
                    print(
                        f"File: {file_name}, Posizione: {subfolder}, Frequenza di campionamento: {audio.frame_rate}, Massima frequenza di riproduzione: {max_frequency}")

                    file_count += 1
                    sys.stdout.write(f"\rProgresso: {(file_count / total_files) * 100:.2f}%")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"Errore durante la decodifica del file {file_path}: {e}")

    sys.stdout.write('\n')  # Vai a capo una volta completato il processo

    print("Analisi delle massime frequenze di riproduzione completata.\nTotale file audio:", file_count)
    return max_frequency_counter


if __name__ == "__main__":
    # Eseguiamo i metodi per ottenere i dizionari con i conteggi delle frequenze
    max_frequency_counter = conteggio_massime_frequenze_riproduzione()

    # Stampiamo i risultati per le massime frequenze di riproduzione
    print("\nMassime frequenze durante la riproduzione:")
    for frequency, count in max_frequency_counter.items():
        print(f"Frequenza massima: {frequency} Hz, Numero di file: {count}")
