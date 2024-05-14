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

def frequenze():
    exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}
    current_file = os.path.abspath(__file__)
    dataset_folder = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    total_files = sum(count_files(os.path.join(dataset_folder, subfolder), exclude_files) for subfolder in subfolders)
    file_count = 0
    frequency_counter = {}

    for subfolder in subfolders:
        path_main = os.path.join(dataset_folder, subfolder)
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name in exclude_files:
                    continue
                file_path = os.path.join(root, file_name)
                frequency = get_audio_frequency(file_path)
                if frequency:
                    frequency_counter[frequency] = frequency_counter.get(frequency, 0) + 1
                file_count += 1
                sys.stdout.write(f"\rProgresso: {(file_count / total_files) * 100:.2f}%")
                sys.stdout.flush()

    sys.stdout.write('\n')  # Vai a capo una volta completato il processo
    print("Frequenze audio lette con successo.\nTotale file audio:", file_count)
    return frequency_counter

if __name__ == "__main__":
    frequencies = frequenze()
    print("Lista delle frequenze con il conteggio dei file per ogni frequenza:")
    for frequency, count in frequencies.items():
        print(f"Frequenza: {frequency}, Numero di file: {count}")