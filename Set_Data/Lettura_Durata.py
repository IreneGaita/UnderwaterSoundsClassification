import os
from pydub import AudioSegment
import sys

def count_files(directory, exclude_files):
    # Conta i file totali escludendo quelli nella lista exclude_files.
    return sum(1 for root, _, files in os.walk(directory) for file in files if file not in exclude_files)

def get_audio_info(file_path):
    # Ottiene informazioni sulla durata di un file audio e restituisce una tupla (minuti, secondi).
    try:
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        return divmod(duration_ms // 1000, 60)
    except Exception as e:
        print(f"Errore durante l'ottenimento delle informazioni audio per il file {file_path}: {e}")
        return None

def audio_info():
    exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Non-Target"]

    total_files = sum(count_files(os.path.join(dataset_folder_path, subfolder), exclude_files) for subfolder in subfolders)
    file_count = 0
    audio_info_dict = {}

    for subfolder in subfolders:
        path_main = os.path.join(dataset_folder_path, subfolder)
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name not in exclude_files:
                    file_path = os.path.join(root, file_name)
                    duration = get_audio_info(file_path)
                    if duration:
                        if file_name not in audio_info_dict:
                            audio_info_dict[file_name] = []
                        audio_info_dict[file_name].append(duration)
                        file_count += 1
                        sys.stdout.write(f"\rProgresso: {(file_count / total_files) * 100:.2f}%")
                        sys.stdout.flush()
                    else:
                        print(f"Errore durante l'ottenimento delle informazioni audio per il file {file_path}")

    sys.stdout.write('\n')  # Vai a capo una volta completato il processo
    print("Informazioni audio lette con successo.\nTotale file audio:", file_count)
    return audio_info_dict

if __name__ == "__main__":
    audio_info_dict = audio_info()
    print("Dizionario delle informazioni audio:")
    for file_name, durations in audio_info_dict.items():
        for duration in durations:
            print(f"File: {file_name}, Durata: {duration[0]} min {duration[1]} sec")