import os
import sys
from pydub import AudioSegment
import numpy as np

def count_files(directory, exclude_files):
    # Conta i file totali escludendo quelli nella lista exclude_files.
    return sum(1 for root, _, files in os.walk(directory) for file in files if file not in exclude_files)

def get_audio_info(file_path):
    try:
        # Carica il file audio utilizzando pydub
        audio = AudioSegment.from_file(file_path)

        # Ottieni la durata dell'audio in millisecondi
        duration_ms = len(audio)

        # Converti la durata da millisecondi a minuti e secondi
        duration_min = duration_ms // 1000 // 60
        duration_sec = duration_ms // 1000 % 60

        return duration_min, duration_sec
    except Exception as e:
        print(f"Errore durante l'ottenimento delle informazioni audio per il file {file_path}: {e}")
        return None

def audio_info():
    file_count = 0
    audio_info_dict = {}  # Dizionario per tenere traccia delle informazioni audio per ogni file
    durations = []  # Lista per tenere traccia di tutte le durate dei file audio
    exclude_files = ['.DS_Store', 'metadata-Target.csv','metadata-NonTarget.csv']  # Lista di file da escludere

    # Ottieni il percorso assoluto del file corrente (dove viene eseguito il codice)
    current_file = os.path.abspath(__file__)

    # Ottieni il percorso della cartella genitore del file corrente
    parent_folder = os.path.dirname(current_file)
    parent_folder = os.path.dirname(parent_folder)  # Sali di un livello nella struttura delle directory

    # Definisci il percorso della cartella "Dataset"
    dataset_folder_path = os.path.join(parent_folder, "Dataset")

    # Definisci le sottocartelle da visitare
    subfolders = ["Target", "Non-Target"]
    total_files = sum(count_files(os.path.join(dataset_folder_path, subfolder), exclude_files) for subfolder in subfolders)

    # Percorri tutte le sottocartelle
    for subfolder in subfolders:
        # Percorso della sottocartella corrente
        path_main = os.path.join(dataset_folder_path, subfolder)

        # Scorri tutti i file nella sottocartella corrente
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name not in exclude_files:  # Verifica se il file deve essere escluso
                    file_path = os.path.join(root, file_name)
                    duration = get_audio_info(file_path)
                    if duration is not None:
                        audio_info_dict[file_name] = duration
                        durations.append(duration[0] * 60 + duration[1])  # Converti la durata in secondi
                        file_count += 1
                        sys.stdout.write(f"\rProgresso: {(file_count / total_files) * 100:.2f}%")
                        sys.stdout.flush()

                    else:
                        print(f"Errore durante l'ottenimento delle informazioni audio per il file {file_path}")
    sys.stdout.write('\n')  # Vai a capo una volta completato il processo

    # Calcola la mediana delle durate
    mediana = np.median(durations) if durations else None
    mediana_min, mediana_sec = divmod(int(mediana), 60) if mediana is not None else (None, None)

    print("Informazioni audio lette con successo.")
    print("Totale file audio:", file_count)
    if mediana is not None:
        print(f"La mediana delle durate dei file audio è: {mediana_min} minuti e {mediana_sec} secondi")
    else:
        print("Non ci sono durate disponibili per calcolare la mediana.")

    return audio_info_dict

if __name__ == "__main__":
    audio_info_dict = audio_info()
