import os
from pydub import AudioSegment

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
    exclude_files = ['.DS_Store', 'metadata-Target.csv']  # Lista di file da escludere

    # Ottieni il percorso assoluto del file corrente (dove viene eseguito il codice)
    current_file = os.path.abspath(__file__)

    # Ottieni il percorso della cartella genitore del file corrente
    parent_folder = os.path.dirname(current_file)
    parent_folder = os.path.dirname(parent_folder)  # Sali di un livello nella struttura delle directory

    # Definisci il percorso della cartella "Dataset"
    dataset_folder_path = os.path.join(parent_folder, "Dataset")

    # Definisci le sottocartelle da visitare
    subfolders = ["Non-Target", "Target"]

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
                        file_count += 1
                        print(f"File letto: {file_count} - Nome: {file_name}, Durata: {duration[0]} min {duration[1]} sec")
                    else:
                        print(f"Errore durante l'ottenimento delle informazioni audio per il file {file_path}")

    print("Informazioni audio lette con successo.")
    print("Totale file audio:", file_count)
    return audio_info_dict

if __name__ == "__main__":
    audio_info_dict = audio_info()
    print("Dizionario delle informazioni audio:")
    for file_name, duration in audio_info_dict.items():
        print(f"File: {file_name}, Durata: {duration[0]} minuti e {duration[1]} secondi")
