from pydub import AudioSegment
import os

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


def get_audio_frequency(file_path):
    try:
        # Controlla l'estensione del file
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == '.wav':
            audio = AudioSegment.from_wav(file_path)
        elif file_extension.lower() == '.mp3':
            # Converti il file MP3 in formato WAV
            audio = AudioSegment.from_mp3(file_path)
        else:
            raise ValueError(f"Formato audio non supportato: {file_extension}")

        frequency = audio.frame_rate  # Frequenza dell'audio
        return frequency
    except Exception as e:
        print(f"Errore durante la decodifica del file {file_path}: {e}")
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

    # Percorsi delle cartelle "Target" e "Non-Target" all'interno di "Dataset"
    folders = ["Target", "Non-Target"]

    for folder in folders:
        path_main = os.path.join(parent_folder, "Dataset", folder)

        # Scorri tutti i file nella cartella target
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name not in exclude_files:  # Verifica se il file deve essere escluso
                    file_path = os.path.join(root, file_name)
                    duration = get_audio_info(file_path)
                    if duration is not None:
                        if file_name in audio_info_dict:
                            print("Duplicato trovato!")
                        # Salva le informazioni audio nel dizionario
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
        print(f"File: {file_name}, Durata: {duration}")

