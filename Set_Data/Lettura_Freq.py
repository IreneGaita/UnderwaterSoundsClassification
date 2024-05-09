import os
from pydub import AudioSegment
import os
from pydub import AudioSegment

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
        return 0


def frequenze():
    file_count = 0
    frequency_counter = {}  # Dizionario per tenere traccia delle frequenze dei file audio

    exclude_files = ['.DS_Store', 'metadata-Target.csv']  # Lista di file da escludere

    # Ottieni il percorso assoluto del file corrente (dove viene eseguito il codice)
    current_file = os.path.abspath(__file__)

    # Ottieni il percorso della cartella genitore del file corrente
    parent_folder = os.path.dirname(current_file)
    parent_folder = os.path.dirname(parent_folder)  # Sali di un livello nella struttura delle directory

    # Definisci il percorso della cartella "Dataset"
    dataset_folder_path = os.path.join(parent_folder, "Dataset")

    # Percorso della cartella target all'interno di "Dataset"
    path_main = os.path.join(dataset_folder_path, "Target")

    # Scorri tutti i file nella cartella target
    for root, _, files in os.walk(path_main):
        for file_name in files:
            if file_name not in exclude_files:  # Verifica se il file deve essere escluso
                file_path = os.path.join(root, file_name)
                frequency = get_audio_frequency(file_path)
                if frequency != 0:
                    # Incrementa il contatore per la frequenza corrente nel dizionario
                    frequency_counter[frequency] = frequency_counter.get(frequency, 0) + 1
                    file_count += 1
                    print(f"File letto: {file_count}")

    print("Frequenze audio lette con successo.")
    print("Totale file audio:", file_count)
    return frequency_counter

if __name__ == "__main__":
    frequencies = frequenze()
    print("Lista delle frequenze con il conteggio dei file per ogni frequenza:")
    for frequency, count in frequencies.items():
        print(f"Frequenza: {frequency}, Numero di file: {count}")
