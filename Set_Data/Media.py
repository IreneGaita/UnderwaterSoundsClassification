import os
from pydub import AudioSegment

def get_audio_length(file_path):
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

        duration = len(audio) / 1000  # Durata in secondi
        return duration
    except Exception as e:
        print(f"Errore durante la decodifica del file {file_path}: {e}")
        return 0

def media(folder_path):
    total_length = 0
    file_count = 0

    # Itera su tutte le sottodirectory e i file all'interno del percorso specificato
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs[:]:  # Copia della lista per evitare modifiche durante l'iterazione
            if not os.listdir(os.path.join(root, dir_name)):
                # Se una directory è vuota, la rimuovi dalla lista delle directory
                dirs.remove(dir_name)

        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Controlla se il file è un formato audio supportato
            _, file_extension = os.path.splitext(file_name)
            if file_extension.lower() not in ['.wav', '.mp3']:
                print(f"File non supportato: {file_path}")
                continue

            # Calcola la durata solo se il file è un formato audio supportato
            duration = get_audio_length(file_path)
            total_length += duration
            file_count += 1

            # Stampa il percorso dell'audio, il nome del file, la durata e il numero di file audio letti
            print(f"Percorso: {file_path}, Nome file: {file_name}, Durata: {duration} secondi")

    average_length = total_length / file_count if file_count > 0 else 0
    print("Lunghezza media dei file .wav nelle cartelle:", average_length, "secondi")
    print('File totali:', file_count)
    Lista_interi = [int(average_length), file_count]
    return Lista_interi
