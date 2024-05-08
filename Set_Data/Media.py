from pydub import AudioSegment
import os


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
            raise ValueError("Formato audio non supportato: {file_extension}")

        duration = len(audio) / 1000  # Durata in secondi
        return duration
    except Exception as e:
        print(f"Errore durante la decodifica del file {file_path}: {e}")
        return 0


def media(folder_path):
    file_names = os.listdir(folder_path)

    total_length = 0
    file_count = 0
    for file_name in file_names:
        path_audio = os.path.join(folder_path, file_name)
        Lista_audio = os.listdir(path_audio)
        for audio_file in Lista_audio:
                file_path = os.path.join(path_audio, audio_file)
                print(file_path)
                total_length += get_audio_length(file_path)
                file_count += 1
    average_length = total_length / file_count if file_count > 0 else 0
    print("Lunghezza media dei file .wav nelle cartelle:", average_length, "secondi")
    print('File totali:', file_count)
    Lista_interi = [int(average_length), file_count]
    return Lista_interi