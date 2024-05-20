import os
import librosa
import soundfile as sf


def resample_audio(file_path, new_rate):
    try:
        print(f"Processing file: {file_path}")
        # Carica l'audio utilizzando librosa
        y, original_rate = librosa.load(file_path, sr=None)
        print(f"Frequenza di campionamento originale: {original_rate} Hz")

        # Ricampionamento
        y_resampled = librosa.resample(y, orig_sr=original_rate, target_sr=new_rate)

        # Salva il file ricampionato, sovrascrivendo il file originale
        sf.write(file_path, y_resampled, new_rate)
        print(f"File audio ricampionato e salvato come: {file_path}")

        return file_path

    except Exception as e:
        print(f"Errore durante la decodifica o il ricampionamento del file {file_path}: {e}")
        return None


def resample_audio_dataset(dataset_path, new_rate):
    try:
        total_files = 0
        processed_files = 0

        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                if file_name.endswith(".wav"):
                    total_files += 1

        print(f"Total number of .wav files in dataset: {total_files}")

        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                if file_name.endswith(".wav"):
                    file_path = os.path.join(root, file_name)
                    result = resample_audio(file_path, new_rate)
                    if result:
                        processed_files += 1
                    else:
                        print(f"Failed to process file: {file_path}")

        print(f"Number of files successfully processed: {processed_files}")
        print(f"Number of files failed to process: {total_files - processed_files}")

    except Exception as e:
        print(f"Errore durante l'elaborazione del dataset: {e}")


if __name__ == "__main__":
    dataset_path = '/Users/irene.gaita/PycharmProjects/UnderwaterSoundsClassification/Dataset/Target'
    new_rate = 86400
    resample_audio_dataset(dataset_path, new_rate)