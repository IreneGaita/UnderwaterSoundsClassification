import os
import librosa
import soundfile as sf


def convert_multichannel_to_mono(directory):
    # Itera attraverso tutti i file nelle directory specificate
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                try:
                    # Carica il file WAV
                    y, sr = librosa.load(file_path, sr=None, mono=False)

                    # Controlla se il file è multicanale
                    if len(y.shape) > 1:
                        # Converte a mono
                        y_mono = librosa.to_mono(y)

                        # Salva il file audio come WAV con bit depth di 16 bit
                        sf.write(file_path, y_mono, sr, subtype='PCM_16')
                        print(f"Converted {file_path} to mono")
                except Exception as e:
                    print(f"Error converting file {file_path}: {e}")


if __name__ == "__main__":
    # Percorso della cartella "Target" e "Non-Target"
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    for subfolder in subfolders:
        dataset_path = os.path.join(dataset_folder_path, subfolder)
        # Converti i file WAV multicanale in mono
        convert_multichannel_to_mono(dataset_path)
