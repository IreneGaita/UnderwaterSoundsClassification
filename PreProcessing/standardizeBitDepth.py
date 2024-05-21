import os
import soundfile as sf


def standardize_bit_depth_to_16bit(directory):
    # Itera attraverso tutti i file nelle directory specificate
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                try:
                    # Leggi il file WAV
                    data, sr = sf.read(file_path)

                    # Salva il file audio come WAV con bit depth di 16 bit
                    sf.write(file_path, data, sr, subtype='PCM_16')
                    print(f"Standardized {file_path} to 16 bit")
                except Exception as e:
                    print(f"Error standardizing file {file_path}: {e}")


if __name__ == "__main__":
    # Percorso della cartella "Target" e "Non-Target"
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    for subfolder in subfolders:
        dataset_path = os.path.join(dataset_folder_path, subfolder)
        # Standardizza i file WAV a 16 bit
        standardize_bit_depth_to_16bit(dataset_path)
