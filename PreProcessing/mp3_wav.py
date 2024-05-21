import os
import librosa
import soundfile as sf

# Definisci i file da escludere
exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}

def convert_mp3_to_wav(directory):
    # Itera attraverso tutti i file nella directory
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".mp3") and file_name not in exclude_files:
                file_path = os.path.join(root, file_name)
                try:
                    # Carica il file MP3
                    y, sr = librosa.load(file_path, sr=None)
                    # Costruisci il nuovo percorso del file WAV
                    wav_file_path = file_path.replace(".mp3", ".wav")
                    # Salva il file audio come WAV
                    sf.write(wav_file_path, y, sr)
                    print(f"Converted {file_path} to {wav_file_path}")
                    # Rimuovi il file MP3 originale
                    os.remove(file_path)
                    print(f"Removed original MP3 file: {file_path}")
                except Exception as e:
                    print(f"Error converting file {file_path}: {e}")

if __name__ == "__main__":
    # Percorso del file corrente
    current_file = os.path.abspath(__file__)
    # Percorso della cartella "Dataset"
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    # Sottocartelle da processare
    subfolders = ["Target", "Non-Target"]

    # Itera attraverso le sottocartelle e converte i file
    for subfolder in subfolders:
        subfolder_path = os.path.join(dataset_folder_path, subfolder)
        convert_mp3_to_wav(subfolder_path)
