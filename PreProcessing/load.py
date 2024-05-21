import os
import librosa
import soundfile as sf
import sys

# Definisci i file da escludere
exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}

def process_audio_files(directory):
    # Conta il numero totale di file da processare
    total_files = sum(len(files) for _, _, files in os.walk(directory) if not exclude_files.intersection(files))
    processed_files = 0
    last_printed_progress = -1

    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Skip excluded files
            if file_name in exclude_files:
                processed_files += 1
                continue

            try:
                if file_name.endswith(".mp3"):
                    # Load MP3 file using librosa
                    y, sr = librosa.load(file_path, sr=None, mono=True)

                    # Convert to WAV format
                    wav_file_path = file_path.replace(".mp3", ".wav")
                    sf.write(wav_file_path, y, sr, subtype='PCM_16')
                    os.remove(file_path)
                else:
                    # Load WAV file (or other formats) using librosa
                    y, sr = librosa.load(file_path, sr=None, mono=True)

                    # Standardize bit depth to 16-bit
                    sf.write(file_path, y, sr, subtype='PCM_16')

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

            # Aggiorna il progresso
            processed_files += 1
            progress = (processed_files / total_files) * 100
            if int(progress) > last_printed_progress:
                last_printed_progress = int(progress)
                sys.stdout.write(f"\rProgresso: {progress:.2f}%")
                sys.stdout.flush()

if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    for subfolder in subfolders:
        subfolder_path = os.path.join(dataset_folder_path, subfolder)
        process_audio_files(subfolder_path)

    sys.stdout.write("\nElaborazione completata.\n")
