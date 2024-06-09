import os
import sys
import librosa
import numpy as np


def count_files(directory, exclude_files):
    # Count total files excluding those in the exclude_files list.
    return sum(1 for root, _, files in os.walk(directory) for file in files if file not in exclude_files)


def get_audio_info(file_path):
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_path, sr=None)

        # Get the duration of the audio in seconds
        duration_sec = librosa.get_duration(y=y, sr=sr)

        # Convert the duration to minutes and seconds
        duration_min = int(duration_sec // 60)
        duration_sec = int(duration_sec % 60)

        return duration_min, duration_sec
    except Exception as e:
        print(f"Errore durante l'ottenimento delle informazioni audio per il file {file_path}: {e}")
        return None


def audio_info():
    file_count = 0
    audio_info_dict = {}  # Dictionary to track audio info for each file
    durations = []  # List to track durations of all audio files
    exclude_files = ['.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv']  # List of files to exclude

    # Get the absolute path of the current file (where the code is executed)
    current_file = os.path.abspath(__file__)

    # Get the parent folder path of the current file
    parent_folder = os.path.dirname(current_file)
    parent_folder = os.path.dirname(parent_folder)  # Go up one level in the directory structure

    # Define the path to the "Dataset" folder
    dataset_folder_path = os.path.join(parent_folder, "Dataset")

    # Define the subfolders to visit
    subfolders = ["Target", "Non-Target"]
    total_files = sum(
        count_files(os.path.join(dataset_folder_path, subfolder), exclude_files) for subfolder in subfolders)

    # Traverse all subfolders
    for subfolder in subfolders:
        # Path of the current subfolder
        path_main = os.path.join(dataset_folder_path, subfolder)

        # Iterate through all files in the current subfolder
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name not in exclude_files:  # Check if the file should be excluded
                    file_path = os.path.join(root, file_name)
                    duration = get_audio_info(file_path)
                    if duration is not None:
                        audio_info_dict[file_name] = duration
                        durations.append(duration[0] * 60 + duration[1])  # Convert duration to seconds
                        file_count += 1
                        sys.stdout.write(f"\rProgresso: {(file_count / total_files) * 100:.2f}%")
                        sys.stdout.flush()
                    else:
                        print(f"Errore durante l'ottenimento delle informazioni audio per il file {file_path}")
    sys.stdout.write('\n')  # Move to the next line once the process is completed

    # Calculate the median duration
    mediana = np.median(durations) if durations else None
    mediana_min, mediana_sec = divmod(int(mediana), 60) if mediana is not None else (None, None)

    print("Informazioni audio lette con successo.")
    print("Totale file audio:", file_count)
    if mediana is not None:
        print(f"La mediana delle durate dei file audio è: {mediana_min} minuti e {mediana_sec} secondi")
    else:
        print("Non ci sono durate disponibili per calcolare la mediana.")

    return audio_info_dict


if __name__ == "__main__":
    audio_info_dict = audio_info()