import os

def load_audio_files():
    exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}
    current_file = os.path.abspath(__file__)
    dataset_folder = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    audio_files = []

    total_files = sum(1 for subfolder in subfolders for _, _, files in os.walk(os.path.join(dataset_folder, subfolder)) for file_name in files if file_name not in exclude_files)
    loaded_files = 0

    for subfolder in subfolders:
        path_main = os.path.join(dataset_folder, subfolder)
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name in exclude_files or not (file_name.endswith(".wav") or file_name.endswith(".mp3")):
                    continue
                file_path = os.path.join(root, file_name)
                audio_files.append(file_path)
                loaded_files += 1
                percentage = (loaded_files / total_files) * 100
                print(f"Progresso: {percentage:.2f}% completato", end="\r")

    print("\nCaricamento completato.")
    return audio_files
