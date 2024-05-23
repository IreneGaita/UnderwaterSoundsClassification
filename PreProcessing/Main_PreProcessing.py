import os
from caricamento import process_audio_files
from SetDurata import process_audio_files

def main():
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]
    subfolder_paths = [os.path.join(dataset_folder_path, subfolder) for subfolder in subfolders]

    print("Inizio PreProcessing")
    print("Conversione audio .mp3 a .wav")
    print("Cambio multi canale a mono")
    print("set frequenza di campionamento a 86400 Hz")
    process_audio_files(subfolder_paths)
    print("Proseguo con segmentazione e taglio a 3 secondi")
    current_file = os.path.abspath(__file__)
    parent_folder = os.path.dirname(os.path.dirname(current_file))
    input_dir = os.path.join(parent_folder, "Dataset")
    output_dir = os.path.join(parent_folder, "NewDataset")
    os.makedirs(output_dir, exist_ok=True)
    segment_length = 3  # segment length in seconds
    process_audio_files(input_dir, output_dir, segment_length)
    print("Creazione Scalogrammi")
    Scalogrammi()