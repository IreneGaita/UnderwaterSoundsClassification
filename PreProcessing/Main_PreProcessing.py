import os
import sys
from caricamento import conversione_audio
from SetDurata import process_audio_files as segment_audio

def main():
    # Define dataset paths
    current_file = os.path.abspath(__file__)
    parent_folder = os.path.dirname(os.path.dirname(current_file))
    dataset_folder_path = os.path.join(parent_folder, "Dataset")
    new_dataset_folder_path = os.path.join(parent_folder, "NewDataset")
    output_base_path = os.path.join(parent_folder, "Scalograms")

    # Convert audio files
    print("Inizio PreProcessing")
    print("Conversione audio .mp3 a .wav")
    print("Cambio multi canale a mono")
    print("Set frequenza di campionamento a 86400 Hz")
    process_audio(dataset_folder_path)

    # Segment and cut audio files
    print("Proseguo con segmentazione e taglio a 3 secondi")
    segment_audio(dataset_folder_path, new_dataset_folder_path, segment_length=3)

    # Create Scalograms
    print("Creazione Scalogrammi")
    create_scalograms(new_dataset_folder_path, output_base_path)


def process_audio(dataset_folder):
    subfolders = ["Target", "Non-Target"]
    subfolder_paths = [os.path.join(dataset_folder, subfolder) for subfolder in subfolders]
    conversione_audio(subfolder_paths)


def create_scalograms(dataset_folder, output_base_path):
    subfolders = ["Target", "Non-Target"]
    subfolder_paths = [os.path.join(dataset_folder, subfolder) for subfolder in subfolders]

    if sys.platform == "darwin":
        from Scalogrammi_Mac import process_scalograms as process_scalograms_mac
        process_scalograms_mac(subfolder_paths, output_base_path)
    else:
        from Scalogrammi import processing_scalograms as process_scalograms_win
        process_scalograms_win(subfolder_paths, output_base_path)


if __name__ == "__main__":
    main()
