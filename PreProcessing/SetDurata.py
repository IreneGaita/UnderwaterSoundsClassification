import os
import librosa
import soundfile as sf
import numpy as np
import sys


def adjust_audio_length(y, sr, target_length):
    current_length = len(y) / sr  # current length in seconds

    if current_length < target_length:
        while len(y) / sr < target_length:
            start = np.random.randint(0, len(y) - 1)
            end = min(start + int(target_length * sr), len(y))
            y = np.concatenate([y, y[start:end]])
        y = y[:int(target_length * sr)]
    return y


def segment_audio(y, sr, segment_length):
    segments = []
    for start in range(0, len(y), segment_length * sr):
        end = min(start + segment_length * sr, len(y))
        segment = y[start:end]
        if len(segment) < segment_length * sr:
            # Ignora la parte in eccesso se è inferiore a 1 secondo
            if len(segment) < sr:
                break
            segment = adjust_audio_length(segment, sr, segment_length)
        segments.append(segment)
    return segments


def process_audio_files(input_dir, output_dir, segment_length):
    total_files = sum([len(files) for _, _, files in os.walk(input_dir)])
    processed_files = 0
    last_printed_progress = -1

    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if not file_name.endswith('.wav'):
                continue
            file_path = os.path.join(root, file_name)
            output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))
            os.makedirs(output_subdir, exist_ok=True)

            try:
                y, sr = librosa.load(file_path, sr=None)
                segments = segment_audio(y, sr, segment_length)
                base_name, ext = os.path.splitext(file_name)
                for idx, segment in enumerate(segments):
                    output_file_path = os.path.join(output_subdir, f"{base_name}_part{idx}{ext}")
                    sf.write(output_file_path, segment, sr)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

            processed_files += 1
            progress = (processed_files / total_files) * 100
            if int(progress) > last_printed_progress:
                last_printed_progress = int(progress)
                sys.stdout.write(f"\rProgresso: {progress:.2f}%")
                sys.stdout.flush()


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    parent_folder = os.path.dirname(os.path.dirname(current_file))
    dataset_folder_path = os.path.join(parent_folder, "Dataset")
    subfolders = ["Target", "Non-Target"]

    new_dataset_folder_path = os.path.join(parent_folder, "NewDataset")
    os.makedirs(new_dataset_folder_path, exist_ok=True)

    segment_length = 3  # segment length in seconds

    for subfolder in subfolders:
        input_subfolder_path = os.path.join(dataset_folder_path, subfolder)
        output_subfolder_path = os.path.join(new_dataset_folder_path, subfolder)
        process_audio_files(input_subfolder_path, output_subfolder_path, segment_length)

    sys.stdout.write("\nElaborazione completata.\n")