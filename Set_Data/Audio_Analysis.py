import os
from pydub import AudioSegment
import sys

def analyze_audio_channels(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        return audio.channels
    except Exception as e:
        print(f"Error analyzing file {file_path}: {e}")
        return None

def count_files(directory, exclude_files):
    return sum(1 for root, _, files in os.walk(directory) for file in files if file not in exclude_files)

def analyze_audio_files():
    exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}
    current_file = os.path.abspath(__file__)
    dataset_folder = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    total_files = sum(count_files(os.path.join(dataset_folder, subfolder), exclude_files) for subfolder in subfolders)

    mono_count = 0
    multi_count = 0
    multi_channel_distribution = {}
    file_info = []
    file_count = 0

    for subfolder in subfolders:
        path_main = os.path.join(dataset_folder, subfolder)
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name in exclude_files:
                    continue
                file_path = os.path.join(root, file_name)
                channels = analyze_audio_channels(file_path)
                if channels is not None:
                    folder_name = os.path.relpath(root, dataset_folder)
                    if channels == 1:
                        mono_count += 1
                        channel_type = 'Mono'
                    else:
                        multi_count += 1
                        channel_type = 'Multichannel'
                        multi_channel_distribution[channels] = multi_channel_distribution.get(channels, 0) + 1

                    file_info.append((file_name, channels, channel_type, folder_name))
                    file_count += 1

                    sys.stdout.write(f"\rProgresso: {(file_count / total_files) * 100:.2f}%")
                    sys.stdout.flush()
                else:
                    print(f"File: {file_name}, Canali non specificati")

    sys.stdout.write('\n')

    print(f"Totale audio monocanale: {mono_count}")
    print(f"Totale audio multicanale: {multi_count}")
    for channels, count in multi_channel_distribution.items():
        print(f"Audio con {channels} canali: {count}")

    return file_info

if __name__ == "__main__":
    audio_file_info = analyze_audio_files()
