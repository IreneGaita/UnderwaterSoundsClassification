import os
import wave
from pydub import AudioSegment
import sys

def get_audio_info(file_path):
    try:
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.mp3':
            return None

        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        minutes, seconds = divmod(duration_ms // 1000, 60)

        if ext.lower() == '.wav':
            with wave.open(file_path, 'rb') as wav_file:
                bit_depth = wav_file.getsampwidth() * 8
            return (minutes, seconds, bit_depth)
        else:
            return (minutes, seconds, None)
    except Exception as e:
        print(f"Errore durante l'ottenimento delle informazioni audio per il file {file_path}: {e}")
        return None

def count_files(directory, exclude_files):
    return sum(1 for root, _, files in os.walk(directory) for file in files if file not in exclude_files and not file.lower().endswith('.mp3'))

def audio_info():
    exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Non-Target"]

    total_files = sum(count_files(os.path.join(dataset_folder_path, subfolder), exclude_files) for subfolder in subfolders)
    file_count = 0
    bit_depth_count = {}

    for subfolder in subfolders:
        path_main = os.path.join(dataset_folder_path, subfolder)
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name not in exclude_files and not file_name.lower().endswith('.mp3'):
                    file_path = os.path.join(root, file_name)
                    info = get_audio_info(file_path)
                    if info:
                        minutes, seconds, bit_depth = info
                        file_count += 1

                        if bit_depth is not None:
                            if bit_depth not in bit_depth_count:
                                bit_depth_count[bit_depth] = 0
                            bit_depth_count[bit_depth] += 1

                    sys.stdout.write(f"\rProgresso: {(file_count / total_files) * 100:.2f}%")
                    sys.stdout.flush()

    sys.stdout.write('\n')
    print("Totale file audio (.wav):", file_count)
    return bit_depth_count

if __name__ == "__main__":
    bit_depth_count = audio_info()

    print("\nContatore per ogni Bit Depth differente:")
    for bit_depth, count in bit_depth_count.items():
        print(f"Bit Depth {bit_depth} bit: {count} file")