import os
import sys
import librosa
import soundfile as sf

def get_audio_info(file_path):
    try:
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.mp3':
            return None

        audio, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        minutes, seconds = divmod(int(duration), 60)

        if ext.lower() == '.wav':
            with sf.SoundFile(file_path) as f:
                bit_depth = f.subtype_info
            return (minutes, seconds, bit_depth)
        else:
            return (minutes, seconds, None)
    except Exception as e:
        print(f"Errore durante l'ottenimento delle informazioni audio per il file {file_path}: {e}")
        return None

def count_files(directory, exclude_files, include_mp3=False):
    return sum(1 for root, _, files in os.walk(directory) for file in files if file not in exclude_files and (include_mp3 or not file.lower().endswith('.mp3')))

def audio_info():
    exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    total_files = sum(count_files(os.path.join(dataset_folder_path, subfolder), exclude_files, include_mp3=False) for subfolder in subfolders)
    file_count = 0
    mp3_count = 0
    bit_depth_count = {}

    for subfolder in subfolders:
        path_main = os.path.join(dataset_folder_path, subfolder)
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name not in exclude_files:
                    file_path = os.path.join(root, file_name)
                    if file_name.lower().endswith('.mp3'):
                        mp3_count += 1
                        continue
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
    print("Totale file audio (.mp3):", mp3_count)

    print("\nContatore per ogni Bit Depth differente:")
    order = [
        "Unsigned 8 bit PCM",
        "Signed 16 bit PCM",
        "Signed 24 bit PCM",
        "32 bit float",
        "Signed 32 bit PCM"
    ]

    for bit_depth in order:
        if bit_depth in bit_depth_count:
            print(f"Bit Depth {bit_depth} bit: {bit_depth_count[bit_depth]} file")

    return bit_depth_count

if __name__ == "__main__":
    bit_depth_count = audio_info()
