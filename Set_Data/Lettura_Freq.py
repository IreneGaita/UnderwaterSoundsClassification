import os
import sys
import librosa
import numpy as np
from collections import defaultdict
from pydub import AudioSegment


def count_files(directory, exclude_files):
    # Count total files excluding those in the exclude_files list.
    return sum(1 for root, _, files in os.walk(directory) for file in files if file not in exclude_files)


def get_audio_frequency(file_path):
    # Get the sampling rate of an audio file.
    try:
        audio = AudioSegment.from_file(file_path)  # Works for all formats supported by pydub
        return audio.frame_rate
    except Exception as e:
        print(f"Error decoding file {file_path}: {e}")
        return 0


def find_max_frequency(file_path, high_pass_filter=False, cutoff_freq=100.0, amplitude_threshold=0.001):
    # Load the audio with the original sampling rate
    y, sr = librosa.load(file_path, sr=None)

    # Check if the audio is silent
    if np.all(y == 0):
        return 0.0, sr

    # Compute the spectrogram with dynamic n_fft
    n_fft = min(len(y), 4096)
    S = np.abs(librosa.stft(y, n_fft=n_fft))

    # Compute the average spectrum (average along time)
    S_mean = np.mean(S, axis=1)

    # Compute the frequencies corresponding to the spectrum values
    frequencies = np.linspace(0, sr / 2, len(S_mean))

    # Find the significant maximum frequency (above threshold)
    threshold = amplitude_threshold * np.max(S_mean)
    significant_indices = np.where(S_mean > threshold)[0]

    if len(significant_indices) > 0:
        max_frequency = frequencies[significant_indices[-1]]
    else:
        max_frequency = frequencies[np.argmax(S_mean)]

    return round(max_frequency), sr


def analyze_audio_files():
    exclude_files = {'.DS_Store', 'metadata-Target.csv', 'metadata-NonTarget.csv'}
    current_file = os.path.abspath(__file__)
    dataset_folder = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Dataset")
    subfolders = ["Target", "Non-Target"]

    sampling_frequency_counter = defaultdict(int)
    max_frequency_counter = defaultdict(int)

    total_files = sum(count_files(os.path.join(dataset_folder, subfolder), exclude_files) for subfolder in subfolders)
    file_count = 0

    for subfolder in subfolders:
        path_main = os.path.join(dataset_folder, subfolder)
        for root, _, files in os.walk(path_main):
            for file_name in files:
                if file_name in exclude_files or not (file_name.endswith(".wav") or file_name.endswith(".mp3")):
                    continue
                file_path = os.path.join(root, file_name)
                try:
                    # Get sampling frequency
                    sampling_frequency = get_audio_frequency(file_path)
                    if sampling_frequency > 0:
                        sampling_frequency_counter[sampling_frequency] += 1

                    # Get maximum frequency
                    max_frequency, sr = find_max_frequency(file_path, high_pass_filter=True, cutoff_freq=100.0,
                                                           amplitude_threshold=0.001)
                    if max_frequency > 0:
                        max_frequency_counter[max_frequency] += 1

                    file_count += 1
                    sys.stdout.write(f"\rProgress: {(file_count / total_files) * 100:.2f}%")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"Error decoding file {file_path}: {e}")

    sys.stdout.write('\n')  # Move to the next line once the process is completed

    return sampling_frequency_counter, max_frequency_counter


if __name__ == "__main__":
    # Execute the method to get the dictionary with frequency counts
    sampling_frequency_counter, max_frequency_counter = analyze_audio_files()

    # Print the results for sampling frequencies
    print("\nSampling Frequencies:")
    for frequency, count in sorted(sampling_frequency_counter.items()):
        print(f"Sampling Frequency: {frequency} Hz, Number of files: {count}")

    # Print the results for the maximum frequencies during playback
    print("\nMaximum Frequencies during playback:")
    for frequency, count in sorted(max_frequency_counter.items()):
        print(f"Maximum Frequency: {frequency} Hz, Number of files: {count}")