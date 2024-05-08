import os
from pydub import AudioSegment
import Media

def adjust_audio_length(audio_path, target_length):
    # Carica il file audio
    audio = AudioSegment.from_file(audio_path)

    # Calcola la durata attuale dell'audio
    current_length = len(audio) / 1000  # converti da millisecondi a secondi

    if current_length > target_length:
        # Se l'audio è più lungo del target, taglia l'audio alla lunghezza desiderata
        audio = audio[:target_length * 1000]  # converte il tempo da secondi a millisecondi
    else:
        # Se l'audio è più corto del target, duplica una parte dell'audio per aggiungere come padding
        while len(audio) / 1000 < target_length:
            # Duplica una parte dell'audio
            audio += audio

        # Taglia l'audio al target length
        audio = audio[:target_length * 1000]

    return audio

# Ottieni il percorso assoluto del file corrente (dove viene eseguito il codice)
current_file = os.path.abspath(__file__)

# Ottieni il percorso della cartella genitore del file corrente
parent_folder = os.path.dirname(current_file)

parent_folder = os.path.dirname(parent_folder)

# Definisci il percorso della cartella "Dataset"
dataset_folder_path = os.path.join(parent_folder, "Dataset")

# Definisci il nome del file target
file_name = "Target"

# Unisci il percorso della cartella "Dataset" con il nome del file target
main_folder_path = os.path.join(dataset_folder_path, file_name)

# Nome della nuova cartella
new_folder_name = 'TargetModificato'

# Percorso completo della nuova cartella
new_folder_path = os.path.join(parent_folder, new_folder_name)

# Creare la nuova cartella
os.mkdir(new_folder_path)

# Definisci il tempo desiderato in secondi
Lista_input = Media.media(main_folder_path) #Prende in input media lunghezza audio e il totale di file

# Inizializza il contatore dei file
contatore_file = 0

# Itera sui file nella cartella principale
for file_name in os.listdir(main_folder_path):
    path_audio = os.path.join(main_folder_path, file_name)
    new_file_path = os.path.join(new_folder_path, file_name)
    os.mkdir(new_file_path)
    # Itera sui file audio nella cartella
    for wav_file in os.listdir(path_audio):
        input_audio_path = os.path.join(path_audio, wav_file)
        output_audio_path = os.path.join(new_file_path, wav_file)
        adjusted_audio = adjust_audio_length(input_audio_path, Lista_input[0])
        adjusted_audio.export(output_audio_path, format='wav')
        contatore_file += 1
        print("Audio modificato salvato in:", output_audio_path)
        print("File: " + str(contatore_file) + ' di ' + str(Lista_input[1]))

