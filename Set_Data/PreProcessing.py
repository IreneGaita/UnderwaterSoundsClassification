import os
from pydub import AudioSegment

import Media

contatore_file = 0
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

# Percorso della cartella principale
main_folder_path = r'C:\Users\biagi\PycharmProjects\pythonProject\Target'

# Percorso della cartella genitore
parent_folder_path = r'C:\Users\biagi\PycharmProjects\pythonProject'

# Nome della nuova cartella
new_folder_name = 'TargetModificato'

# Percorso completo della nuova cartella
new_folder_path = os.path.join(parent_folder_path, new_folder_name)

# Creare la nuova cartella
os.mkdir(new_folder_path)

# Definisci il tempo desiderato in secondi
Lista_input = Media.media(main_folder_path) #Prende in input media lunghezza audio e il totale di file

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
        contatore_file = contatore_file + 1
        print("Audio modificato salvato in:", output_audio_path)
        print("File: " + str(contatore_file) + ' di ' + str(Lista_input[1]))