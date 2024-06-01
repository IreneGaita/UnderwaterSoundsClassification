import os
import sys

def count_files_in_subfolders(main_folder, file_extension):
    # Dizionario per memorizzare il conteggio dei file per ogni sottocartella
    file_counts = {}
    total_count = 0

    # Percorrere tutte le directory e i file all'interno della cartella principale
    for root, dirs, files in os.walk(main_folder):
        # Inizializzare il conteggio per la directory corrente
        file_count = 0

        # Contare i file con l'estensione specificata nella directory corrente
        for file in files:
            if file.endswith(file_extension):
                file_count += 1

        # Se ci sono file con l'estensione specificata nella directory corrente, aggiungere al dizionario
        if file_count > 0:
            relative_path = os.path.relpath(root, main_folder)
            file_counts[relative_path] = file_count
            total_count += file_count

    return file_counts, total_count

if __name__ == "__main__":
    # Ottenere il percorso della cartella del dataset
    current_file = os.path.abspath(__file__)
    dataset_folder_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), "Allenamento")

    # Estensione dei file da contare
    file_extension = '.png'

    # Contare i file nella cartella principale e nelle sottocartelle
    wav_file_counts, grand_total_count = count_files_in_subfolders(dataset_folder_path, file_extension)

    # Stampare i risultati
    for folder, total_count in wav_file_counts.items():
        print(f"Cartella: {folder}, Numero di file totali {file_extension}: {total_count}")

    print(f"\nConteggio totale dei file {file_extension}: {grand_total_count}")

    sys.stdout.write("\nConteggio completato!\n")
