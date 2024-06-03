
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

def calculate_70_percent(file_counts):
    # Dizionario per memorizzare il 70% dei file per ogni sottocartella
    seventy_percent_counts = {}

    # Calcolare il 70% dei file per ogni sottocartella
    for folder, count in file_counts.items():
        seventy_percent = int(count * 0.7)
        seventy_percent_counts[folder] = seventy_percent

    return seventy_percent_counts

if __name__ == "__main__":
    import os
    import shutil


    def bilancia_file(cartella1, cartella2):
        # Ottiene tutte le sottocartelle comuni
        sottocartelle = set(os.listdir(cartella1)) & set(os.listdir(cartella2))

        for sottocartella in sottocartelle:
            # Percorsi completi per le sottocartelle
            percorso1 = os.path.join(cartella1, sottocartella)
            percorso2 = os.path.join(cartella2, sottocartella)

            if os.path.isdir(percorso1) and os.path.isdir(percorso2):
                # Ottieni tutti i file nelle sottocartelle
                file1 = os.listdir(percorso1)
                file2 = os.listdir(percorso2)

                # Calcola il numero totale di file e il target per bilanciare
                totale_file = len(file1) + len(file2)
                target_per_sottocartella = totale_file // 2

                # Se la differenza è significativa, sposta i file
                while len(file1) > target_per_sottocartella:
                    file_da_spostare = file1.pop()
                    shutil.move(os.path.join(percorso1, file_da_spostare), percorso2)
                    file2.append(file_da_spostare)

                while len(file2) > target_per_sottocartella:
                    file_da_spostare = file2.pop()
                    shutil.move(os.path.join(percorso2, file_da_spostare), percorso1)
                    file1.append(file_da_spostare)


    # Esempio di utilizzo
    cartella1 = r"C:\Users\biagi\PycharmProjects\gruppo17\Validazione_Norm"
    cartella2 = r"C:\Users\biagi\PycharmProjects\gruppo17\Testing_Norm"

    bilancia_file(cartella1, cartella2)
