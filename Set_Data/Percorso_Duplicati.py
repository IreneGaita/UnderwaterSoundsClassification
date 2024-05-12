import os

def find_duplicates_in_non_target(non_target_folder, duplicates_folder):
    import os

    def find_duplicates_in_non_target(non_target_folder, duplicates_folder):
        duplicates_dict = {}
        duplicate_count = 0

        # Percorri tutti i file nella cartella "Duplicates"
        for root, _, files in os.walk(duplicates_folder):
            for file_name in files:
                # Costruisci il percorso completo del file duplicato
                duplicate_file_path = os.path.join(root, file_name)

                # Costruisci il nome del file senza estensione
                file_name_without_extension, _ = os.path.splitext(file_name)

                # Cerca il file corrispondente nella cartella "Non-Target"
                for non_target_root, _, non_target_files in os.walk(non_target_folder):
                    for non_target_file in non_target_files:
                        # Se il nome del file nella cartella "Non-Target" corrisponde al nome del file duplicato
                        if non_target_file.startswith(file_name_without_extension):
                            # Costruisci il percorso completo del file corrispondente in "Non-Target"
                            non_target_file_path = os.path.join(non_target_root, non_target_file)

                            # Aggiungi il percorso del file corrispondente al dizionario dei duplicati
                            duplicates_dict[duplicate_file_path] = non_target_file_path
                            duplicate_count += 1

        return duplicates_dict, duplicate_count

    if __name__ == "__main__":
        # Definisci il percorso delle cartelle "Non-Target" e "Duplicates"
        non_target_folder_path = os.path.join("../Dataset", "Non-Target")
        duplicates_folder_path = os.path.join("../Dataset", "Duplicates")

        # Trova i duplicati nella cartella "Non-Target" utilizzando i nomi dei file presenti in "Duplicates"
        duplicates_dict, duplicate_count = find_duplicates_in_non_target(non_target_folder_path, duplicates_folder_path)

        # Stampare i risultati
        if duplicates_dict:
            print(f"Duplicati trovati: {duplicate_count}")
            for duplicate_file_path, non_target_file_path in duplicates_dict.items():
                print(f"File duplicato in Duplicates: {duplicate_file_path}")
                print(f"Corrispondente in Non-Target: {non_target_file_path}")
        else:
            print("Nessun duplicato trovato.")


if __name__ == "__main__":
    # Definisci il percorso delle cartelle "Non-Target" e "Duplicates"
    non_target_folder_path = os.path.join("../Dataset", "Non-Target")
    duplicates_folder_path = os.path.join("../Dataset", "Duplicates")

    # Trova i duplicati nella cartella "Non-Target" utilizzando i nomi dei file presenti in "Duplicates"
    duplicates_dict = find_duplicates_in_non_target(non_target_folder_path, duplicates_folder_path)

    # Stampare i risultati
    if duplicates_dict:
        print("Duplicati trovati:")
        for duplicate_file_path, non_target_file_path in duplicates_dict.items():
            print(f"File duplicato in Duplicates: {duplicate_file_path}")
            print(f"Corrispondente in Non-Target: {non_target_file_path}")
    else:
        print("Nessun duplicato trovato.")
