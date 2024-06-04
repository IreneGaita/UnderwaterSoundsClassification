import os
import shutil

def bilancia_file(cartella1, cartella2):
    def bilancia(cartella1, cartella2):
        # Ottiene tutte le sottocartelle nelle due cartelle
        sottocartelle1 = set(os.listdir(cartella1))
        sottocartelle2 = set(os.listdir(cartella2))

        # Unione delle sottocartelle per assicurarsi che tutte vengano considerate
        tutte_le_sottocartelle = sottocartelle1 | sottocartelle2

        for sottocartella in tutte_le_sottocartelle:
            # Percorsi completi per le sottocartelle
            percorso1 = os.path.join(cartella1, sottocartella)
            percorso2 = os.path.join(cartella2, sottocartella)

            # Controlla se esiste già un file con lo stesso nome della sottocartella
            if os.path.isfile(percorso1) or os.path.isfile(percorso2):
                continue

            # Crea la sottocartella in percorso2 se non esiste
            if not os.path.exists(percorso1):
                os.makedirs(percorso1)
            if not os.path.exists(percorso2):
                os.makedirs(percorso2)

            # Procede se entrambe le sottocartelle sono directory
            if os.path.isdir(percorso1) and os.path.isdir(percorso2):
                # Chiama la funzione ricorsiva per gestire eventuali sottocartelle
                bilancia(percorso1, percorso2)

                # Ottieni tutti i file nelle sottocartelle
                file1 = [f for f in os.listdir(percorso1) if os.path.isfile(os.path.join(percorso1, f))]
                file2 = [f for f in os.listdir(percorso2) if os.path.isfile(os.path.join(percorso2, f))]

                # Calcola il numero totale di file e il target per bilanciare
                totale_file = len(file1) + len(file2)
                target_per_sottocartella = totale_file // 2

                # Se la differenza è significativa, sposta i file
                while len(file1) > target_per_sottocartella:
                    file_da_spostare = file1.pop()
                    destinazione = os.path.join(percorso2, file_da_spostare)
                    if not os.path.exists(destinazione):
                        shutil.move(os.path.join(percorso1, file_da_spostare), destinazione)
                        file2.append(file_da_spostare)

                while len(file2) > target_per_sottocartella:
                    file_da_spostare = file2.pop()
                    destinazione = os.path.join(percorso1, file_da_spostare)
                    if not os.path.exists(destinazione):
                        shutil.move(os.path.join(percorso2, file_da_spostare), destinazione)
                        file1.append(file_da_spostare)

    # Chiama la funzione ricorsiva iniziale
    bilancia(cartella1, cartella2)

# Esempio di utilizzo
cartella1 = r"C:\Users\biagi\PycharmProjects\gruppo17\Validazione_Norm"
cartella2 = r"C:\Users\biagi\PycharmProjects\gruppo17\Testing_Norm"

bilancia_file(cartella1, cartella2)
