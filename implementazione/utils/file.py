import os
import pandas as pd

def get_absolute_path(file_name):
    # Ottieni il percorso assoluto della directory corrente
    current_directory = os.path.abspath(os.getcwd())
    
    # Attraversa la directory corrente e le sue sottodirectory
    for root, _, files in os.walk(current_directory):
        # Cerca il file con il nome specificato
        if file_name in files:
            # Costruisci il percorso assoluto del file
            file_path = os.path.join(root, file_name)
            return file_path
    
    # Se il file non è stato trovato
    print(f"Il file '{file_name}' non è stato trovato nella directory corrente o nelle sue sottodirectory.\n")
    return None

# Funzione per esportare un dataset "dataset" in formato CSV di nome "csv_file_name" nella directory specificata "dir_name"
def export_dataset(dataset, csv, directory):
    # Specifica il percorso della directory 'dir_name' rispetto alla directory corrente
    dir_dataset = os.path.join(os.path.dirname(__file__), directory)

    # Assicurati che la directory 'dir_name' esista, altrimenti creala
    if not os.path.exists(dir_dataset):
        os.makedirs(dir_dataset)

    # Esporta il DataFrame 'dataSet' in formato CSV nel percorso specificato
    dataset.to_csv(os.path.join(dir_dataset, csv), index=False)
