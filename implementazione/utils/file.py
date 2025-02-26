"""
Questo modulo contiene funzioni per la gestione dei file e delle directory.

Questo modulo contiene funzioni per ottenere il percorso assoluto di un file, e per esportare un dataset in formato CSV.

Autore: Christian Corvino
Data: 26/02/2025
"""

import os

def get_absolute_path(file_name):
    """
    Restituisce il percorso assoluto di un file dato il suo nome.
    
    Attraversa la directory corrente e le sue sottodirectory per cercare il file.
    
    Args:
        file_name (str): Il nome del file da cercare.
    
    Returns:
        str: Il percorso assoluto del file, oppure None se non è stato trovato
    """
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

def export_dataset(dataset, file_csv, directory):
    """
    Esporta un dataset in formato CSV nella directory specificata.
    
    Crea la directory se non esiste.
    
    Args:
        dataset (pd.DataFrame): Il dataset da esportare.
        file_csv (str): Il nome del file CSV.
        directory (str): Il nome della directory in cui esportare il file.
    
    Returns:
        None
    """
    # Specifica il percorso della directory 'dir_name' rispetto alla directory corrente
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    dir_dataset = os.path.join(project_root, directory)
    
    # Normalizza il percorso per evitare mescolanza di separatori
    dir_dataset = os.path.normpath(dir_dataset)

    # Se 'dir_name' non esiste la crea
    if not os.path.exists(dir_dataset):
        os.makedirs(dir_dataset)

    # Esporta il DataFrame 'dataSet' in formato CSV nel percorso specificato
    dataset.to_csv(os.path.join(dir_dataset, file_csv), index=False)
    print(f"Dataset export: '{os.path.join(dir_dataset, file_csv)}'")
