import os
import pandas as pd

"""
    Carica un dataset CSV dato il nome del file e la cartella di destinazione.

    :param folder_path: Percorso della cartella contenente il dataset.
    :param file_name: Nome del file CSV.
    :return: DataFrame pandas contenente il dataset, oppure un messaggio di errore.
"""
def load_dataset(folder_path, file_name):
    
    # Costruisci il percorso completo del file
    file_path = os.path.join(folder_path, file_name)
    
    # Verifica se il file esiste
    if not os.path.isfile(file_path):
        print(f"Errore: il file '{file_name}' non esiste nella cartella '{folder_path}'.")
        return None
    
    # Carica il dataset
    try:
        df = pd.read_csv(file_path)
        return df
    except pd.errors.ParserError:
        print(f"Errore: il file '{file_name}' non è un file CSV valido.")
        return None
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        return None

"""
    Scrive in un file Markdown i nomi delle colonne e i relativi tipi di dati di un dataset.
    
    :param df: DataFrame pandas contenente il dataset.
    :param filename: Il nome del file Markdown in cui scrivere i dati.
"""
def print_columns_and_types_to_md(df, output_md='progettazione/dataset/operation/TypesColumnsDataset.md'):
    try:
        # Crea o apre il file Markdown in modalità scrittura
        with open(output_md, 'w') as f:
            # Scrive l'intestazione della tabella
            f.write("# Colonne e Tipi di Dati\n\n")
            f.write("| Nome Colonna | Tipo di Dato |\n")
            f.write("|--------------|--------------|\n")
            
            # Scrive le righe della tabella
            for column in df.columns:
                f.write(f"| {column} | {df[column].dtype} |\n")
        
        print(f"Risultati salvati in {output_md}")
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")

"""
    Scrive in un file Markdown i nomi delle colonne di un dataset.
    
    :param df: DataFrame pandas contenente il dataset.
    :param filename: Il nome del file Markdown in cui scrivere i dati.
"""
def print_stations_columns_to_md(df, filename='progettazione/dataset/operation/StationsColumns.md'):
    try:
        # Crea o apre il file Markdown in modalità scrittura
        with open(filename, 'w') as f:
            # Scrive l'intestazione
            f.write("# Colonne del dataset stations_name\n\n")
            f.write("| Nome Colonna |\n")
            f.write("|--------------|\n")
            
            # Scrive i nomi delle colonne
            for column in df.columns:
                f.write(f"| {column} |\n")
        
        print(f"Tabella scritta con successo in {filename}")
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")

"""
    Calcola e stampa la percentuale di valori non nulli per ogni colonna di un dataset.

    :param df: DataFrame pandas contenente il dataset.
"""
def print_column_not_null(df):
    try:
        print("Percentuale di valori non nulli per colonna:\n")
        for idx, column in enumerate(df.columns, start=1):
            total_rows = len(df)  # Numero totale di righe
            non_null_count = df[column].notnull().sum()  # Valori non nulli
            percentage = (non_null_count / total_rows) * 100  # Percentuale
            print(f"{idx}: {percentage:.2f}% -> {non_null_count}/{total_rows} righe")
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")

"""
    Calcola la percentuale di valori non nulli e non zero per ogni colonna di un dataset
    e salva i risultati in un file Markdown.

    :param file_path: Percorso del file CSV.
    :param output_md: Percorso del file Markdown di output (default 'output_percentage.md').
"""
def save_columns_not_null_and_not_zero_to_md(df, output_md='progettazione/dataset/operation/StatusDataset.md'):
    # Creiamo una lista per salvare i risultati
    try:
        results = []

        # Itera sulle colonne e calcola la percentuale
        for idx, column in enumerate(df.columns, start=1):
            total_rows = len(df)  # Numero totale di righe
            non_null_count = df[column].notnull().sum()  # Valori non nulli
            non_zero_count = (df[column] != 0).sum()  # Valori non zero
            percentage_non_null = (non_null_count / total_rows) * 100  # Percentuale di non nulli
            percentage_non_zero = (non_zero_count / total_rows) * 100  # Percentuale di non zero
            
            results.append([idx, column, non_null_count, non_zero_count, total_rows, f"{percentage_non_null:.2f}%", f"{percentage_non_zero:.2f}%"])

        # Scrivi i risultati in un file Markdown
        with open(output_md, 'w') as md_file:
            md_file.write("# Percentuale di Valori Non Nulli e Non Zero per Colonna\n\n")
            md_file.write("| Indice | Colonna | Non Nulli | Non Zero | Totale Righe | Percentuale Non Nulli | Percentuale Non Zero |\n")
            md_file.write("|--------|---------|-----------|----------|--------------|-----------------------|----------------------|\n")
            
            for result in results:
                md_file.write(f"| {result[0]} | {result[1]} | {result[2]} | {result[3]} | {result[4]} | {result[5]} | {result[6]} |\n")
        
        print(f"Risultati salvati in {output_md}")
        
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
