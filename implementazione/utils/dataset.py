"""
Funzioni di utilità per la gestione dei dataset.

Questo modulo contiene funzioni per il caricamento, l'analisi e la manipolazione dei dataset.

Autore: Christian Corvino
Data: 26/02/2025
"""

import os
import re
import pandas as pd
import numpy as np

def load_dataset(folder_path, file_name):
    """
    Carica un dataset CSV dato il nome del file e la cartella di destinazione.
    
    Args:
        folder_path (str): Percorso della cartella contenente il dataset.
        file_name (str): Nome del file CSV.
    
    Returns:
        pd.DataFrame: DataFrame pandas contenente il dataset, oppure None in caso di errore.
    """
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

def print_columns_and_types_to_md(df, filename='progettazione/dataset/operation/TypesColumnsDataset.md'):
    """
    Scrive in un file Markdown i nomi delle colonne e i relativi tipi di dati di un dataset.
    
    Args:
        df (pd.DataFrame): DataFrame pandas contenente il dataset.
        filename (str): Il nome del file Markdown in cui scrivere i dati.
    
    Returns:
        None
    """
    try:
        # Crea o apre il file Markdown in modalità scrittura
        with open(filename, 'w') as f:
            # Scrive l'intestazione della tabella
            f.write("# Colonne e Tipi di Dati\n\n")
            f.write("| Nome Colonna | Tipo di Dato |\n")
            f.write("|--------------|--------------|\n")
            
            # Scrive le righe della tabella
            for column in df.columns:
                f.write(f"| {column} | {df[column].dtype} |\n")
        
        print(f"Risultati salvati in {filename}")
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")

def print_stations_columns_to_md(df, filename='progettazione/dataset/operation/StationsColumns.md'):
    """
    Scrive in un file Markdown i nomi delle colonne di un dataset.
    
    Args:
        df (pd.DataFrame): DataFrame pandas contenente il dataset.
        filename (str): Il nome del file Markdown in cui scrivere i dati.
    
    Returns:
        None
    """
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

def print_column_not_null(df):
    """
    Calcola e stampa la percentuale di valori non nulli per ogni colonna di un dataset.
    
    Args:
        df (pd.DataFrame): DataFrame pandas contenente il dataset.
    
    Returns:
        None
    """
    try:
        print("Percentuale di valori non nulli per colonna:\n")
        for idx, column in enumerate(df.columns, start=1):
            total_rows = len(df)  # Numero totale di righe
            non_null_count = df[column].notnull().sum()  # Valori non nulli
            percentage = (non_null_count / total_rows) * 100  # Percentuale
            print(f"{idx}: {percentage:.2f}% -> {non_null_count}/{total_rows} righe")
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")

def save_columns_not_null_and_not_zero_to_md(df, output_md='progettazione/dataset/operation/StatusDataset.md'):
    """
    Calcola la percentuale di valori non nulli e non zero per ogni colonna di un dataset e salva i risultati in un file Markdown.
    
    Args:
        df (pd.DataFrame): DataFrame pandas contenente il dataset.
        output_md (str): Percorso del file Markdown di output (default 'output_percentage.md').
    
    Returns:
        None
    """
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

def count_year_month_matches_period(df):
    """
    Calcola e stampa il numero di tuple del df che hanno 'Year-Month' uguale a 'Period'.
    
    Args:
        df (pd.DataFrame): DataFrame pandas contenente il dataset.
    
    Returns:
        None
    """
    # Converti 'Month' in intero con due cifre
    df['Month_str'] = df['Month'].astype(int).astype(str).str.zfill(2)

    # Crea la colonna 'Year-Month' nel formato 'YYYY-MM'
    df['Year-Month'] = df['Year'].astype(str) + '-' + df['Month_str']

    # Conta le righe dove 'Year-Month' è uguale a 'Period'
    corrispondenze = (df['Year-Month'] == df['Period']).sum()

    print(f"Righe con corrispondenza Year-Month e Period: {corrispondenze} / {len(df)}")

    # Rimuove le colonne temporanee
    df.drop(['Month_str', 'Year-Month'], axis=1, inplace=True)

def remove_content_Parentheses(s):
    """
    Rimuove le parentesi e il loro contenuto da una stringa.
    
    Args:
        s (str): Stringa da cui rimuovere le parentesi.
    
    Returns:
        str: Stringa senza parentesi.
    """
    return re.sub(r'\([^)]*\)', '', s)

def convert_float_to_time_format(value):
    r"""
    Converti un valore float in minuti e secondi con la gestione di anticipi e ritardi.

    Args:
        value (float): Il valore in minuti (positivo per ritardo, negativo per anticipo).

    Returns:
        str (str): Il valore convertito in formato "X' Y\"" con l'aggiunta di "in anticipo" se negativo.
    """
    if isinstance(value, np.ndarray):
        value = value.item()

    if value == 0:
        return '0s'

    sign = "in anticipo" if value < 0 else ""

    # Valore assoluto per il calcolo di minuti e secondi
    abs_value = abs(value)

    # Calcola minuti e secondi
    minutes = int(abs_value)
    seconds = round((abs_value - minutes) * 60)

    # Formatta il risultato
    if minutes == 0:
        return f'{seconds}s {sign}'
    else:
        return f"{minutes}m {seconds}s {sign}"

def inverse_transform_column(scaled_value, scaler, column_name):
    """
    Ripristina i valori originali di una singola colonna scalata.
    
    Poiché lo scaler è stato adattato su più colonne, per invertire la normalizzazione
    di una colonna specifica è necessario usare la media e lo scale corrispondenti.
    
    Args:
        scaled_value (float): Valore scalato da ripristinare.
        scaler (StandardScaler): Lo scaler adattato ai dati.
        column_name (str): Nome della colonna da ripristinare.
    
    Returns:
        float: Valore originale della colonna.
    """
    # Se la colonna non è presente nei dati dello scaler, non fare nulla
    if column_name not in scaler.feature_names_in_:
        return scaled_value

    # Trovo l'indice della colonna all'interno dello scaler
    col_index = list(scaler.feature_names_in_).index(column_name)
    
    # Calcolo il valore originale usando la formula inversa: valore_originale = valore_scalato * scale + mean
    original_values = scaled_value * scaler.scale_[col_index] + scaler.mean_[col_index]
    
    return original_values
