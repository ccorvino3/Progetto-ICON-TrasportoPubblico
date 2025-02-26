"""
Modulo che contiene le funzioni per il preprocessing dei dati.

Questo modulo contiene funzioni per il preprocessing delle variabili continue e discrete di un dataset.

Autore: Christian Corvino
Data: 26/02/2025
"""

import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.file import export_dataset

def preprocess_continuous(df, X, y):
    """
    Preprocessa le variabili continue del DataFrame df, normalizzandole e gestendo i valori mancanti.
    
    Rinomina alcune colonne.
    
    Rende la colonna 'Month' un intero anziché float.
    
    Salva il DataFrame preprocessato in un file CSV.
    
    Args:
        df (pd.DataFrame): il DataFrame da preprocessare.
        X (list): la lista delle colonne da considerare.
        y (str): il nome della colonna target.
    
    Returns:
        tuple: Una tuple contenente:
            - il DataFrame preprocessato
            - lo StandardScaler usato per la normalizzazione
    """
    df_contin = df[X].copy()
    
    # Aggiunge la colonna target al DataFrame
    df_contin[y] = df[y]

    # Rinomina le colonne per renderle più leggibili
    clean_column_names(df_contin, X)

    # Rendo la colonna 'Month' un intero anziché float
    if "Month" in X:
        df_contin['Month'] = df_contin['Month'].astype(int)

    # Siccome copio in df_contin solo le colonne X, non mi serve droppare tutte quelle non rilevanti per l'analisi

    # Gestisce i valori mancanti
    float_columns = df_contin.select_dtypes(include=['float64']).columns
    df_contin[float_columns] = df_contin[float_columns].fillna(df_contin[float_columns].mean())

    # Normalizza le variabili numeriche
    scaler = StandardScaler()
    df_contin[float_columns] = scaler.fit_transform(df_contin[float_columns])

    print("Preprocessing completato.")
    export_dataset(df_contin, "df_preprocessed_continuous.csv", "progettazione/dataset")
    print("---------------------------------------\n")

    return df_contin, scaler

def clean_column_names(df, features):
    """
    Pulisce i nomi delle colonne del DataFrame df e aggiorna la lista features.
    
    Rimuove il contenuto tra parentesi tonde (incluse le parentesi) per le colonne che contengono "% trains late". 
    Aggiorna i nomi delle colonne nel DataFrame e nella lista delle features.
    
    Args:
        df (pd.DataFrame): Il DataFrame da pulire.
        features (list): La lista delle colonne da considerare.

    Returns:
        tuple: Una tupla contenente:
            - pd.DataFrame: Il DataFrame con i nomi delle colonne puliti.
            - list: La lista delle features aggiornata con i nuovi nomi delle colonne.
    """
    # Modifica sia il DataFrame che la lista features in un unico ciclo
    for _, col in enumerate(df.columns):
        if "% trains late" in col:
            # Rimuove il contenuto tra parentesi tonde (incluse le parentesi)
            cleaned_col = re.sub(r'\(.*?\)', '', col).strip()
            df.rename(columns={col: cleaned_col}, inplace=True)
            
            # Aggiorna la lista features direttamente
            if col in features:
                features[features.index(col)] = cleaned_col
    
    return df, features

def preprocess_discrete(df, columns, bins=4):
    """
    Preprocessa le variabili discrete del DataFrame df, discretizzandole e gestendo i valori mancanti.
    
    Rende la colonna 'Month' un intero anziché float.
    
    Salva il DataFrame preprocessato in un file CSV.
    
    Args:
        df (pd.DataFrame): il DataFrame da preprocessare.
        columns (list): la lista delle colonne da considerare.
        bins (int): il numero di bin in cui discretizzare le variabili numeriche.
    
    Returns:
        pd.DataFrame: il DataFrame preprocessato.
    """
    df_disc = df[columns].copy()

    # Rendo la colonna 'Month' un intero anziché float
    if 'Month' in columns:
        df_disc['Month'] = df['Month'].astype(int)

    # Gestisce i valori mancanti
    float_columns = df_disc.select_dtypes(include=['float64']).columns
    df_disc[float_columns] = df_disc[float_columns].fillna(df_disc[float_columns].mean())

    # Discretizza le variabili numeriche
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_disc[col]) and df_disc[col].nunique() > bins:
            df_disc[col] = pd.qcut(df_disc[col], q=bins, labels=False, duplicates='drop')

    print("Preprocessing completato.")
    export_dataset(df_disc, "df_preprocessed_discrete.csv", "progettazione/dataset")
    print("---------------------------------------\n")

    return df_disc

# if __name__ == "__main__":
#     main()
