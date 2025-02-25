import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.file import export_dataset

def preprocess_continuous(df, X, y):
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
    # Modifica sia il DataFrame che la lista features in un unico ciclo
    for i, col in enumerate(df.columns):
        if "% trains late" in col:
            # Rimuove il contenuto tra parentesi tonde (incluse le parentesi)
            cleaned_col = re.sub(r'\(.*?\)', '', col).strip()
            df.rename(columns={col: cleaned_col}, inplace=True)
            
            # Aggiorna la lista features direttamente
            if col in features:
                features[features.index(col)] = cleaned_col
    
    return df, features

def preprocess_discrete(df, columns, bins=4):
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
