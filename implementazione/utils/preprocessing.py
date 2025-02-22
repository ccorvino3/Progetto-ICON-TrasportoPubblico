import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.file import export_dataset

def preprocess_continuous(df):
    # Rinomina le colonne per renderle più leggibili
    df.rename(columns={
        '% trains late due to external causes (weather, obstacles, suspicious packages, malevolence, social movements, etc.)': '% trains late due to external causes',
        '% trains late due to railway infrastructure (maintenance, works)': '% trains late due to railway infrastructure',
        '% trains late due to traffic management (rail line traffic, network interactions)': '% trains late due to traffic management',
        '% trains late due to passenger traffic (affluence, PSH management, connections)': '% trains late due to passenger traffic'
    }, inplace=True)

    # Rendo la colonna 'Month' un intero anziché float
    df['Month'] = df['Month'].astype(int)

    # Droppa le colonne non rilevanti per l'analisi
    # stations_name = df[['Departure station', 'Arrival station']]
    df = df.drop(
        columns=[
            'Comment (optional) delays at departure', 'Comment (optional) delays on arrival', # Commenti non rilevanti per l'analisi
            "Departure station", "Arrival station",  # Stazioni di partenza e arrivo non rilevanti per l'analisi
            "Period"] # Periodo superflua per l'analisi perché sta già la divisione in 'Anno' e 'Mese'
    )

    # Gestisce i valori mancanti
    float_columns = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].fillna(df[float_columns].mean())

    # Normalizza le variabili numeriche
    scaler = StandardScaler()
    df[float_columns] = scaler.fit_transform(df[float_columns])

    print("Preprocessing completato.")
    export_dataset(df, "df_preprocessed_continuous.csv", "progettazione/dataset")
    print("---------------------------------------\n")

    return df, scaler

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
