from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

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

    return df

def preprocess_discrete(df):
    # Rinomina le colonne per renderle più leggibili
    df.rename(columns={
        '% trains late due to external causes (weather, obstacles, suspicious packages, malevolence, social movements, etc.)': '% trains late due to external causes',
        '% trains late due to railway infrastructure (maintenance, works)': '% trains late due to railway infrastructure',
        '% trains late due to traffic management (rail line traffic, network interactions)': '% trains late due to traffic management',
        '% trains late due to passenger traffic (affluence, PSH management, connections)': '% trains late due to passenger traffic',
        '% trains late due to rolling stock': '% trains late due to rolling stock',
        '% trains late due to station management and reuse of material': '% trains late due to station management'
    }, inplace=True)

    # Rendo la colonna 'Month' un intero anziché float
    df['Month'] = df['Month'].astype(int)

    # Crea una colonna binaria: 1 se ci sono treni in ritardo > 15 min, altrimenti 0
    df['Late > 15 min'] = (df['Number of late trains > 15min'] > 0).astype(int)

    # Elimina le colonne non rilevanti
    df.drop(columns=[
        'Comment (optional) delays at departure', 'Comment (optional) delays on arrival',
        'Departure station', 'Arrival station', 'Period', 'Number of late trains > 15min'
    ], inplace=True)

    # Gestisce i valori mancanti
    float_columns = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].fillna(df[float_columns].mean())

    # Discretizza le variabili continue
    discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
    df[float_columns] = discretizer.fit_transform(df[float_columns])

    print("Preprocessing completato.")
    export_dataset(df, "df_preprocessed_discrete.csv", "progettazione/dataset")
    print("---------------------------------------\n")

    return df

# if __name__ == "__main__":
#     main()
