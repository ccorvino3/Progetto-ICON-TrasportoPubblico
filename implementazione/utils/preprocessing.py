from sklearn.discriminant_analysis import StandardScaler

def preprocess_data(dataset):
    # Rinomina le colonne per renderle più leggibili
    dataset.rename(columns={
        '% trains late due to external causes (weather, obstacles, suspicious packages, malevolence, social movements, etc.)': '% trains late due to external causes',
        '% trains late due to railway infrastructure (maintenance, works)': '% trains late due to railway infrastructure',
        '% trains late due to traffic management (rail line traffic, network interactions)': '% trains late due to traffic management',
        '% trains late due to passenger traffic (affluence, PSH management, connections)': '% trains late due to passenger traffic'
    }, inplace=True)

    # Droppa le colonne non rilevanti per l'analisi
    stations_name = dataset[['Departure station', 'Arrival station']]
    dataset = dataset.drop(
        columns=[
            'Comment (optional) delays at departure', 'Comment (optional) delays on arrival',
            "Departure station", "Arrival station"]
    )

    # Trasforma le variabili categoriche in numeriche (encoding target-based per ridurre la dimensionalità)
    # label_encoder = LabelEncoder()
    # for col in ['Departure station', 'Arrival station']:
    #     dataset[col] = label_encoder.fit_transform(dataset[col])

    # Gestisce i valori mancanti
    float_columns = dataset.select_dtypes(include=['float64']).columns
    dataset[float_columns] = dataset[float_columns].fillna(dataset[float_columns].mean())

    # Normalizza le variabili numeriche
    scaler = StandardScaler()
    dataset[float_columns] = scaler.fit_transform(dataset[float_columns])

    print("Preprocessing completato.")
    print("---------------------------------------\n")

    return dataset, stations_name
