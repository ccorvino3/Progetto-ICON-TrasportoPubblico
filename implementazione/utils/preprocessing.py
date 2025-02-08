import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(dataset):
    # Rinomina le colonne per renderle più leggibili
    dataset.rename(columns={
        '% trains late due to external causes (weather, obstacles, suspicious packages, malevolence, social movements, etc.)': '% trains late due to external causes',
        '% trains late due to railway infrastructure (maintenance, works)': '% trains late due to railway infrastructure',
        '% trains late due to traffic management (rail line traffic, network interactions)': '% trains late due to traffic management',
        '% trains late due to passenger traffic (affluence, PSH management, connections)': '% trains late due to passenger traffic'
    }, inplace=True)

    # Droppa le colonne non rilevanti per l'analisi
    dropped_columns = dataset[['Comment (optional) delays at departure', 'Comment (optional) delays on arrival']]
    dataset = dataset.drop(columns=dropped_columns.columns)

    # Trasforma le variabili categoriche in numeriche (encoding target-based per ridurre la dimensionalità)
    label_encoder = LabelEncoder()
    for col in ['Departure station', 'Arrival station']:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    # Imputazione dei valori mancanti
    numeric_columns = dataset.select_dtypes(include=['float64']).columns
    imputer = SimpleImputer(strategy='mean')  # Usa la media per imputare i valori mancanti
    dataset[numeric_columns] = imputer.fit_transform(dataset[numeric_columns])

    # Normalizza le variabili numeriche escludendo le colonne categoriche
    columns_to_normalize = ['Average travel time (min)', 'Number of cancelled trains', 'Number of late trains at departure',
                            'Average delay of late departing trains (min)', 'Average delay of all departing trains (min)',
                            'Number of trains late on arrival', 'Average delay of late arriving trains (min)', 
                            'Average delay of all arriving trains (min)', '% trains late due to external causes',
                            '% trains late due to railway infrastructure', '% trains late due to traffic management', 
                            '% trains late due to rolling stock', '% trains late due to station management and reuse of material', 
                            '% trains late due to passenger traffic', 'Number of late trains > 15min', 'Average train delay > 15min',
                            'Number of late trains > 30min', 'Number of late trains > 60min']
    scaler = MinMaxScaler()
    dataset[columns_to_normalize] = scaler.fit_transform(dataset[columns_to_normalize])

    # Restituisce anche le stazioni per uso futuro
    stations_name = dataset[['Departure station', 'Arrival station']]

    print("Preprocessing completato.")
    print("---------------------------------------\n")

    return dataset, stations_name
