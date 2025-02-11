from utils.dataset import load_dataset
from utils.preprocessing import preprocess_data
from supervised_learning.prediction import predict
from prolog.prolog import main as prolog_main
from utils.file import export_dataset
    
def main():
    # 1. Caricamento del dataset
    dataset = load_dataset("progettazione/dataset", "Regularities_by_liaisons_Trains_France.csv")
    
    # 2. Preprocessing del dataset
    dataset, _ = preprocess_data(dataset)
    export_dataset(dataset, "dataset_preprocessed.csv", "progettazione/dataset")
    
    # Selezionare le colonne per le caratteristiche (X)
    features = [ # Lista delle feature (18 colonne)
        'Average travel time (min)', 'Number of cancelled trains', 'Number of late trains at departure',
        'Average delay of late departing trains (min)', 'Average delay of all departing trains (min)',
        'Number of trains late on arrival', 'Average delay of late arriving trains (min)', 
        '% trains late due to external causes', '% trains late due to railway infrastructure', 
        '% trains late due to traffic management', '% trains late due to rolling stock',
        '% trains late due to station management and reuse of material', '% trains late due to passenger traffic',
        'Number of late trains > 15min', 'Average train delay > 15min', 'Number of late trains > 30min',
        'Number of late trains > 60min', 'Period'
    ]
    
    # 3. Predizione
    random_forest = predict(dataset, features, 'Average delay of all arriving trains (min)')
    features.remove('Period') # Lo rimuoviamo perché non mi è più utile da dopo la predizione
    
    # 4. Prolog
    # prolog_main(dataset, features, random_forest)
    
    # 5. Apprendimento bayesiano
    # TODO: Implementare l'apprendimento bayesiano

if __name__ == "__main__":
    print("----------------------------------------------------------------------------------------------")
    print("\nProgetto ICON - Ingegneria della Conoscenza\n")
    main()
    print("----------------------------------------------------------------------------------------------")
