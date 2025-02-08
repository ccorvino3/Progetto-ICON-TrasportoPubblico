# main.py
from utils.dataset import load_dataset
from utils.data_preprocessing import preprocess_data
    
def main():
    # 1. Caricamento del dataset
    dataset = load_dataset("dataset", "Regularities_by_liaisons_Trains_France.csv")
    
    # 2. Preprocessing del dataset
    # Preprocessing del dataset
    dataset, stations_name = preprocess_data(dataset)

    # 3. Clustering delle stazioni e delle rotte
    # Colonne per il clustering delle stazioni e delle rotte
    # clustering_columns = ['Average delay of all departing trains (min)', 
    #                         '% trains late due to external causes', 
    #                         'Number of cancelled trains', 
    #                         'Average delay of late arriving trains (min)']
    # # Segmentazione delle stazioni e delle rotte in cluster omogenei in base alle caratteristiche di ritardo e cancellazione dei treni
    # perform_clustering(dataset, clustering_columns)
    
    # # 4. Previsione della domanda di trasporto pubblico
    # # Carica il dataset clusterizzato
    # clustered_dataset_path = get_absolute_path("clustered_dataset.csv")
    # clustered_dataset = get_dataset(clustered_dataset_path)
    # # Previsione della domanda di trasporto pubblico sul dataset clusterizzato
    # predict_transport_demand(clustered_dataset, 'Number of expected circulations')
    
    # # TODO: fare la 5. e capire se interessa clustering e predizione domanda trasporto pubblico
    # # 5. Utilizzo di reti neurali per la previsione
    # # Addestramento di una rete neurale per la previsione della media dei ritardi di partenza dei treni
    # train_neural_network(dataset)
    
    '''
    # Creazione di ontologie per la rappresentazione della conoscenza
    #onto_cause_delay_trains(dataset) # TODO: da finire o da rivedere
    
    # Ragionamento con vincoli
    solve_csp(dataset)
    
    # Rimozione delle colonne 'Departure station' e 'Arrival station' dal dataset in quanto non sono più necessarie (TODO: da capire)
    dataset = dataset.drop(columns=['Departure station', 'Arrival station'])
    
    # . Ottimizzazione della pianificazione del trasporto pubblico (A* search)
    # Riaggiunge le colonne Depature station e Arrival station al dataset
    dataset['Departure station'] = stations_name[0]
    dataset['Arrival station'] = stations_name[1]
    # Ottimizzazione della pianificazione del trasporto
    optimize_transport_plan(dataset)
    '''

if __name__ == "__main__":
    print("----------------------------------------------------------------------------------------------")
    print("\nProgetto ICON - Ingegneria della Conoscenza\n")
    main()