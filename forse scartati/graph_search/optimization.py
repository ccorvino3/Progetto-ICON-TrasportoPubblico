# optimization.py
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

# Inizializza colorama
init(autoreset=True)

# Funzione che prende un dataset e ritorna un insieme contenente i valori unici della colonna 'stations_name_column'
def get_unique_stations(dataset, stations_name_column):
    unique_stations = set(dataset[stations_name_column].drop_duplicates())
    return unique_stations

# Funzione che stampa le stazioni uniche in un insieme, con un numero specificato (line_length) di stazioni per riga
# e colori diversi per ciascuna stazione (per una migliore visualizzazione dei dati)
def print_unique_stations(stations_set, line_length=5):
    colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA]
    stations_list = list(stations_set)
    for i in range(0, len(stations_list), line_length):
        line_elements = []
        for j in range(line_length):
            if i + j < len(stations_list):
                color = colors[j % len(colors)]
                line_elements.append(color + stations_list[i + j] + Style.RESET_ALL)
        print(", ".join(line_elements))

# Funzione che crea un grafo vuoto per il network di trasporto pubblico
def create_graph():
    G = nx.Graph()
    return G

# Funzione che inizializza il grafo con i nodi e gli archi del network di trasporto pubblico a partire dal dataset
def init_graph_by_dataset(G, dataset):
    for i, row in dataset.iterrows(): # i: non usata ma serve per accedere ai valori delle righe del dataset, else errore: 'i not defined'
        G.add_edge(row['Departure station'], row['Arrival station'], weight=row['Average travel time (min)'])

# Funzione che visualizza il grafo del network di trasporto pubblico con le stazioni come nodi e le rotte come 
# archi pesati con il tempo di viaggio medio in minuti tra le stazioni collegate da ciascun arco
def visualize_graph(G):
    ''' 
    # Metodo 1: 
    pos = nx.spring_layout(G, seed=42)  # posizione dei nodi
    edge_labels = nx.get_edge_attributes(G, 'weight')  # etichette degli archi

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    '''
    
    # Metodo 2: Disposizione con Fruchterman-Reingold e aumento della distanza
    pos = nx.spring_layout(G, k=3.0)
    
    # Metodo 3: Disposizione a conchiglia
    #pos = nx.shell_layout(G)

    # Visualizza il grafo con i nodi posizionati secondo il layout a molla
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='grey', node_size=500, font_size=7, font_weight='bold')
    plt.title('Transportation Network Graph')
    plt.show()

# Funzione che restituisce il nome di una stazione di partenza o arrivo valida nel grafo del network di trasporto pubblico G
def get_station_name(G, type_station):
    while True: # Ciclo finché non si inserisce un nome di stazione valido
        # Chiedi all'utente di inserire il nome della stazione e convertilo in maiuscolo se non lo è già
        station_name = input(f"Inserisci il nome della stazione di {type_station}: ")
        station_name = convert_station_name_uppercase(station_name)
        
        if station_name == "": # Se il nome della stazione è vuoto
            print("Il nome della stazione non può essere vuoto. Riprova.")
        elif station_name not in G.nodes: # Se la stazione non esiste nel grafo
            print("La stazione inserita non esiste. Riprova.")
        else: # Se il nome della stazione è valido
            return station_name

# Funzione che converte il nome di una stazione in maiuscolo se non lo è già e restituisce la nuova stazione convertita
def convert_station_name_uppercase(station_name):
    if not station_name.isupper():
        station_name = station_name.upper()
    return station_name

# Ottiene l'insieme delle stazioni di 'type_station' uniche e lo stampa, e chiede all'utente di inserire il nome della stazione di 'type_station'
def print_and_get_station_name(dataset, G, column_station_name, type_station):
    unique_stations = get_unique_stations(dataset, column_station_name)
    print_unique_stations(unique_stations)
    station = get_station_name(G, type_station)
    print("\n")
    return station
        
def optimize_transport_plan(dataset):
    # Crea un grafo per il network di trasporto pubblico
    G = create_graph()
    print("1. Grafo creato per il network di trasporto pubblico.")
    
    # Aggiunge nodi e archi al grafo
    init_graph_by_dataset(G, dataset)
    print("2. Nodi e archi aggiunti al grafo.")

    # Visualizza il grafo
    visualize_graph(G)
    print("3. Grafo visualizzato.")
    
    # Funzione euristica per l'algoritmo A* che calcola la distanza euclidea tra due stazioni
    def heuristic(u, v):
        return nx.shortest_path_length(G, source=u, target=v, weight='weight')
    
    # Ottieni il nome della stazione di partenza e di arrivo, per quante volte l'utente vuole
    while input("Vuoi inserire le stazioni di partenza e arrivo? (s/n): ").lower() == 's':
        print("\n")
        source_station = print_and_get_station_name(dataset, G, 'Departure station', 'partenza')
        target_station = print_and_get_station_name(dataset, G, 'Arrival station', 'arrivo')
        if source_station != target_station:
            # Calcola il percorso ottimale tra la stazione di partenza e la stazione di arrivo utilizzando l'algoritmo A* con l'euristica specificata e lo stampa
            best_path = nx.astar_path(G, source_station, target_station, heuristic=heuristic)
            print(f"Percorso ottimale da {source_station} a {target_station}:", best_path)
        else:
            print("La stazione di partenza e di arrivo non possono essere uguali. Riprova.")
    print("4. Scelta per l'utente delle stazioni di partenza e arrivo.")
    
    print("Ottimizzazione della pianificazione del trasporto completata.")
    print("----------------------------------------\n")