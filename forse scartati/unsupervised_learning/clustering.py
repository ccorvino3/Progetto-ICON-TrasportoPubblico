# clustering.py
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from utils.manage_file import export_dataset
from utils.dataset import load_dataset

# Funzione che calcola il numero di cluster ottimale per il dataset mediante il metodo del gomito
def rule_gomito(dataSet):
    inerzia = []
    
    # Fisso un range di k da 1 a 10
    maxK = 10
    for i in range(1, maxK):
        # Eseguo il kmeans per ogni k, con 5 inizializzazioni diverse e con inizializzazione random. Prendo la migliore
        kmeans = KMeans(n_clusters=i,n_init=5,init='random')
        kmeans.fit(dataSet)
        inerzia.append(kmeans.inertia_)
    
    # Con la libreria kneed trovo il k ottimale
    kl = KneeLocator(range(1, maxK), inerzia, curve="convex", direction="decreasing")
    
    # Visualizza il grafico con la nota per il miglior k
    plt.plot(range(1, maxK), inerzia, 'bx-')
    plt.scatter(kl.elbow, inerzia[kl.elbow - 1], c='red', label=f'Miglior k: {kl.elbow}')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inerzia')
    plt.title('Metodo del gomito per trovare il k ottimale')
    plt.legend()
    plt.show()
    
    return kl.elbow

# Funzione che esegue il KMeans sul dataset e restituisce le etichette e i centroidi
def compute_cluster(dataSet):
    # Calcola il numero ottimale di cluster utilizzando il metodo del gomito
    k = rule_gomito(dataSet)
    
    # Inizializza l'algoritmo KMeans con il numero ottimale di cluster, 10 inizializzazioni e inizializzazione casuale
    kmeans = KMeans(n_clusters=k, n_init=10, init='random')
    
    # Esegue il fitting del modello KMeans sul dataset
    kmeans = kmeans.fit(dataSet)
    
    # Ottiene le etichette dei cluster assegnati a ciascun punto del dataset
    etichette = kmeans.labels_
    
    # Ottiene le coordinate dei centroidi dei cluster
    centroidi = kmeans.cluster_centers_
    
    return etichette, centroidi

def show_clusters(dataSet, labels):
    # Visualizza i cluster
    plt.scatter(dataSet['Average travel time (min)'], dataSet['Average delay of all departing trains (min)'], c=labels, cmap='viridis')
    plt.xlabel('Average travel time (min)')
    plt.ylabel('Average delay of all departing trains (min)')
    plt.show()

# Funzione che esegue il clustering KMeans sul dataset e salva il dataset con le etichette di clustering 
# in un file CSV nella directory 'dataset'
def perform_clustering(dataset, clustering_columns):
    # Applica il clustering KMeans alle feature selezionate
    features = dataset[clustering_columns]
    labels, centroids = compute_cluster(features)
    print("1. Clustering KMeans.")

    # Crea una copia del dataset e aggiunge le etichette di clustering al dataset originale
    dataset_clustered = dataset.copy()
    dataset_clustered['ClusterIndex'] = labels
    print("2. Aggiunta delle etichette di clustering al dataset.")

    # Salva i cluster in un file csv nella directory 'dataset'
    export_dataset(dataset_clustered, 'clustered_dataset.csv', 'unsupervised_learning/dataset')
    print("3. Salvataggio del dataset con le etichette di clustering.")
    
    # Visualizza i cluster
    show_clusters(dataset_clustered, labels)
    print("4. Visualizzazione dei cluster.")
    
    path_file_clustered_dataset = load_dataset("unsupervised_learning", "clustered_dataset.csv")
    print(f"Il dataset con le etichette di clustering è stato salvato in {path_file_clustered_dataset}.")
    print("5. Salvataggio del file di clustering.")
    
    print("Clustering completato.")
    print("-----------------------------------------\n")