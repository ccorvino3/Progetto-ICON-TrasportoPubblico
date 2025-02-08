# main.py
from utils.dataset import load_dataset
from utils.preprocessing import preprocess_data
from supervised_learning.prediction import predict
    
def main():
    # 1. Caricamento del dataset
    dataset = load_dataset("progettazione/dataset", "Regularities_by_liaisons_Trains_France.csv")
    
    # 2. Preprocessing del dataset
    dataset, stations_name = preprocess_data(dataset)
    
    # 3. Predizione
    predict(dataset, 'Average delay of all arriving trains (min)')
    
    print("----------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    print("----------------------------------------------------------------------------------------------")
    print("\nProgetto ICON - Ingegneria della Conoscenza\n")
    main()
