"""
Main del progetto ICON - Ingegneria della Conoscenza.

Crea la cartella per i risultati.
Definisce le colonne features e target per la predizione.

Questo script esegue le seguenti operazioni:
1. Caricamento del dataset
2. Preprocessing continuo del dataset
3. Predizione della colonna target (incluso addestramento e valutazione dei modelli)
4. Prolog: predizione della colonna target tramite regole Prolog
5. Apprendimento bayesiano: analisi della congestione ferroviaria tramite apprendimento bayesiano

Per eseguire il progetto, eseguire questo script.

Per eseguire il progetto:
- nel mio caso: py implementazione/main.py
- oppure: python implementazione/main.py

Autore: Christian Corvino
Data: 26/02/2025
"""

from utils.dataset import load_dataset
from utils.preprocessing import preprocess_continuous, preprocess_discrete
from supervised_learning.supervised_prediction import main as supervised_main
from KB_prolog.prolog_prediction import main as prolog_main
from bayesian_learning.bayesian_prediction import main as bayesian_main
import os

# Creazione della cartella per i risultati
if not os.path.exists("documentazione/res/drawable"):
    os.makedirs("documentazione/res/drawable")
if not os.path.exists("documentazione/res/drawable/img_supervised"):
    os.makedirs("documentazione/res/drawable/img_supervised")
if not os.path.exists("documentazione/res/drawable/img_bayesian"):
    os.makedirs("documentazione/res/drawable/img_bayesian")

# Feature e target per la predizione
X_supervised = [ # Colonne features # 15 colonne
    "Year",
    "Month",
    "Number of expected circulations",
    "Number of cancelled trains",
    "Number of late trains at departure",
    "Average travel time (min)",
    "Average delay of late departing trains (min)",
    "Average delay of all departing trains (min)",
    "% trains late due to external causes (weather, obstacles, suspicious packages, malevolence, social movements, etc.)",
    "% trains late due to railway infrastructure (maintenance, works)",
    "% trains late due to traffic management (rail line traffic, network interactions)",
    "% trains late due to passenger traffic (affluence, PSH management, connections)",
    "Delay due to external causes",
    "Delay due to railway infrastructure",
    "Delay due to traffic management"
]
target = 'Average delay of late arriving trains (min)' # Colonna target

# Colonne per l'analisi della congestione ferroviaria
X_bayesian = [
    "Year",
    "Month",
    "Number of expected circulations",
    "Average delay of all departing trains (min)",
    "Average delay of late arriving trains (min)",
    "Number of late trains at departure",
    "Number of trains late on arrival",
    "Average travel time (min)" # opzionale, per valutare i tempi di percorrenza
]

def main():
    """
    Esegui il Main del progetto ICON - Ingegneria della Conoscenza.
    
    Esegue le seguenti operazioni:
    1. Caricamento del dataset
    2. Preprocessing continuo del dataset
    3. Predizione della colonna target (incluso addestramento e valutazione dei modelli)
    4. Prolog: predizione della colonna target tramite regole Prolog
    5. Apprendimento bayesiano: analisi della congestione ferroviaria tramite apprendimento bayesiano
    
    Returns:
        None
    """
    # 1. Caricamento del dataset
    df = load_dataset("progettazione/dataset", "Regularities_by_liaisons_Trains_France.csv")

    # 2. Preprocessing continuo del dataset
    df_continuous, scaler = preprocess_continuous(df, X_supervised, target)

    # 3. Predizione
    random_forest = supervised_main(df_continuous[X_supervised], df_continuous[target])

    # 4. Prolog
    prolog_main(df_continuous[X_supervised], random_forest, target, scaler)

    # 5. Apprendimento bayesiano
    df_discrete = preprocess_discrete(df, X_bayesian)
    bayesian_main(df_discrete)

if __name__ == "__main__":
    print("----------------------------------------------------------------------------------------------")
    print("\nProgetto ICON - Ingegneria della Conoscenza\n")
    main()
    print("----------------------------------------------------------------------------------------------")
