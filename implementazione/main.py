from utils.dataset import load_dataset
from utils.preprocessing import preprocess_continuous, preprocess_discrete
from supervised_learning.supervised_prediction import main as supervised_main
from KB_prolog.prolog_prediction import main as prolog_main
from bayesian_learning.bayesian_prediction import main as bayesian_main

# Per eseguirlo su vscode: comando_py implementazione/main.py
# - nel mio caso: py implementazione/main.py

# Feature e target per la predizione
features = [ # Lista delle feature (19 / 26=(32 totali - 5 rimosse) - 1 (target) colonne)
    'Average travel time (min)', 'Number of cancelled trains', 
    'Number of late trains at departure', 'Number of trains late on arrival',
    'Average delay of late departing trains (min)', 'Average delay of all departing trains (min)', 'Average delay of late arriving trains (min)', 
    '% trains late due to external causes', '% trains late due to railway infrastructure', 
    '% trains late due to traffic management', '% trains late due to rolling stock',
    '% trains late due to station management and reuse of material', '% trains late due to passenger traffic',
    'Number of late trains > 15min', 'Number of late trains > 30min', 'Number of late trains > 60min',
    'Average train delay > 15min', 'Year', 'Month'
]
target = 'Average delay of all arriving trains (min)' # Colonna target

# Colonne per l'analisi della congestione ferroviaria
columns_analysis = [
    "Year",
    "Month",
    "Number of expected circulations",
    "Average delay of all departing trains (min)",
    "Average delay of all arriving trains (min)",
    "Number of late trains at departure",
    "Number of trains late on arrival",
    "Average travel time (min)" # opzionale, per valutare i tempi di percorrenza
]

def main():
    # 1. Caricamento del dataset
    df = load_dataset("progettazione/dataset", "Regularities_by_liaisons_Trains_France.csv")

    # 2. Preprocessing continuo del dataset
    df_continuous, scaler = preprocess_continuous(df)

    # 3. Predizione
    random_forest = supervised_main(df_continuous[features], df_continuous[target])

    # 4. Prolog
    prolog_main(df_continuous[features], random_forest, target, scaler)

    # 5. Apprendimento bayesiano
    df_discrete = preprocess_discrete(df, columns_analysis)
    bayesian_main(df_discrete)

if __name__ == "__main__":
    print("----------------------------------------------------------------------------------------------")
    print("\nProgetto ICON - Ingegneria della Conoscenza\n")
    main()
    print("----------------------------------------------------------------------------------------------")
