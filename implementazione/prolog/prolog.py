import os
from pathlib import Path
from pyswip import Prolog

def main(dataset, features, random_forest):
    """Esegue la parte di Prolog del progetto."""
    # Esegui la conversione del modello in regole Prolog
    export_random_forest_to_prolog(random_forest, features)
    
    # Richiedi all'utente di scegliere una riga
    index = get_user_input(dataset)
    if index is None:
        return

    # Prepara i dati della riga selezionata
    feature_values = prepare_data_for_query(dataset, index)

    # Esegui la predizione tramite Prolog
    delay_predicted = query_prolog_delay(feature_values)

    # Visualizza la predizione
    display_prediction(delay_predicted)

    print("---------------------------------------\n")


def export_random_forest_to_prolog(model, X, filepath="implementazione/prolog/randomForest_rules.pl"):
    """Converte un modello Random Forest in regole Prolog e le salva in un file."""
    print("Conversione del modello in regole Prolog...")

    rules = tree_to_prolog_rules(model.estimators_[0], X, tree_id=0)

    with open(filepath, "w") as f:
        f.write("\n".join(rules))

    print(f"Conversione completata! Regole salvate in {filepath}")

def tree_to_prolog_rules(tree, X, tree_id):
    """Converte un albero di decisione in regole Prolog."""
    tree_ = tree.tree_
    rules = []

    def recurse(node, conditions):
        # Se il nodo è una foglia: non ha figli (i valori children_left e children_right sono uguali)
        if tree_.children_left[node] == tree_.children_right[node]:
            # Estraiamo il valore predetto (per regressione, consideriamo il primo elemento)
            predicted_value = tree_.value[node][0][0]
            # Prepariamo la testa del predicato.
            # Usiamo le feature come variabili: per farlo, convertiamo i nomi in minuscolo e senza spazi
            feature_vars = []
            for fname in X:
                var = fname.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "perc").replace(">", "gt").replace("<", "lt")
                feature_vars.append(var)
            head = f"delay_{tree_id}({', '.join(feature_vars)}, Delay)"

            # Se sono presenti condizioni, le uniamo con la virgola (in Prolog significa AND)
            if conditions:
                body = ", ".join(conditions)
                rule = f"{head} :- {body}, Delay is {predicted_value}."
            else:
                rule = f"{head} :- Delay is {predicted_value}."
            rules.append(rule)
        else:
            # Nodo decisionale: otteniamo l'indice della feature e la soglia per il test
            feature_index = tree_.feature[node]
            feature_name = X[feature_index]
            threshold = tree_.threshold[node]
            # Convertiamo il nome della feature in una variabile (stessa logica di prima)
            var = feature_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "perc").replace(">", "gt").replace("<", "lt")

            # Per il ramo sinistro: la condizione è che la feature sia minore della soglia
            left_condition = f"{var} < {threshold}"
            # Per il ramo destro: la condizione è che la feature sia maggiore o uguale alla soglia
            right_condition = f"{var} >= {threshold}"

            # Ricorsione per i due rami, aggiungendo la condizione corrente alla lista
            recurse(tree_.children_left[node], conditions + [left_condition])
            recurse(tree_.children_right[node], conditions + [right_condition])

    # Partiamo dalla radice (nodo 0) con nessuna condizione iniziale
    recurse(0, [])
    return rules

def get_user_input(dataset):
    '''Richiede all'utente di scegliere una riga da predire.'''
    print(f"Scegli una riga da predire (da 1 a {len(dataset)})")
    try:
        index = int(input("Inserisci l'indice della riga: ")) - 1
        
        # Verifica che l'indice sia valido
        if index < 0 or index >= len(dataset):
            print("Indice non valido. Riprova.")
            return None

        return index
    except ValueError:
        print("Per favore, inserisci un numero valido come indice.")
        return None

def prepare_data_for_query(dataset, index):
    """Prepara i dati della riga selezionata per la query Prolog."""
    print(f"\nPreparazione dei dati per la riga {index + 1}...")
    example_data = dataset.iloc[index]

    # Prepara i dati per la query (usiamo un dizionario con i valori delle feature)
    feature_values = {
        'average_travel_time__min': example_data['Average travel time (min)'],
        'number_of_cancelled_trains': example_data['Number of cancelled trains'],
        'number_of_late_trains_at_departure': example_data['Number of late trains at departure'],
        'average_delay_of_late_departing_trains__min': example_data['Average delay of late departing trains (min)'],
        'average_delay_of_all_departing_trains__min': example_data['Average delay of all departing trains (min)'],
        'number_of_trains_late_on_arrival': example_data['Number of trains late on arrival'],
        'average_delay_of_late_arriving_trains__min': example_data['Average delay of late arriving trains (min)'],
        '%_trains_late_due_to_external_causes': example_data['% trains late due to external causes'],
        '%_trains_late_due_to_railway_infrastructure': example_data['% trains late due to railway infrastructure'],
        '%_trains_late_due_to_traffic_management': example_data['% trains late due to traffic management'],
        '%_trains_late_due_to_rolling_stock': example_data['% trains late due to rolling stock'],
        '%_trains_late_due_to_station_management_and_reuse_of_material': example_data['% trains late due to station management and reuse of material'],
        '%_trains_late_due_to_passenger_traffic': example_data['% trains late due to passenger traffic'],
        'number_of_late_trains__greater_than_15min': example_data['Number of late trains > 15min'],
        'average_train_delay__greater_than_15min': example_data['Average train delay > 15min'],
        'number_of_late_trains__greater_than_30min': example_data['Number of late trains > 30min'],
        'number_of_late_trains__greater_than_60min': example_data['Number of late trains > 60min']
    }

    return feature_values

def query_prolog_delay(X, tree_id=0, prolog_file="implementazione/prolog/randomForest_rules.pl"):
    """Esegue una query Prolog per ottenere la predizione del ritardo."""
    print("Esecuzione della query Prolog...")

    prolog = Prolog()
    consult_prolog_file(prolog, prolog_file)

    # Prepara la query: ad esempio, se le feature sono:
    # "average_travel_time__min", "number_of_cancelled_trains", "number_of_late_trains_at_departure", ...
    # costruiamo la query come:
    # delay_0(Val1, Val2, Val3, ..., Delay).
    feature_vars = []
    for fname in X.keys():
        # Assumiamo che il nome usato per la variabile in Prolog sia già in forma "lower_case" e con "_" al posto degli spazi
        feature_vars.append(str(X[fname]))
    query_str = f"delay_{tree_id}({', '.join(feature_vars)}, Delay)"
    
    results = list(prolog.query(query_str))
    if results:
        return results[0]['Delay']
    else:
        return None

def display_prediction(delay_predicted):
    """Visualizza il risultato della predizione."""
    if delay_predicted is not None:
        print(f"Predizione del ritardo per i treni: {delay_predicted}")
    else:
        print("Non è stato possibile ottenere una predizione.")

# if __name__ == "__main__":
#     main()

def consult_prolog_file(prolog, prolog_file):
    """
    Tenta di consultare un file Prolog usando diversi metodi per risolvere il percorso.

    Metodi tentati:
      1. Converte il percorso in assoluto e sostituisce i backslash con slash.
      2. Usa il percorso relativo rispetto a __file__.
      3. Usa il percorso grezzo (raw).

    :param prolog: istanza di Prolog (pyswip)
    :param prolog_file: stringa con il percorso del file Prolog (relativo o assoluto)
    :return: il percorso usato se consult riuscito, altrimenti None
    """
    # Metodo 1: Converti in percorso assoluto e sostituisci i backslash con slash
    try:
        abs_path = str(Path(prolog_file).resolve()).replace("\\", "/")
        print(f"Tentativo 1: Uso percorso assoluto: {abs_path}")
        prolog.consult(abs_path)
        print("Consult riuscito con il percorso assoluto.")
        return abs_path
    except Exception as e:
        print("Tentativo 1 fallito con errore:", e)

    # Metodo 2: Usa il percorso relativo a __file__
    try:
        base_dir = Path(__file__).parent
        candidate = (base_dir / prolog_file).resolve()
        candidate_path = str(candidate).replace("\\", "/")
        print(f"Tentativo 2: Uso percorso relativo a __file__: {candidate_path}")
        prolog.consult(candidate_path)
        print("Consult riuscito con il percorso relativo a __file__.")
        return candidate_path
    except Exception as e:
        print("Tentativo 2 fallito con errore:", e)

    # Metodo 3: Usa il percorso grezzo (raw)
    try:
        print(f"Tentativo 3: Uso percorso grezzo: {prolog_file}")
        prolog.consult(prolog_file)
        print("Consult riuscito con il percorso grezzo.")
        return prolog_file
    except Exception as e:
        print("Tentativo 3 fallito con errore:", e)

    print("Consult non riuscito con nessun metodo.")
    return None
