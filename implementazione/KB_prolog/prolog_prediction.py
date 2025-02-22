from pyswip import Prolog
from utils.dataset import inverse_transform_column, convert_float_to_time_format as c_time_f
import numpy as np

# Per evitare che i numeri siano stampati in notazione scientifica
np.set_printoptions(suppress=True)

# File prolog per forza presenti nella working directory se no la funzione consult dà errore di path
FILE_FACTS = "random_forest_facts.pl"
FILE_MODEL = "model_delay.pl"

def main(X, random_forest, target, scaler):
    # Esegui la conversione dell'albero in regole Prolog
    tree = random_forest.estimators_[0]
    convert_tree_in_fact(tree)
    
    # Init Prolog e i due treni di esempio
    prolog = Prolog()
    prolog.consult(FILE_FACTS)
    prolog.consult(FILE_MODEL)
    train1 = list(X.iloc[0].values.astype(float))
    train2 = list(X.iloc[1].values.astype(float))

    # Query per ottenere il ritardo predetto per i due treni di esempio
    delay1 = query_predicted_delay(prolog, train1)
    delay2 = query_predicted_delay(prolog, train2)
    delay1_denormalized = inverse_transform_column(delay1, scaler, target)
    delay2_denormalized = inverse_transform_column(delay2, scaler, target)
    print(f"Ritardo predetto per\n\ttrain1 -> {c_time_f(delay1_denormalized)}\n\tper train2 -> {c_time_f(delay2_denormalized)}\n")

    dtree_pred_1 = tree.predict([X.iloc[0].values])
    dtree_pred_2 = tree.predict([X.iloc[1].values])
    dtree_pred_1_denormalized = inverse_transform_column(dtree_pred_1, scaler, target)
    dtree_pred_2_denormalized = inverse_transform_column(dtree_pred_2, scaler, target)
    print(f"Scikit prediction: df[0] = {X.iloc[0].values} -> {c_time_f(dtree_pred_1_denormalized)}")
    print(f"Scikit prediction: df[1] = {X.iloc[1].values} -> {c_time_f(dtree_pred_2_denormalized)}\n")
    
    # Query per individuare il treno con il ritardo massimo
    best_train, best_delay = query_max_delay(prolog, [train1, train2])
    best_delay_denormalized = inverse_transform_column(best_delay, scaler, target)
    print(f"Treno {best_train} con il ritardo massimo {c_time_f(best_delay_denormalized)}")
    
    print("Predizione in Prolog e KB completata.")
    print("---------------------------------------\n")

def convert_tree_in_fact(estimator, filepath=FILE_FACTS):
    with open(filepath, 'w') as f:
        # Scrivere l'intestazione del file Prolog
        f.write("% Predizione dell'albero Random Forest\n")

        # Estrai l'albero
        tree = estimator.tree_

        # Scrivere nodi intermedi
        for node_id in range(tree.node_count):
            # I nodi intermedi sono quelli con almeno un figlio a sinistra o destra
            if tree.children_left[node_id] != tree.children_right[node_id]:
                feature_index = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]

                f.write(f"node({node_id}, {feature_index}, {threshold}, {left_child}, {right_child}).\n")

        # Scrivere le foglie (nodi terminali)
        for node_id in range(tree.node_count):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Valore predetto: media del target in questa foglia
                predicted_value = tree.value[node_id][0, 0]
                f.write(f"leaf({node_id}, {predicted_value}).\n")

    print(f"File Prolog generato: {filepath}")

def query_predicted_delay(prolog, features):
    # Converte la lista Python in una lista Prolog
    features_str = '[' + ','.join(str(x) for x in features) + ']'
    query_str = f"predire_ritardo({features_str}, Ritardo)."
    
    result = list(prolog.query(query_str))
    if result:
        return result[0]['Ritardo']
    else:
        return None

def query_max_delay(prolog, trains):    
    # Costruisce la rappresentazione della lista Prolog dei treni
    trains_str = '[' + ','.join('[' + ','.join(str(x) for x in train) + ']' for train in trains) + ']'
    query_str = f"treno_piu_in_ritardo({trains_str}, Treno, Ritardo)."
    
    result = list(prolog.query(query_str))
    if result:
        return result[0]['Treno'], result[0]['Ritardo']
    else:
        return None, None

# TODO: Devo capire se da qui in poi posso cancellarlo oppure mi sarà utile più avanti

def main2(dataset, features, random_forest):
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

def export_random_forest_to_prolog(model, X, filepath="randomForest_rules.pl"):
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

def query_prolog_delay(X, tree_id=0, filepath="randomForest_rules.pl"):
    """Esegue una query Prolog per ottenere la predizione del ritardo."""
    print("Esecuzione della query Prolog...")

    prolog = Prolog()
    prolog.consult(filepath)

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
