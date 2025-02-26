"""
Questo script contiene le funzioni per la predizione del ritardo dei treni in Prolog.

Include la conversione dell'albero Random Forest in regole Prolog, la predizione del ritardo per due treni di esempio,
la ricerca del treno con il ritardo massimo e la verifica se un ritardo proposto è conveniente rispetto a quello predetto.

Autore: Christian Corvino
Data: 26/02/2025
"""

from pyswip import Prolog
from utils.dataset import inverse_transform_column, convert_float_to_time_format as c_time_f
import numpy as np

# Per evitare che i numeri siano stampati in notazione scientifica
np.set_printoptions(suppress=True)

# File prolog per forza presenti nella working directory se no la funzione consult dà errore di path
PL_FACTS = "random_forest_facts.pl"
PL_RULES = "model_delay_rules.pl"

def main(X, random_forest, target, scaler):
    """
    Esegue la predizione del ritardo dei treni in Prolog.
    
    Args:
        X (DataFrame): Il DataFrame contenente le features dei treni.
        random_forest (RandomForestRegressor): Il modello Random Forest addestrato.
        target (str): Il nome della colonna target.
        scaler (StandardScaler): Lo scaler utilizzato per normalizzare i dati delle features.
    
    Returns:
        None
    """
    # Esegui la conversione dell'albero in regole Prolog
    tree = random_forest.estimators_[0]
    convert_tree_in_fact(tree)

    # Init Prolog e i due treni di esempio
    prolog = Prolog()
    prolog.consult(PL_FACTS)
    prolog.consult(PL_RULES)
    train1 = list(X.iloc[0].values.astype(float))
    train2 = list(X.iloc[1].values.astype(float))

    # Ripristina i valori originali delle righe di esempio
    X_0 = [inverse_transform_column(val, scaler, col) for val, col in zip(X.iloc[0].values, X.columns)]
    X_1 = [inverse_transform_column(val, scaler, col) for val, col in zip(X.iloc[1].values, X.columns)]
    print(f"Date le features dei due treni di esempio:\n{X_0}\n{X_1}\n")
    # Predici il ritardo per i due treni di esempio
    # Prima in Prolog
    delay1 = query_predicted_delay(prolog, train1)
    delay2 = query_predicted_delay(prolog, train2)
    delay1_denormalized = inverse_transform_column(delay1, scaler, target)
    delay2_denormalized = inverse_transform_column(delay2, scaler, target)
    print(f"Prolog prediction:\ndf[0] -> {c_time_f(delay1_denormalized)}\ndf[1] -> {c_time_f(delay2_denormalized)}\n")
    # Poi in Scikit
    dtree_pred_1 = tree.predict([X.iloc[0].values])
    dtree_pred_2 = tree.predict([X.iloc[1].values])
    dtree_pred_1_denormalized = inverse_transform_column(dtree_pred_1, scaler, target)
    dtree_pred_2_denormalized = inverse_transform_column(dtree_pred_2, scaler, target)
    print(f"Scikit prediction:\ndf[0] -> {c_time_f(dtree_pred_1_denormalized)}\ndf[1] -> {c_time_f(dtree_pred_2_denormalized)}\n")

    # Query per individuare il treno con il ritardo massimo
    _, best_delay = query_max_delay(prolog, [train1, train2])
    best_delay_denormalized = inverse_transform_column(best_delay, scaler, target)
    print(f"Treno con il ritardo massimo {c_time_f(best_delay_denormalized)}\n")

    # Query per verificare se un ritardo proposto è conveniente rispetto a quello predetto
    proposed_delay = 20.0  # Un valore ipotetico per il ritardo proposto
    is_convenient = query_best_delay(prolog, train1, proposed_delay)
    if is_convenient:
        print(f"Il ritardo proposto {proposed_delay} è più conveniente di quello predetto.\n")
    else:
        print(f"Il ritardo proposto {proposed_delay} NON è più conveniente di quello predetto.\n")

    print("Predizione in Prolog e KB completata.")
    print("---------------------------------------\n")

def convert_tree_in_fact(estimator, filepath=PL_FACTS):
    """
    Converti un albero di decisione in regole Prolog e le scrive in un file.
    
    Args:
        estimator (DecisionTreeRegressor): L'albero di decisione addestrato.
        filepath (str): Il percorso del file in cui scrivere le regole Prolog.
    
    Returns:
        None
    """
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
    """
    Interroga Prolog per predire il ritardo di un treno.
    
    Args:
        prolog (Prolog): L'istanza di Prolog.
        features (list): La lista di features del treno.
    
    Returns:
        float: Il ritardo predetto.
    """
    # Converte la lista Python in una lista Prolog
    features_str = '[' + ','.join(str(x) for x in features) + ']'
    query_str = f"predire_ritardo({features_str}, Ritardo)."
    
    result = list(prolog.query(query_str))
    if result:
        return result[0]['Ritardo']
    else:
        return None

def query_max_delay(prolog, trains):
    """
    Interroga Prolog per trovare il treno con il ritardo massimo.
    
    Args:
        prolog (Prolog): L'istanza di Prolog.
        trains (list): La lista di treni.
    
    Returns:
        tuple: Una tupla contenente:
            - str: Il treno con il ritardo massimo.
            - float: Il ritardo massimo.
    """
    # Costruisce la rappresentazione della lista Prolog dei treni
    trains_str = '[' + ','.join('[' + ','.join(str(x) for x in train) + ']' for train in trains) + ']'
    query_str = f"treno_piu_in_ritardo({trains_str}, Treno, Ritardo)."
    
    result = list(prolog.query(query_str))
    if result:
        return result[0]['Treno'], result[0]['Ritardo']
    else:
        return None, None

def query_best_delay(prolog, features, proposed_delay=20.0):
    """
    Interroga Prolog per verificare se il ritardo proposto è più conveniente rispetto a quello predetto.
    
    Args:
        prolog (Prolog): L'istanza di Prolog.
        features (list): La lista di features del treno.
        proposed_delay (float): Il ritardo proposto.
    
    Returns:
        bool: True se il ritardo proposto è conveniente, False altrimenti.
    """
    # Converte la lista Python in una lista Prolog
    features_str = '[' + ','.join(str(x) for x in features) + ']'
    
    # Formatta la query per Prolog
    query_str = f"ritardo_conveniente({features_str}, {proposed_delay})."
    
    # Esegui la query
    result = list(prolog.query(query_str))
    
    return bool(result)  # Se la lista non è vuota, il ritardo proposto è conveniente

# if __name__ == "__main__":
#     main()
