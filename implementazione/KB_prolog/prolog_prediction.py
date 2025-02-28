"""
Questo script contiene le funzioni per la predizione del ritardo dei treni in Prolog.

Include la conversione dell'albero Random Forest in regole Prolog, la predizione del ritardo per due treni di esempio,
la ricerca del treno con il ritardo massimo e la verifica se un ritardo proposto è conveniente rispetto a quello predetto.

Autore: Christian Corvino
Data: 26/02/2025
"""

from pyswip import Prolog
from utils.dataset import inverse_transform_column, convert_float_to_time_format as c_time_f, normalize_value
import numpy as np

# Per evitare che i numeri siano stampati in notazione scientifica
np.set_printoptions(suppress=True)

# File prolog per forza presenti nella working directory se no la funzione consult dà errore di path
PL_FACTS_TREE = "random_forest_facts.pl"
PL_RULES_TREE = "model_delay_rules.pl"
PL_RULES = "trains_delay_rules.pl"

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
    from_prolog(prolog, X, target, scaler)
    from_tree(prolog, X, tree, target, scaler)

    print("Predizione in Prolog e KB completata.")
    print("---------------------------------------\n")

def from_prolog(prolog, X, target, scaler):
    """
    Esegue la predizione del ritardo dei treni in Prolog.
    
    - Predice il ritardo per dei treni di esempio.
    
    - Individua il treno con il ritardo massimo.
    
    - Verifica se un ritardo proposto è conveniente rispetto a quello predetto.
    
    Args:
        prolog (Prolog): L'istanza di Prolog.
        X (DataFrame): Il DataFrame contenente le features dei treni.
        target (str): Il nome della colonna target.
        scaler (StandardScaler): Lo scaler utilizzato per normalizzare i dati delle features.
    
    Returns:
        None
    """
    print(f"Loading {PL_RULES}...")
    prolog.consult(PL_RULES)
    
    train3 = list(X.iloc[2].values.astype(float))
    train4 = list(X.iloc[3].values.astype(float))
    train5 = list(X.iloc[4].values.astype(float))

    # Ripristina i valori originali delle righe di esempio
    X_3 = [inverse_transform_column(val, scaler, col) for val, col in zip(X.iloc[2].values, X.columns)]
    X_4 = [inverse_transform_column(val, scaler, col) for val, col in zip(X.iloc[3].values, X.columns)]
    X_5 = [inverse_transform_column(val, scaler, col) for val, col in zip(X.iloc[4].values, X.columns)]
    print(f"Date le features dei treni di esempio:\n{X_3}\n{X_4}\n{X_5}\n")
    # Predici il ritardo per i treni di esempio
    delay3 = query_predicted_delay(prolog, train3)
    delay4 = query_predicted_delay(prolog, train4)
    delay5 = query_predicted_delay(prolog, train5)
    print(f"Prolog prediction:\n" +
          f"df[2] -> {c_time_f(inverse_transform_column(delay3, scaler, target))}\n" +
          f"df[3] -> {c_time_f(inverse_transform_column(delay4, scaler, target))}\n" +
          f"df[4] -> {c_time_f(inverse_transform_column(delay5, scaler, target))}\n"
    )

    # Query per individuare il treno con il ritardo massimo
    _, best_delay = query_max_delay(prolog, [train3, train4, train5])
    print(f"Treno con il ritardo massimo {c_time_f(inverse_transform_column(best_delay, scaler, target))}\n")

    # Query per verificare se un ritardo proposto è conveniente rispetto a quello predetto
    proposed_delay = 20 # Un valore ipotetico per il ritardo proposto
    proposed_delay_normalized = normalize_value(proposed_delay, scaler, target)
    predetto, is_convenient = query_best_delay(prolog, train3, proposed_delay_normalized)
    predetto_denormalized = c_time_f(inverse_transform_column(predetto, scaler, target))
    if is_convenient:
        print(f"Il ritardo proposto {proposed_delay} è più conveniente di quello predetto ({predetto_denormalized}).\n")
    else:
        print(f"Il ritardo proposto {proposed_delay} NON è più conveniente di quello predetto ({predetto_denormalized}).\n")

    print(f"Unloading {PL_RULES}...\n")
    list(prolog.query(f"unload_file('{PL_RULES}')."))

def from_tree(prolog, X, tree, target, scaler):
    """
    Esegue la predizione del ritardo dei treni in Prolog considerando l'albero di decisione.
    
    - Predice il ritardo per dei treni di esempio.
    
    - Individua il treno con il ritardo massimo.
    
    - Verifica se un ritardo proposto è conveniente rispetto a quello predetto.
    
    Args:
        prolog (Prolog): L'istanza di Prolog.
        X (DataFrame): Il DataFrame contenente le features dei treni.
        tree (DecisionTreeRegressor): L'albero di decisione addestrato.
        target (str): Il nome della colonna target.
        scaler (StandardScaler): Lo scaler utilizzato per normalizzare i dati delle features.
    
    Returns:
        None
    """
    print(f"Loading {PL_FACTS_TREE} and {PL_RULES_TREE}...")
    prolog.consult(PL_FACTS_TREE)
    prolog.consult(PL_RULES_TREE)

    train1 = list(X.iloc[0].values.astype(float))
    train2 = list(X.iloc[1].values.astype(float))

    # Ripristina i valori originali delle righe di esempio
    X_0 = [inverse_transform_column(val, scaler, col) for val, col in zip(X.iloc[0].values, X.columns)]
    X_1 = [inverse_transform_column(val, scaler, col) for val, col in zip(X.iloc[1].values, X.columns)]
    print(f"Date le features dei treni di esempio:\n{X_0}\n{X_1}\n")
    # Predici il ritardo per i treni di esempio
    # Prima in Prolog
    delay1 = query_predicted_delay(prolog, train1)
    delay2 = query_predicted_delay(prolog, train2)
    print(f"Prolog prediction:\n" +
          f"df[0] -> {c_time_f(inverse_transform_column(delay1, scaler, target))}\n" +
          f"df[1] -> {c_time_f(inverse_transform_column(delay2, scaler, target))}\n"
    )
    # Poi in Scikit
    dtree_pred_1 = tree.predict([X.iloc[0].values])
    dtree_pred_2 = tree.predict([X.iloc[1].values])
    print(f"Scikit prediction:\n" +
          f"df[0] -> {c_time_f(inverse_transform_column(dtree_pred_1, scaler, target))}\n" +
          f"df[1] -> {c_time_f(inverse_transform_column(dtree_pred_2, scaler, target))}\n"
    )

    # Query per individuare il treno con il ritardo massimo
    _, best_delay = query_max_delay(prolog, [train1, train2])
    print(f"Treno con il ritardo massimo {c_time_f(inverse_transform_column(best_delay, scaler, target))}\n")

    # Query per verificare se un ritardo proposto è conveniente rispetto a quello predetto
    proposed_delay = 20  # Un valore ipotetico per il ritardo proposto
    predetto, is_convenient = query_best_delay(prolog, train1, normalize_value(proposed_delay, scaler, target))
    predetto_denormalized = c_time_f(inverse_transform_column(predetto, scaler, target))
    if is_convenient:
        print(f"Il ritardo proposto {proposed_delay} è più conveniente di quello predetto ({predetto_denormalized}).\n")
    else:
        print(f"Il ritardo proposto {proposed_delay} NON è più conveniente di quello predetto ({predetto_denormalized}).\n")

    print(f"Unloading {PL_FACTS_TREE} and {PL_RULES_TREE}...\n")
    list(prolog.query(f"unload_file('{PL_FACTS_TREE}')."))
    list(prolog.query(f"unload_file('{PL_RULES_TREE}')."))

def convert_tree_in_fact(estimator, filepath=PL_FACTS_TREE):
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
        tuple: Una tupla contenente:
            - str: Il ritardo predetto.
            - bool: True se il ritardo proposto è conveniente, False altrimenti.
    """
    # Converte la lista Python in una lista Prolog
    features_str = '[' + ','.join(str(x) for x in features) + ']'
    
    # Formatta la query per Prolog
    query_str = f"ritardo_conveniente({features_str}, {proposed_delay}, RitardoPredetto, Conveniente)."
    
    # Esegui la query
    result = list(prolog.query(query_str))
    if result:
        ritardo_predetto = result[0]['RitardoPredetto']
        conveniente = result[0]['Conveniente']
        if isinstance(conveniente, str):
            conveniente = (conveniente.lower() == "true")
        return ritardo_predetto, conveniente
    else:
        return None, None

# if __name__ == "__main__":
#     main()
