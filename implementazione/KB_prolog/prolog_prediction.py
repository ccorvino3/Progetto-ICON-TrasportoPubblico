from pyswip import Prolog
from utils.dataset import inverse_transform_column, convert_float_to_time_format as c_time_f
import numpy as np

# Per evitare che i numeri siano stampati in notazione scientifica
np.set_printoptions(suppress=True)

# File prolog per forza presenti nella working directory se no la funzione consult dà errore di path
FILE_FACTS = "random_forest_facts.pl"
FILE_MODEL = "model_delay_rules.pl"

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
    print(f"Ritardo predetto per\n\ttrain1 (df[0]) -> {c_time_f(delay1_denormalized)}\n\ttrain2 (df[1]) -> {c_time_f(delay2_denormalized)}\n")

    dtree_pred_1 = tree.predict([X.iloc[0].values])
    dtree_pred_2 = tree.predict([X.iloc[1].values])
    dtree_pred_1_denormalized = inverse_transform_column(dtree_pred_1, scaler, target)
    dtree_pred_2_denormalized = inverse_transform_column(dtree_pred_2, scaler, target)
    print(f"Scikit prediction: df[0] = \n{X.iloc[0].values} -> {c_time_f(dtree_pred_1_denormalized)}")
    print(f"Scikit prediction: df[1] = \n{X.iloc[1].values} -> {c_time_f(dtree_pred_2_denormalized)}\n")
    
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

# if __name__ == "__main__":
#     main()
