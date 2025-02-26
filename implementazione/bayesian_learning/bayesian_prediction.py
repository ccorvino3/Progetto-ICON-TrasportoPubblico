"""
Questo modulo contiene le funzioni per addestrare una rete bayesiana sui dati forniti,.

visualizzare la rete risultante e fare inferenza su una variabile target.

Autore: Christian Corvino
Data: 26/02/2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# Per evitare che i numeri vengano stampati in notazione scientifica
np.set_printoptions(suppress=True)

def main(df):
    """
    Esegue l'apprendimento bayesiano sul dataset fornito e visualizza i risultati.
    
    Args:
        df (pd.DataFrame): Il DataFrame contenente i dati.
    
    Returns:
        None
    """
    # Definizione della struttura della rete bayesiana
    model = bayesian_learning(df)
    print("1. Rete bayesiana addestrata con successo.")

    # Visualizzazione della rete bayesiana
    display_bayesian_graph(model)
    print("2. Visualizzazione completata.")

    # Inferenza sulla rete bayesiana e grafico a barre
    infer_bayesian(model, df, "Number of trains late on arrival", row_index=0)
    print("3. Inferenza completata.")

    print("---------------------------------------\n")

def bayesian_learning(df):
    """
    Esegue l'apprendimento bayesiano sul dataset fornito.
    
    Crea il modello bayesiano, addestra le Conditional Probability Distributions (CPD) e verifica la validità del modello.

    Args:
        df (pd.DataFrame): Il DataFrame contenente i dati.
    
    Returns:
        BayesianNetwork: Il modello bayesiano addestrato.
    """
    # Creazione del modello bayesiano
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=BicScore(df))
    model = BayesianNetwork(best_model.edges())

    cpds = []
    for node in df.columns:
        parents = list(model.get_parents(node))
        
        # Ottieni gli stati possibili per il nodo.
        # Se possibile, li converto in int, altrimenti li lascio com'è.
        try:
            possible_states = list(map(int, sorted(df[node].unique())))
        except Exception:
            possible_states = sorted(df[node].unique())
        num_states = len(possible_states)
        
        if not parents:
            # CPD marginale per variabili senza genitori
            prob_series = df[node].value_counts(normalize=True)
            try:
                prob_series.index = list(map(int, prob_series.index))
            except Exception:
                pass
            prob_series = prob_series.reindex(possible_states, fill_value=0)
            prob_values = prob_series.values.astype(float)
            total = prob_values.sum()
            if total != 0:
                prob_values /= total
            else:
                prob_values = np.full(prob_values.shape, 1.0 / num_states)
            cpd = TabularCPD(variable=node, variable_card=num_states,
                             values=prob_values.reshape(-1, 1))
        else:
            # CPD condizionata per variabili con genitori
            # Per ogni genitore, otteniamo i possibili stati (convertendo in int se possibile)
            parents_states = {}
            for parent in parents:
                try:
                    parents_states[parent] = list(map(int, sorted(df[parent].unique())))
                except Exception:
                    parents_states[parent] = sorted(df[parent].unique())
            parent_card = [len(parents_states[parent]) for parent in parents]
            
            # Raggruppa per i genitori e calcola la distribuzione normalizzata
            # Il risultato di groupby:
            #   - Index: combinazioni dei genitori
            #   - Colonne: valori unici della variabile (node)
            grouped = df.groupby(parents)[node].value_counts(normalize=True).unstack(fill_value=0)
            # Trasponi in modo da avere:
            #   - Index: valori della variabile (node)
            #   - Colonne: combinazioni dei genitori
            grouped = grouped.T
            
            # Reindicizza le righe con i possibili stati
            reindexed_rows = grouped.reindex(possible_states, fill_value=0)
            
            # Crea il MultiIndex completo per le colonne dai possibili stati dei genitori
            new_columns = pd.MultiIndex.from_product(
                [parents_states[parent] for parent in parents],
                names=parents
            )
            reindexed_df = reindexed_rows.reindex(columns=new_columns, fill_value=0)
            
            cpd_values = reindexed_df.values.astype(float)
            
            # Normalizza ogni colonna: se la somma è zero, assegna distribuzione uniforme
            sum_values = cpd_values.sum(axis=0, keepdims=True)
            for j in range(cpd_values.shape[1]):
                if sum_values[0, j] == 0:
                    cpd_values[:, j] = 1.0 / num_states
                else:
                    cpd_values[:, j] /= sum_values[0, j]
                    
            cpd = TabularCPD(variable=node, variable_card=num_states, values=cpd_values,
                             evidence=parents, evidence_card=parent_card)
            
        cpds.append(cpd)

    # Aggiungi tutte le CPD al modello
    model.add_cpds(*cpds)

    # Verifica la validità del modello
    try:
        model.check_model()
        print("\n✅ Il modello è valido.")
    except ValueError as e:
        print("\n❌ Errore nel modello bayesiano:")
        print(e)

    return model


def display_bayesian_graph(model):
    """
    Visualizza la rete bayesiana addestrata con layout circolare.
    
    Salva l'immagine in documentazione/res/drawable/img_bayesian/bayesian_network.png.
    
    Args:
        model (BayesianNetwork): Il modello bayesiano addestrato.
    
    Returns:
        None
    """
    print("Visualizzazione della rete bayesiana in layout circolare...")
    graph = nx.DiGraph()
    graph.add_edges_from(model.edges())

    # Visualizzare la rete bayesiana con layout circolare
    plt.figure(figsize=(10, 8))
    pos = nx.circular_layout(graph)  # Layout circolare
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', 
            font_size=10, font_weight='bold', arrows=True)
    plt.title("Rete Bayesiana del Ritardo dei Treni", fontsize=15)
    # plt.show()
    plt.savefig("documentazione/res/drawable/img_bayesian/bayesian_network.png")
    plt.close()
    print("Salvato in documentazione/res/drawable/img_bayesian/bayesian_network.png")

def infer_bayesian(model, df, target_var, row_index=0):
    """
    Esegue l'inferenza sulla rete bayesiana addestrata.

    Visualizza la distribuzione a posteriori per la variabile target data una riga di esempio.
    
    Args:
        model (BayesianNetwork): Il modello bayesiano addestrato.
        df (pd.DataFrame): Il DataFrame contenente i dati.
        target_var (str): La variabile target per l'inferenza.
        row_index (int): L'indice della riga di esempio nel DataFrame.
    
    Returns:
        TabularCPD: La distribuzione a posteriori per la variabile target.
    """
    print("3. Inferenza sulla rete bayesiana...")

    # Eseguiamo l'inferenza
    inference = VariableElimination(model)

    # Selezioniamo la riga come esempio
    example = df.iloc[row_index].to_dict()

    # Prepariamo le evidenze eliminando il target
    evidence = example.copy()
    evidence.pop(target_var, None)

    # Eseguiamo la query per ottenere la distribuzione a posteriori sul target
    query_result = inference.query([target_var], evidence=evidence)

    # Visualizziamo i risultati
    print(f"Inferenza per il target '{target_var}' data la tupla esempio:\n{query_result}")
    plot_inference_results(query_result, target_var, row_index)

    return query_result

def plot_inference_results(query_result, target_var, row_index):
    """
    Visualizza la distribuzione a posteriori per la variabile target data una riga di esempio.
    
    Salva l'immagine in documentazione/res/drawable/img_bayesian/posterior_distribution.png.
    
    Args:
        query_result (TabularCPD): La distribuzione a posteriori per la variabile target.
        target_var (str): La variabile target per l'inferenza.
        row_index (int): L'indice della riga di esempio nel DataFrame.
    
    Returns:
        None
    """
    print(f"Visualizzazione della distribuzione a posteriori per '{target_var}'...")
    factor = query_result
    states = list(range(factor.cardinality[0]))

    pd.Series(factor.values, index=states).plot(kind='bar', color='skyblue')
    plt.title(f"Distribuzione: '{target_var}' ({row_index} riga)")
    plt.xlabel("Valore discretizzato")
    plt.ylabel("Probabilità")
    # plt.show()
    plt.savefig("documentazione/res/drawable/img_bayesian/posterior_distribution.png")
    plt.close()
    print("Salvato in documentazione/res/drawable/img_bayesian/posterior_distribution.png")

# if __name__ == "__main__":
#     main()
