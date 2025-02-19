from matplotlib import pyplot as plt
import networkx as nx
from pgmpy.models import BayesianNetwork as Bayesian
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

import numpy as np
np.set_printoptions(suppress=True) # Per evitare che i numeri vengano stampati in notazione scientifica

def main(df):
    # Definizione della struttura della rete bayesiana più complessa
    model = bayesian_learning(df)
    print("1. Rete bayesiana addestrata con successo.")

    # Visualizzazione della rete bayesiana
    show_bayesian_network(model)
    print("2. Visualizzazione completata.")

    # Grafico di inferenza
    # Esempio di inferenza
    infer = VariableElimination(model)
    evidence = {
        'Month': 4,
        'Average travel time (min)': 2,
        '% trains late due to external causes': 2,
        '% trains late due to railway infrastructure': 1,
        '% trains late due to traffic management': 1,
        '% trains late due to rolling stock': 1,
        '% trains late due to station management': 1,
        '% trains late due to passenger traffic': 0
    }
    plot_inference_results(infer, evidence)
    print("3. Grafico di inferenza completato.")

    prediction = infer.query(variables=['Late > 15 min'], evidence=evidence)
    print(f"Prediction (evidence={evidence}): {prediction.values[1]  * 100:.2f}% chance of being late > 15 min")
    print("4. Predizione completata.")
    print("---------------------------------------\n")

def bayesian_learning(df):
    print("Addestramento della rete bayesiana...")
    model = Bayesian([
        # Relazioni tra il mese e le cause di ritardo
        ('Month', '% trains late due to external causes'),
        ('Month', '% trains late due to railway infrastructure'),
        ('Month', '% trains late due to traffic management'),
        ('Month', '% trains late due to rolling stock'),
        ('Month', '% trains late due to station management'),
        ('Month', '% trains late due to passenger traffic'),

        # Relazione tra il mese e il tempo di viaggio medio
        ('Month', 'Average travel time (min)'),

        # Relazioni tra le cause di ritardo e il ritardo maggiore di 15 minuti
        ('% trains late due to external causes', 'Late > 15 min'),
        ('% trains late due to railway infrastructure', 'Late > 15 min'),
        ('% trains late due to traffic management', 'Late > 15 min'),
        ('% trains late due to rolling stock', 'Late > 15 min'),
        ('% trains late due to station management', 'Late > 15 min'),
        ('% trains late due to passenger traffic', 'Late > 15 min'),

        # Relazione tra ritardi passeggeri e gestione della stazione
        ('% trains late due to passenger traffic', '% trains late due to station management'),

        # Altri possibili legami
        ('% trains late due to rolling stock', '% trains late due to railway infrastructure'),
        ('% trains late due to station management', '% trains late due to rolling stock'),

        # Relazioni tra il tempo di viaggio medio e le cause di ritardo
        ('Average travel time (min)', '% trains late due to external causes'),
        ('Average travel time (min)', '% trains late due to traffic management'),
        ('Average travel time (min)', '% trains late due to railway infrastructure'),
        ('Average travel time (min)', '% trains late due to rolling stock'),

        # Dipendenza tra il tempo di viaggio e il ritardo per la gestione del traffico
        ('Average travel time (min)', '% trains late due to traffic management')
    ])

    # Stima delle probabilità condizionate (CPD) con Maximum Likelihood
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    
    return model

def show_bayesian_network(model):
    print("Visualizzazione della rete bayesiana in layout circolare...")
    graph = nx.DiGraph()
    graph.add_edges_from(model.edges())

    # Visualizzare la rete bayesiana con layout circolare
    plt.figure(figsize=(10, 8))
    pos = nx.circular_layout(graph)  # Layout circolare
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', 
            font_size=10, font_weight='bold', arrows=True)
    plt.title("Rete Bayesiana del Ritardo dei Treni", fontsize=15)
    plt.show()

def plot_inference_results(infer, evidence):
    # Inferenza sulla probabilità di 'Late > 15 min'
    prediction = infer.query(variables=['Late > 15 min'], evidence=evidence)

    # Estrazione dei dati per il grafico
    labels = prediction.values  # Valori di probabilità per 'Late > 15 min'
    outcome = prediction.state_names['Late > 15 min']  # Stati possibili (0 o 1)

    # Creazione del grafico
    plt.bar(outcome, labels, color='skyblue')
    plt.xlabel("Late > 15 min (Yes/No)")
    plt.ylabel("Probabilità")
    plt.title("Probabilità Predetta di Ritardo Maggiore di 15 Minuti")
    plt.show()

# if __name__ == "__main__":
#     main()
