"""
Predizione dei ritardi dei treni in arrivo utilizzando modelli di Machine Learning.

Include l'addetramento e la valutazione di diversi modelli di regressione e la generazione di grafici per la valutazione.

Autore: Christian Corvino
Data: 26/02/2025
"""

import matplotlib
matplotlib.use('Agg')  # Usa un backend che non dipende da Tkinter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, learning_curve, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Per evitare che i numeri vengano stampati in notazione scientifica
np.set_printoptions(suppress=True) # Per evitare che i numeri vengano stampati in notazione scientifica

# Per la previsione:
# - y_pred = gridsearch.best_model_.predict(X_test)
#   ||
# - y_pred = gridsearch.best_estimator_.predict(X_test)

def main(X, y):
    """
    Funzione principale per la predizione dei ritardi dei treni in arrivo.
    
    Args:
        X (DataFrame): Features per la predizione.
        y (Series): Target per la predizione.
    
    Returns:
        dict: Random Forest addestrato e valutato da usare per Prolog.
    """
    print("Predizione in corso...")

    # Addestrare i modelli
    models, kfold = train_models(X, y)
    print("1. Modelli addestrati.")

    # Valutare i modelli
    evaluation_metrics = evaluate_models(models, X, y, kfold)
    print("2. Modelli valutati.")

    # Stampare le metriche di valutazione
    print_metrics(evaluation_metrics)
    print("3. Metriche di valutazione calcolate.")

    # Identificare il miglior modello in base a RMSE
    best_model_name = min(evaluation_metrics, key=lambda x: evaluation_metrics[x]['RMSE'])
    print(f"4. Best model: {best_model_name}")

    # Plotting
    # y_pred_cv = evaluation_metrics[best_model_name]['Predictions']
    plot_models_mae(evaluation_metrics)
    for (model_name, model) in models.items():
        plot_learning_curve(model, X, y, kfold, model_name)
    print("5. Grafici generati.")

    print("Predizione completata.")
    print("---------------------------------------\n")

    return models["RandomForestRegressor"] # Per usarlo per Prolog

def train_models(X, y):
    """
    Addestra diversi modelli di regressione e ottimizza gli iperparametri con GridSearchCV.
    
    Args:
        X (DataFrame): Features per la predizione.
        y (Series): Target per la predizione.
    
    Returns:
        tuple: Una tupla contenente:
            dict: Modelli addestrati.
            KFold: K-Fold Cross Validation.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
    }

    param_grids = {
        'LinearRegression': {},  # Nessun iperparametro da ottimizzare
        'SVR': {
            'C': [0.1, 1, 10],  # Prova solo valori più limitati per C
            'kernel': ['linear', 'rbf'],  # Elimina il polinomiale per velocizzare
            'gamma': ['scale', 0.1],  # Gamma: valore più semplice
            'epsilon': [0.1, 0.2]  # Usa epsilon fisso per ridurre la ricerca
        },
        'RandomForestRegressor': {
            'n_estimators': [100, 200],  # Più alberi per migliorare le performance.
            'max_depth': [None, 5, 10],  # Profondità degli alberi.
            'min_samples_split': [2, 5, 10],  # Numero minimo di campioni per dividere un nodo.
            'min_samples_leaf': [1, 2],  # Numero minimo di campioni in un nodo foglia.
            'bootstrap': [True, False]  # Se usare il bootstrap (campionamento con ripetizione).
        },
        'GradientBoostingRegressor': {
            'n_estimators': [100, 150],  # Numero di alberi.
            'learning_rate': [0.01, 0.1],  # Tasso di apprendimento, che controlla la velocità di aggiornamento.
            'max_depth': [None, 5, 10],  # Profondità massima di ciascun albero.
            'subsample': [0.1, 0.5],  # Percentuale di campioni da usare per ogni albero.
            'min_samples_split': [2, 5],  # Numero minimo di campioni richiesti per fare uno split.
            'min_samples_leaf': [1, 2],  # Numero minimo di campioni in un nodo foglia.
            'loss': ['squared_error']  # Funzione di perdita da minimizzare.
        }
    }

    best_models = {}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        print(f"Addestramento del modello {model_name}...")
        
        # GridSearch con K-Fold Cross Validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=kfold,  # K-Fold Cross Validation
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # Addestra con la K-Fold Cross Validation (ogni fold è utilizzato come set di test)
        grid_search.fit(X, y) # Addestra su tutti i dati, applicando la K-Fold
        best_models[model_name] = grid_search.best_estimator_
        print(f"\tModello migliore trovato con i seguenti parametri: {grid_search.best_estimator_.get_params()}")

    return best_models, kfold

def evaluate_models(models, X, y, kfold):
    """
    Valuta i modelli addestrati con K-Fold Cross Validation.
    
    Utilizza le metriche di valutazione MAE, MSE e RMSE.
    
    Args:
        models (dict): Modelli addestrati.
        X (DataFrame): Features per la predizione.
        y (Series): Target per la predizione.
        kfold (KFold): K-Fold Cross Validation.
    
    Returns:
        dict: Metriche di valutazione per ogni modello.
    """
    evaluation_metrics = {}
    print("\nValutazione dei modelli:")

    for model_name, model in models.items():
        print(f"Modello {model_name}...")

        # Inizializza liste per metriche e predizioni
        fold_mae, fold_mse, fold_rmse = [], [], []
        y_pred_all = np.zeros(len(y)) # Placeholder per le predizioni finali

        for train_index, test_index in kfold.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Allena il modello e genera predizioni
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Salva le predizioni nei rispettivi indici
            y_pred_all[test_index] = y_pred

            # Calcola le metriche per il fold corrente
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5

            fold_mae.append(mae)
            fold_mse.append(mse)
            fold_rmse.append(rmse)

        # Memorizza i risultati per ogni modello
        evaluation_metrics[model_name] = {
            'MAE': np.mean(fold_mae),
            'MSE': np.mean(fold_mse),
            'RMSE': np.mean(fold_rmse),
            'Predictions': y_pred_all # Tutte le predizioni combinate
        }

    return evaluation_metrics

def print_metrics(evaluation_metrics):
    """
    Stampa le metriche di valutazione dei modelli in una tabella.
    
    Args:
        evaluation_metrics (dict): Metriche di valutazione per ogni modello.
    
    Returns:
        None
    """
    # Converte il dizionario in DataFrame e trasforma l'indice in una colonna (modello)
    tb = pd.DataFrame(evaluation_metrics).T[['MAE', 'MSE', 'RMSE']]
    tb = tb.round(2) # Arrotonda a 2 decimali e formatta in stringa per una visualizzazione più chiara
    print(tb.to_string(), "\n")

# Plot Learning Curve
def plot_learning_curve(model, X, y, kfold, model_name):
    """
    Genera la curva di apprendimento per un modello specifico.
    
    Salva il grafico in documentazione/res/drawable/img_supervised/learning_curve_{model_name}.png.
    
    Args:
        model: Modello addestrato.
        X (DataFrame): Features per la predizione.
        y (Series): Target per la predizione.
        kfold (KFold): K-Fold Cross Validation.
        model_name (str): Nome del modello.
    
    Returns:
        None
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    # Poiché il punteggio è negativo, inverto il segno per ottenere il MAE
    train_mae = -np.mean(train_scores, axis=1)
    test_mae = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(10,6))
    plt.plot(train_sizes, train_mae, label='Training MAE', marker='o', color='blue')
    plt.plot(train_sizes, test_mae, label='Cross-Validation MAE', marker='s', color='red')
    plt.title(f'Learning Curve ({model_name}, MAE)')
    plt.xlabel('Numero di esempi di training')
    plt.ylabel('Errore Medio Assoluto (MAE)')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"documentazione/res/drawable/img_supervised/learning_curve_{model_name}.png")
    plt.close()
    print(f"Salvato in documentazione/res/drawable/img_supervised/learning_curve_{model_name}.png")

def plot_models_mae(evaluation_metrics):
    """
    Crea un grafico a barre per confrontare le performance dei modelli in base al MAE.
    
    Salva il grafico in documentazione/res/drawable/img_supervised/mae_models.png.
    
    Args:
        evaluation_metrics (dict): Metriche di valutazione per ogni modello.
    
    Returns:
        None
    """
    # Estrai i nomi dei modelli e i corrispondenti MAE
    models = list(evaluation_metrics.keys())
    mae_values = [evaluation_metrics[m]['MAE'] for m in models]
    
    # Crea il grafico a barre
    plt.figure(figsize=(8, 6))
    plt.bar(models, mae_values, color='skyblue')
    plt.xlabel('Modelli')
    plt.ylabel('MAE')
    plt.title('Confronto delle performance dei modelli (MAE)')
    plt.ylim(0, max(mae_values) * 1.2)  # Imposta un limite superiore per migliorare la visualizzazione
    # plt.show()
    plt.savefig("documentazione/res/drawable/img_supervised/mae_models.png")
    plt.close()
    print(f"Salvato in documentazione/res/drawable/img_supervised/mae_models.png")

# if __name__ == "__main__":
#     main()
