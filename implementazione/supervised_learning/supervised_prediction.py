import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold, learning_curve, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
np.set_printoptions(suppress=True) # Per evitare che i numeri vengano stampati in notazione scientifica

# Per la previsione:
# - y_pred = gridsearch.best_model_.predict(X_test)
#   ||
# - y_pred = gridsearch.best_estimator_.predict(X_test)

def main(X, y):
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
    print(f"4. Il miglior modello è: {best_model_name}")

    # Plotting
    y_pred_cv = evaluation_metrics[best_model_name]['Predictions']
    period = X['Year'].astype(str) + '-' + X['Month'].astype(str).str.zfill(2)
    plot_predictions(best_model_name, y, y_pred_cv, period)
    plot_learning_curve(models[best_model_name], X, y, kfold, best_model_name)
    plot_residuals(y, y_pred_cv, best_model_name)
    plot_feature_importance(models[best_model_name], X.columns, best_model_name)
    print("5. Grafici generati.")

    print("Predizione completata.")
    print("---------------------------------------\n")

    return models["RandomForestRegressor"] # Per usarlo per Prolog

def train_models(X, y):
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
    evaluation_metrics = {}
    print("\nValutazione dei modelli:")

    for model_name, model in models.items():
        print(f"Modello {model_name}...")

        # Inizializza liste per metriche e predizioni
        fold_mae, fold_mse, fold_rmse = [], [], []
        y_pred_all = np.zeros(len(y))  # Placeholder per le predizioni finali

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
            'Predictions': y_pred_all  # Tutte le predizioni combinate
        }

    return evaluation_metrics

def print_metrics(evaluation_metrics):
    for model_name, metrics in evaluation_metrics.items():
        print(f"\nModello: {model_name}")
        print(f" - MAE: {metrics['MAE']:.2f}")
        print(f" - MSE: {metrics['MSE']:.2f}")
        print(f" - RMSE: {metrics['RMSE']:.2f}")
    print() # Per leggibilità nella console

# Plot Predizioni vs Valori Reali
def plot_predictions(best_model_name, y, y_pred, period):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16,8))
    plt.plot(period, y, label='Actual Delay', color='blue', marker='o')
    plt.plot(period, y_pred, label='Predicted Delay', color='red', linestyle='--', marker='x')
    plt.title(f'Average Train Delay: Actual vs Predicted ({best_model_name})')
    plt.xlabel('Period')
    plt.ylabel('Delay (min)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot Learning Curve
def plot_learning_curve(model, X, y, kfold, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(10,6))
    plt.plot(train_sizes, train_mean, label='Training Score', marker='o', color='blue')
    plt.plot(train_sizes, test_mean, label='Cross-Validation Score', marker='s', color='red')
    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Negative Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Residuals
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10,6))
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.title(f'Residual Distribution for {model_name}')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals vs Predicted for {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

# Plot Feature Importance
def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, 'feature_importances_'):
        print(f"Il modello {model_name} non supporta la feature importance.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], hue=np.array(feature_names)[indices], palette="viridis", legend=False)
    plt.title(f'Feature Importance for {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

# if __name__ == "__main__":
#     main()