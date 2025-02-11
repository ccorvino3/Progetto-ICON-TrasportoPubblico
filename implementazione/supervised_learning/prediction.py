import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict(dataset, features, target, period_column='Period'):
    print("Predizione in corso...")

    # Dividere i dati
    X_train, X_test, y_train, y_test = split_data(dataset, features, target)
    print("1. Dati suddivisi in set di training e di test.")

    # Prendo 'Period' per il grafico
    period_column_values = X_test[period_column]
    X_test = X_test.drop(columns=[period_column])
    X_train = X_train.drop(columns=[period_column])
    y_train = y_train.drop(columns=[period_column])
    y_test = y_test.drop(columns=[period_column])
    print("2. Colonna 'Period' rimossa dai set di training e di test.")

    # Addestrare i modelli
    models = train_models(X_train, y_train)
    print("3. Modelli addestrati.")

    # Valutare i modelli
    evaluation_metrics = evaluate_models(models, X_test, y_test, X_train, y_train)
    print("4. Modelli valutati.")

    # Stampare le metriche di valutazione
    print_metrics(evaluation_metrics)
    print("5. Metriche di valutazione calcolate.")

    # Generare il grafico per il miglior modello
    # Identificare il miglior modello in base a RMSE
    best_model_name = min(evaluation_metrics, key=lambda x: evaluation_metrics[x]['RMSE'])
    print(f"6. Il miglior modello è: {best_model_name}")
    # plot_predictions(best_model_name, y_test, evaluation_metrics[best_model_name]['Predictions'], period_column_values)
    # plot_learning_curve(models[best_model_name], X_train, y_train, best_model_name)

    print("Predizione completata.")
    print("---------------------------------------\n")

    return models["RandomForestRegressor"] # Per usarlo per Prolog

def split_data(dataset, features, target_column):
    # Suddividere i dati in caratteristiche (X) e target (y)
    X = dataset[features]
    y = dataset[target_column]

    # Dividere i dati in set di training e di test
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    models = {
        "LinearRegression": LinearRegression(),
        #"SVR": SVR(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
    }

    param_grids = {
        'LinearRegression': {},  # Nessun iperparametro da ottimizzare
        # 'SVR': {
        #     'C': [0.1, 1, 10],  # Prova solo valori più limitati per C
        #     'kernel': ['linear', 'rbf'],  # Elimina il polinomiale per velocizzare
        #     'gamma': ['scale', 0.1],  # Gamma: valore più semplice
        #     'epsilon': [0.1]  # Usa epsilon fisso per ridurre la ricerca
        # },
        'RandomForestRegressor': {
            'n_estimators': [50, 100],  # Più alberi per migliorare le performance.
            'max_depth': [10, 20, None],  # Profondità degli alberi.
            'min_samples_split': [2, 5],  # Numero minimo di campioni per dividere un nodo.
            'min_samples_leaf': [1, 2],  # Numero minimo di campioni in un nodo foglia.
            'max_features': ['sqrt', 'log2'],  # Numero di caratteristiche da considerare per la divisione.
            'bootstrap': [True, False]  # Se usare il bootstrap (campionamento con ripetizione).
        },
        'GradientBoostingRegressor': {
            'n_estimators': [50, 100],  # Numero di alberi.
            'learning_rate': [0.05, 0.1],  # Tasso di apprendimento, che controlla la velocità di aggiornamento.
            'max_depth': [3, 4],  # Profondità massima di ciascun albero.
            'subsample': [0.8, 1.0],  # Percentuale di campioni da usare per ogni albero.
            'min_samples_split': [2],  # Numero minimo di campioni richiesti per fare uno split.
            'min_samples_leaf': [1, 2],  # Numero minimo di campioni in un nodo foglia.
            'loss': ['squared_error']  # Funzione di perdita da minimizzare.
        }
    }

    best_models = {}
    for model_name, model in models.items():
        param_grid = param_grids[model_name]
        print(f"Addestramento del modello {model_name}...")
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cross_val_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2', n_jobs=-1)
        mean_cross_val_score = cross_val_scores.mean()
        print(f"\tK-Fold R² medio: {mean_cross_val_score:.4f}")

        if param_grid:  # Se ci sono iperparametri da ottimizzare
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        else:  # Se il modello non ha parametri da ottimizzare (LinearRegression)
            model.fit(X_train, y_train)
            best_model = model

        best_models[model_name] = best_model
        print(f"\tAddestrato con i migliori parametri: {best_model}")

    return best_models

def evaluate_models(models, X_test, y_test, X_train, y_train):
    evaluation_metrics = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        cross_val_r2 = cross_val_score(model, X_test, y_test, cv=5, scoring='r2').mean()
        cross_val_r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        mean_cross_val_r2 = cross_val_r2_scores.mean()
        evaluation_metrics[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'CrossVal_R2': cross_val_r2,
            'Cross_val_train': cross_val_r2_scores,
            'Mean_CrossVal_R2': mean_cross_val_r2,
            'Predictions': y_pred
        }
    return evaluation_metrics

def print_metrics(evaluation_metrics):
    for model_name, metrics in evaluation_metrics.items():
        print(f"\nModello: {model_name}")
        print(f" - MAE: {metrics['MAE']:.2f}")
        print(f" - MSE: {metrics['MSE']:.2f}")
        print(f" - RMSE: {metrics['RMSE']:.2f}")
        print(f" - R²: {metrics['R2']:.4f}")
        print(f" - Cross-Val R²: {metrics['CrossVal_R2']:.4f}")
        
        print("\nK-Fold Cross Validation:")
        for i, score in enumerate(metrics['Cross_val_train']):
            print(f"  Fold {i+1}: R² = {score:.4f}")
        print(f"  Media Cross-Val R²: {metrics['Mean_CrossVal_R2']:.4f}")

        print()

def plot_predictions(best_model_name, y_test, y_pred, period_column_values):
    # Creare un DataFrame con i valori reali e previsti
    df_results = pd.DataFrame({
        'Actual Delay': y_test,
        'Predicted Delay': y_pred
    })

    # Impostare lo stile di Seaborn
    sns.set_theme(style="whitegrid")

    # Creare il grafico a linee
    plt.figure(figsize=(16, 8))
    plt.plot(period_column_values, df_results['Actual Delay'], label='Actual Delay', color='b', marker='o')
    plt.plot(period_column_values, df_results['Predicted Delay'], label='Predicted Delay', color='r', linestyle='--', marker='x')
    plt.title(f'Average Train Delay: Actual vs Predicted ({best_model_name})')
    plt.xlabel('Date')
    plt.ylabel('Delay (min)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_learning_curve(model, X_train, y_train, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)  # Prova con frazioni del dataset
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', marker='o', color='blue')
    plt.plot(train_sizes, test_mean, label='Cross-Validation Score', marker='s', color='red')

    plt.title(f'Learning Curve per {model_name}')
    plt.xlabel('Numero di Esempi di Training')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid()
    plt.show()
