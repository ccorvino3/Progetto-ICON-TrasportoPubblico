import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict(dataset, target_column, period_column='Period'):
    # Dividere i dati
    X_train, X_test, y_train, y_test = split_data(dataset, target_column)
    print("1. Dati suddivisi in set di training e di test.")
    
    # Prendere la colonna 'Period' dal dataset e la droppo dal dataset di training e di test di X e y
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
    evaluation_metrics = evaluate_models(models, X_test, y_test)
    print("4. Modelli valutati.")
    
    # Stampare le metriche di valutazione
    print_metrics(evaluation_metrics)
    print("5. Metriche di valutazione calcolate.")
    
    # Generare il grafico per il miglior modello
    # Identificare il miglior modello in base a RMSE
    best_model_name = min(evaluation_metrics, key=lambda x: evaluation_metrics[x]['RMSE'])
    print(f"6. Il miglior modello è: {best_model_name}")
    plot_predictions(best_model_name, y_test, evaluation_metrics[best_model_name]['Predictions'], period_column_values)
    
    print("Predizione completata.")
    print("---------------------------------------\n")

def split_data(dataset, target_column):
    # Selezionare le colonne per le caratteristiche (X)
    features = [
        'Average travel time (min)', 'Number of cancelled trains', 'Number of late trains at departure',
        'Average delay of late departing trains (min)', 'Average delay of all departing trains (min)',
        'Number of trains late on arrival', 'Average delay of late arriving trains (min)', 
        '% trains late due to external causes', '% trains late due to railway infrastructure', 
        '% trains late due to traffic management', '% trains late due to rolling stock',
        '% trains late due to station management and reuse of material', '% trains late due to passenger traffic',
        'Number of late trains > 15min', 'Average train delay > 15min', 'Number of late trains > 30min',
        'Number of late trains > 60min', 'Period'
    ]

    # Suddividere i dati in caratteristiche (X) e target (y)
    X = dataset[features]
    y = dataset[target_column]
    
    # Dividere i dati in set di training e di test
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    models = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
    }

    param_grids = {
        'LinearRegression': {},  # Nessun iperparametro da ottimizzare
        'SVR': {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100]
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 120],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 5, 10]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9]
        }
    }

    best_models = {}
    for model_name, model in models.items():
        param_grid = param_grids[model_name]
        print(f"Addestramento del modello {model_name}...")

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

def evaluate_models(models, X_test, y_test):
    evaluation_metrics = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        cross_val_r2 = cross_val_score(model, X_test, y_test, cv=5, scoring='r2').mean()
        evaluation_metrics[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'CrossVal_R2': cross_val_r2,
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
