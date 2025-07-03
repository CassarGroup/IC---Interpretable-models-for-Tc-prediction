import sys
sys.path.append(r'C:\Users\julia24002\OneDrive - ILUM ESCOLA DE CIÊNCIA\Iniciação Científica\IC---Interpretable-models-for-Tc-prediction')
from script_glm import Clustering_GLM

def cria_instancia_modelo_knn(trial):
    """Cria uma instância do modelo desejado (kNN)"""
    
    parametros = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 50), 
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]), 
        }
        
    modelo = KNeighborsRegressor(**parametros)

    return modelo

def funcao_objetivo(trial, X, y, num_folds):
    """Computa o RMSE - com a utilização de validação cruzada - para teste a eficiência das instâncias geradas """

    modelo = cria_instancia_modelo_knn(trial) 

    metricas = cross_val_score(
            modelo,
            X,
            y,
            scoring="neg_root_mean_squared_error",
            cv=num_folds,
        )
    return -metricas.mean()

def funcao_objetivo_parcial(trial):
    "Função objetivo que apenas possui como argumento o objeto trial"
    return funcao_objetivo(trial, X_treino, y_treino, 10)

estudo_knn = create_study(
        direction="minimize",
        study_name="k_nn_dataset_condutores_optuna",
        storage=f"sqlite:///k_nn_dataset_condutores_optuna.db",
        load_if_exists=True,
    )

estudo_knn.optimize(funcao_objetivo_parcial, n_trials=100)

melhor_trial_knn = estudo_knn.best_trial

parametros_melhor_trial_knn = melhor_trial_knn.params
print(f"Parâmetros do melhor trial: {parametros_melhor_trial_knn}")