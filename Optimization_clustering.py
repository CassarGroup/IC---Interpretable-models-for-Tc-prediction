import sys
sys.path.append(r'C:\Users\julia24002\OneDrive - ILUM ESCOLA DE CIÊNCIA\Iniciação Científica\IC---Interpretable-models-for-Tc-prediction')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import optuna
from optuna import create_study

from script_glm import Clustering_GLM
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ucimlrepo import fetch_ucirepo

import numpy as np

def create_instance(trial):
    """Create a instance of Clustering_GLM"""
    
    clusterer_type = trial.suggest_categorical("clusterer", ["kmeans"])
    
    if clusterer_type == "kmeans":
        n_clusters = trial.suggest_int("n_clusters", 1, 10)
        clusterer = KMeans(n_clusters=n_clusters, init='k-means++')
        
    else:
        eps = trial.suggest_float("eps", 0.05, 1.0)
        min_samples = trial.suggest_int("min_samples", 3, 10)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    distribution_name = trial.suggest_categorical(
        "distribution", ["gamma", "gaussian", "inverse_gaussian"]
    )
    
    link = {
        "log": sm.families.links.Log(),
        "identity": sm.families.links.Identity(),
        "inverse": sm.families.links.InversePower(),
        "inverse_squared": sm.families.links.InverseSquared()
    }
      
    if distribution_name == "gamma":
        link_name = "log"
        family = sm.families.Gamma(link=link["log"])
                                   
    elif distribution_name == "gaussian":
        link_name = trial.suggest_categorical("link_gaussian", ["identity", "log"])
        family = sm.families.Gaussian(link=link[link_name])
    
    elif distribution_name == "inverse_gaussian":
        link_name = trial.suggest_categorical("link_inverse_gaussian", ["log", "inverse_squared"])
        family = sm.families.InverseGaussian(link=link[link_name])
        

    # link_name = trial.suggest_categorical(
    #     "link", ["log", "identity", "inverse", "inverse_squared"]
    # )


    # if distribution_name == "gamma":
    #     if link_name != "log":
    #         raise optuna.exceptions.TrialPruned()
    #     family = sm.families.Gamma(link=link["log"])

    # elif distribution_name == "gaussian":
    #     if link_name not in ["identity", "log"]:
    #         raise optuna.exceptions.TrialPruned()
    #     family = sm.families.Gaussian(link=link[link_name])

    # elif distribution_name == "inverse_gaussian":
    #     if link_name not in ["log", "inverse_squared"]:
    #         raise optuna.exceptions.TrialPruned()
    #     family = sm.families.InverseGaussian(link=link[link_name])

    # elif distribution_name == "tweedie":
    #     # Tweedie só com link log
    #     if link_name != "log":
    #         raise optuna.exceptions.TrialPruned()
    #     family = sm.families.Tweedie(link=link["log"], var_power=1.5)


    return Clustering_GLM(clusterer=clusterer, distribution=family)

def make_objective(X_train, y_train):
    """Calculates the objective function"""
    def objective(trial):
        model = create_instance(trial)
        try:
            model.fit(X_train, y_train)  # ← pode falhar
            score = model.cross_validation(X_train, y_train)["Mean"] 
        except (ValueError, FloatingPointError):
            raise optuna.exceptions.TrialPruned()
        return score
    return objective


def optimization(X_train, y_train):
    """Make the Optuna study"""
    clusters_study = create_study(
            direction="minimize",
            study_name="optimization_clusters_glm_teste37",
            storage=f"sqlite:///optimization_clusters_glm_teste37.db",
            load_if_exists=True,
        )

    objective_fn = make_objective(X_train, y_train)
    clusters_study.optimize(objective_fn, n_trials=100)

    best_trial = clusters_study.best_trial

    parameters_best_trial = best_trial.params
    return parameters_best_trial


