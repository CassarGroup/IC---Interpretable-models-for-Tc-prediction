# import sys

# sys.path.append(
#    r"C:\Users\julia24002\OneDrive - ILUM ESCOLA DE CIÊNCIA\Iniciação Científica\IC---Interpretable-models-for-Tc-prediction\Main_Scripts"
#)

import os
import pickle
# import pandas as pd
import time
import numpy as np
import optuna
import statsmodels.api as sm
from optuna import create_study
from sklearn.cluster import (
    AffinityPropagation,
    Birch,
    BisectingKMeans,
    KMeans,
    MeanShift,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from script_sr import Clustering_SR, train_validation


now = time.localtime()
data = time.strftime("%d_%m_%Y_%H_%M_%S", now)

TEST_SIZE = 0.1
NUM_FOLDS = 5
N_TRIALS = 100
DATA_FILE = "superconductivity_data.pkl"


def create_instance(trial):
    """Create a instance of Clustering_GLM"""

    clusterer_type = trial.suggest_categorical(
        "clusterer",
        ["kmeans", "affinity_propagation", "mean_shift", "bisecting_kmeans"],
    )

    if clusterer_type == "kmeans":
        n_clusters = trial.suggest_int("n_clusters", 1, 10)
        clusterer = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=1203,
        )

    elif clusterer_type == "affinity_propagation":
        damping = trial.suggest_float("damping", 0.5, 1.0)
        # affinity = trial.suggest_categorical("affinity", ["euclidean", "precomputed"])
        clusterer = AffinityPropagation(damping=damping)

    elif clusterer_type == "mean_shift":
        bandwidth = trial.suggest_float("bandwidth", 0.8, 0.9)
        clusterer = MeanShift(bandwidth=bandwidth)

    elif clusterer_type == "bisecting_kmeans":
        n_clusters = trial.suggest_int("n_clusters_bkmeans", 1, 20)
        bisecting_strategy = trial.suggest_categorical(
            "bisecting_strategy", ["biggest_inertia", "largest_cluster"]
        )
        clusterer = BisectingKMeans(
            n_clusters=n_clusters, bisecting_strategy=bisecting_strategy
        )

    elif clusterer_type == "birch":
        n_clusters = trial.suggest_int("n_clusters_birch", 1, 10)
        branching_factor = trial.suggest_int("branching_factor", 50, 1000)
        threshold = trial.suggest_float("threshold", 0.1, 1)
        clusterer = Birch(
            n_clusters=n_clusters,
            branching_factor=branching_factor,
            threshold=threshold,
        )

    n_iterations = trial.suggest_int("n_iterations", 10, 50)
    maxsize = trial.suggest_int("maxsize", 10, 50)
    maxdepth = trial.suggest_int("maxdepth", 1, 15)

    select_k_features = trial.suggest_int("n_features", 10, 80)

    return Clustering_SR(clusterer=clusterer, 
                              n_iterations=n_iterations, 
                              maxsize=maxsize,
                              maxdepth=maxdepth, 
                              select_k_features=select_k_features)

def make_objective(X_train, y_train):
    """Calculates the objective function"""

    def objective(trial):
        model = create_instance(trial)

        try:
            score = train_validation(
                    X=X_train,
                    y=y_train,
                    clusterer=model.clusterer,
                    n_iterations=model.n_iterations,  
                    maxsize=model.maxsize,
                    maxdepth=model.maxdepth,                   
                    select_k_features=model.select_k_features
                )

        except ValueError:  # FloatingPointError):
            raise optuna.exceptions.TrialPruned()
            
        return score

    return objective


def optimization(X_train, y_train):
    """Make the Optuna study"""

    clusters_study = create_study(
        direction="minimize",
        study_name=f"optimization_clusters_sr_teste{data}",
        storage=f"sqlite:///../Optuna_files/optimization_clusters_sr_teste{data}.db",
        load_if_exists=True,
    )

    objective_fn = make_objective(X_train, y_train)
    clusters_study.optimize(objective_fn, n_trials=N_TRIALS)

    best_trial = clusters_study.best_trial

    parameters_best_trial = best_trial.params
    return parameters_best_trial


superconductivity_data = fetch_ucirepo(id=464)

    # Data treatment and splitting
if __name__ == "__main__":

    # Loading data
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            superconductivity_data = pickle.load(f)
    else:
        superconductivity_data = fetch_ucirepo(id=464)
        with open(DATA_FILE, "wb") as f:
            pickle.dump(superconductivity_data, f)

    # Data treatment and splitting
    X = superconductivity_data.data.features
    y = superconductivity_data.data.targets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=1702
    )
    y_test = np.clip(y_test, 1e-6, None)
    y_train = np.clip(y_train, 1e-6, None)

    # Optimization
    resultado = optimization(X_train, y_train)
