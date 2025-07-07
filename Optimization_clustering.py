import sys

sys.path.append(
    r"C:\Users\julia24002\OneDrive - ILUM ESCOLA DE CIÊNCIA\Iniciação Científica\IC---Interpretable-models-for-Tc-prediction"
)


from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    MeanShift,
    BisectingKMeans,
    Birch,
)
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

    clusterer_type = trial.suggest_categorical(
        "clusterer",
        ["kmeans", "affinity_propagation", "mean_shift", "bisecting_kmeans"],
    )

    if clusterer_type == "kmeans":
        n_clusters = trial.suggest_int("n_clusters", 1, 10)
        clusterer = KMeans(n_clusters=n_clusters, init="k-means++", random_state=1203)

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

    distribution_name = trial.suggest_categorical(
        "distribution", ["gamma", "gaussian", "inverse_gaussian"]
    )

    link = {
        "log": sm.families.links.Log(),
        "identity": sm.families.links.Identity(),
        "inverse": sm.families.links.InversePower(),
        "inverse_squared": sm.families.links.InverseSquared(),
    }

    if distribution_name == "gamma":
        link_name = "log"
        family = sm.families.Gamma(link=link["log"])

    elif distribution_name == "gaussian":
        link_name = trial.suggest_categorical("link_gaussian", ["identity", "log"])
        family = sm.families.Gaussian(link=link[link_name])

    elif distribution_name == "inverse_gaussian":
        link_name = trial.suggest_categorical(
            "link_inverse_gaussian", ["log", "inverse_squared"]
        )
        family = sm.families.InverseGaussian(link=link[link_name])

    return Clustering_GLM(clusterer=clusterer, distribution=family)


def make_objective(X_train, y_train):
    """Calculates the objective function"""

    def objective(trial):
        model = create_instance(trial)
        try:
            model.fit(X_train, y_train)
            score = model.cross_validation(X_train, y_train)["Mean"]
        except (ValueError, FloatingPointError):
            raise optuna.exceptions.TrialPruned()
        return score

    return objective


def optimization(X_train, y_train):
    """Make the Optuna study"""
    clusters_study = create_study(
        direction="minimize",
        study_name="optimization_clusters_glm_teste43",
        storage=f"sqlite:///optimization_clusters_glm_teste43.db",
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(),
    )

    objective_fn = make_objective(X_train, y_train)
    clusters_study.optimize(objective_fn, n_trials=100)

    best_trial = clusters_study.best_trial

    parameters_best_trial = best_trial.params
    return parameters_best_trial


superconductivty_data = fetch_ucirepo(id=464)

# Data (dataframe)
X = superconductivty_data.data.features
y = superconductivty_data.data.targets


X_test, X_train, y_test, y_train = train_test_split(
    X, y, test_size=0.9, random_state=1702
)

y_test = (np.clip(y_test, 1e-6, None)).values

y_train = (np.clip(y_train, 1e-6, None)).values
resultado = optimization(X_train, y_train)
