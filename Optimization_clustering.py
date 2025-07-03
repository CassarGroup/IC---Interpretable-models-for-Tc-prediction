import sys
sys.path.append(r'C:\Users\julia24002\OneDrive - ILUM ESCOLA DE CIÊNCIA\Iniciação Científica\IC---Interpretable-models-for-Tc-prediction')

from sklearn.cluster import KMeans, DBSCAN
import optuna
from optuna import create_study

from script_glm import Clustering_GLM
import statsmodels as sm

def create_instance(trial):
    """Create a instance of Clustering_GLM"""
    
    clusterer_type = trial.suggest_categorical("clusterer", ["kmeans", "dbscan"])
    
    if clusterer_type == "kmeans":
        n_clusters = trial.suggest_int("n_clusters", 1, 10)
        clusterer = KMeans(n_clusters=n_clusters)
    else:
        eps = trial.suggest_float("eps", 0.05, 1.0)
        min_samples = trial.suggest_int("min_samples", 3, 10)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    distribution_name = trial.suggest_categorical("distribution", ["gamma", "gaussian", "inverse_gaussian", "tweedie"])
    link_name = trial.suggest_categorical("link", ["log", "identity", "inverse_power", "inverse_squared"])

    link = {
        "log": sm.families.links.Log(),
        "identity": sm.families.links.Identity(),
        "inverse_power": sm.families.links.InversePower(),
        "inverse_squared": sm.families.links.InverseSquared(),
    }[link_name]

    family = {
        "gamma": sm.families.Gamma(link=link),
        "gaussian": sm.families.Gaussian(link=link),
        "inverse_gaussian": sm.families.InverseGaussian(link=link),
        "tweedie": sm.families.Tweedie(link=link)
    }[distribution_name]

    return Clustering_GLM(clusterer, distribution=family)


def make_objective(X_train, y_train):
    def objective(trial):
        model = create_instance(trial)
        model.fit(X_train, y_train)
        return model.cross_validation()["Mean"]
    return objective

def optimization(X_train, y_train):

    clusters_study = create_study(
            direction="minimize",
            study_name="clusters_optimization_glm",
            storage=f"sqlite:///clusters_optimization_glm.db",
            load_if_exists=False,
        )

    objective_fn = make_objective(X_train, y_train)
    clusters_study.optimize(objective_fn, n_trials=100)

    best_trial = clusters_study.best_trial

    parameters_best_trial = best_trial.params
    return parameters_best_trial