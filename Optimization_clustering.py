import sys
sys.path.append(r'C:\Users\julia24002\OneDrive - ILUM ESCOLA DE CIÊNCIA\Iniciação Científica\IC---Interpretable-models-for-Tc-prediction')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import optuna
from optuna import create_study

from script_glm import Clustering_GLM
import statsmodels.api as sm

def create_instance(trial):
    """Create a instance of Clustering_GLM"""
    
    clusterer_type = trial.suggest_categorical("clusterer", ["kmeans"])
    
    if clusterer_type == "kmeans":
        n_clusters = trial.suggest_int("n_clusters", 1, 10)
        clusterer = KMeans(n_clusters=n_clusters, init='k-means++')
        
    # else:
    #     eps = trial.suggest_float("eps", 0.05, 1.0)
    #     min_samples = trial.suggest_int("min_samples", 3, 10)
    #     clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    distribution_name = trial.suggest_categorical("distribution", ["gamma", "gaussian"])
    
    link = {
        "log": sm.families.links.Log(),
        "identity": sm.families.links.Identity(),
        "inverse": sm.families.links.InversePower(),
    }   
    
    if distribution_name == "gamma":
        link_name = trial.suggest_categorical("link", ["identity", "log"])  
        family = sm.families.Gamma(link=link[link_name])
    elif distribution_name == "gaussian":
        link_name = trial.suggest_categorical("link", ["identity", "log"])
        family = sm.families.Gaussian(link=link[link_name])

    return Clustering_GLM(clusterer, distribution=family)


def make_objective(X_train, y_train):
    """Calculates the objective function"""
    def objective(trial):
        model = create_instance(trial)
        model.fit(X_train, y_train)
        return model.cross_validation()["Mean"]
    return objective

def optimization(X_train, y_train):
    """Make the Optuna study"""
    clusters_study = create_study(
            direction="minimize",
            study_name="optimization_clusters_glm11",
            storage=f"sqlite:///optimization_clusters_glm11.db",
            load_if_exists=True,
        )

    objective_fn = make_objective(X_train, y_train)
    clusters_study.optimize(objective_fn, n_trials=100)

    best_trial = clusters_study.best_trial

    parameters_best_trial = best_trial.params
    return parameters_best_trial