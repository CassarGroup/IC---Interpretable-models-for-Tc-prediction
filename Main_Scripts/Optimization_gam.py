import sys

sys.path.append(
    r"C:\Users\julia24002\OneDrive - ILUM ESCOLA DE CIÊNCIA\Iniciação Científica\IC---Interpretable-models-for-Tc-prediction\Main_Scripts"
)


import os
import pickle

import time
import numpy as np
import optuna
import statsmodels.api as sm
from optuna import create_study

from pygam import s
import pygam 

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.families.links import (
    Identity,
    InversePower,
    InverseSquared,
    Log,
)
from ucimlrepo import fetch_ucirepo


now = time.localtime()
data = time.strftime("%d_%m_%Y_%H_%M_%S", now)

TEST_SIZE = 0.1
NUM_FOLDS = 5
N_TRIALS = 100
DATA_FILE = "superconductivity_data.pkl"


def create_instance(trial):
    """Create a instance of Clustering_GLM"""

    distribution_name = trial.suggest_categorical(
        "distribution", ["gamma", "normal", "inv_gauss"]
    )

    if distribution_name == "gamma":
        link_name = "log"

    elif distribution_name == "normal":
        link_name = trial.suggest_categorical(
            "link_gaussian", ["identity", "log"]
        )

    elif distribution_name == "inv_gauss":
        link_name = trial.suggest_categorical(
            "link_inverse_gaussian", ["log", "inverse_squared"]
        )
    
    lam = trial.suggest_float("lam", 1e-2, 1e2, log=True)
    n_splines = trial.suggest_int("n_splines", 5, 25)

    terms = s(0)

    for i in range(1, 81):
        terms += s(i)
    
    return pygam.pygam.GAM(terms, 
                        distribution=distribution_name, 
                        link=link_name, 
                        lam=lam, 
                        n_splines=n_splines)


def make_objective(X_train, y_train):
    """Calculates the objective function"""
    def train_validation(X, y, distribution_name, link_name, lam, n_splines):

        TEST_SIZE = 0.1
        RANDOM_SEED = 1203
    
        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        terms = s(0)

        for i in range(1, 81):
            terms += s(i)
        
        model = pygam.pygam.GAM(terms,
                        distribution=distribution_name,
                        link=link_name,
                        lam=lam,
                        n_splines=n_splines)
            
        model.fit(X_train, y_train)

        y_pred = model.predict(X_validation)
        rmse = root_mean_squared_error(y_validation, y_pred)

        return rmse

    def objective(trial):
        model = create_instance(trial)

        try:
            score = train_validation(
                    X=X_train,
                    y=y_train,
                    distribution_name=model.distribution,  
                    link_name=model.link,                   
                    lam=model.lam,
                    n_splines=model.n_splines,
                )


        except ValueError:  # , FloatingPointError):
            raise optuna.exceptions.TrialPruned()
            
        return score

    return objective


def optimization(X_train, y_train):
    """Make the Optuna study"""

    gam_study = create_study(
        direction="minimize",
        study_name=f"optimization_gam_teste{data}",
        storage=f"sqlite:///optimization_gam_teste{data}.db",
        load_if_exists=True,
    )

    objective_fn = make_objective(X_train, y_train)
    gam_study.optimize(objective_fn, n_trials=N_TRIALS)

    best_trial = gam_study.best_trial

    parameters_best_trial = best_trial.params
    return parameters_best_trial


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
