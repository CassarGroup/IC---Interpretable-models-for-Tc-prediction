import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from statsmodels.genmod.families import Gamma, Gaussian, InverseGaussian
from statsmodels.genmod.families.links import log as Log, identity as Identity, inverse_power as InversePower, inverse_squared as InverseSquared
from ucimlrepo import fetch_ucirepo
import os
import pickle
import pandas as pd

TEST_SIZE = 0.1
DATA_FILE = "superconductivity_data.pkl"

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

family_ls = []
link_ls = []
rmse_ls = []

distributions = [Gamma(), Gaussian(), InverseGaussian()]
links = [Log(), Identity(), InversePower(), InverseSquared()]

best_model = None
best_rmse = np.inf

for dist in distributions:
    for link in links:
        try:
            family = type(dist)(link=link)
            model = sm.GLM(y_train, X_train, family=family)
            result = model.fit()
                
            y_pred = result.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_pred)
        
            family_ls.append(family.__class__.__name__)
            link_ls.append(link.__class__.__name__)
            rmse_ls.append(rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = (dist, link, result)

        except Exception as e:
            print(f"Erro com {dist}, {link}: {e}")
            continue 

if best_model:
    dist, link, result = best_model
    print(f"Melhor modelo: {dist.__class__.__name__} com link {link.__class__.__name__}")
    print(f"RMSE: {best_rmse} K")

results = {"Family" : family_ls,
            "Link": link_ls,
            "RMSE": rmse_ls}

df = pd.DataFrame(results)
df.to_csv("gridsearch_glm_results.csv")
