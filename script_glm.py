import statistics as st

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def load_dataset():
    """Import the dataset from UCI repository"""

    # Import dataset
    superconductivty_data = fetch_ucirepo(id=464)

    # Data (dataframe)
    X = superconductivty_data.data.features
    y = superconductivty_data.data.targets

    return X, y


class Clustering_GLM(BaseEstimator, RegressorMixin):
    def __init__(self, clusterer, distribution):
        self.clusterer = clusterer
        self.distribution = distribution

    def fit(self, X, y):
        """Ajustando um modelo para cada cluster"""

        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        self.X = X
        self.y = y

        # Fit the clustering algorithm
        self.clusterer_ = clone(self.clusterer)

        cluster_labels = self.clusterer_.fit_predict(X)

        self.cluster_labels_ = cluster_labels

        unique_clusters = np.unique(cluster_labels)
        valid_clusters = [c for c in unique_clusters if c != -1]

        self.models_ = {}

        # For each cluster, fit a supervised model
        self.data_by_cluster_ = {}

        for cluster in valid_clusters:  # identify the classes
            idx = np.where(cluster_labels == cluster)[0]

            X_cluster = X.iloc[idx]
            y_cluster = y.iloc[idx]

            self.data_by_cluster_[cluster] = {
                "X": X_cluster,
                "y": y_cluster,
                "indices": idx,
            }

            model_glm = sm.GLM(y_cluster, X_cluster, family=self.distribution)
            glm = model_glm.fit()

            self.models_[cluster] = glm

        return self

    def predict(self, X):
        """Realiza previsão considerando cada cluster"""

        X = self.scaler_X_.transform(X)
        cluster_labels = self.clusterer_.predict(X)
        y_pred = np.zeros(X.shape[0])

        for cluster in np.unique(cluster_labels):
            idx = np.where(cluster_labels == cluster)[0]
            if cluster in self.models_:
                y_pred[idx] = self.models_[cluster].predict(X[idx])
            else:
                y_pred[idx] = np.nan

        return y_pred


def cross_validation(X, y, clusterer, distribution, random_seed=1203, n_splits=5):
    """Realiza validação cruzada com clusterização"""

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    all_rmse = {}

    for split, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = Clustering_GLM(clusterer=clone(clusterer), distribution=distribution)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)

        all_rmse[split] = rmse
        
    fold_values = list(all_rmse.values())
    all_rmse["Mean"] = st.mean(fold_values)
    all_rmse["Median"] = st.median(fold_values)

    return all_rmse
